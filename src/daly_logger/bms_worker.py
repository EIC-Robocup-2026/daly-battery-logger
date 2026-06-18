import asyncio
import time
from collections import deque

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from dalybms import DalyBMSBluetooth
from daly_logger.data_logger import DataLogger

_ESTIMATE_MIN_SAMPLES = 10
_ESTIMATE_SLOPE_THRESHOLD = 0.001  # %/s minimum meaningful rate
_EMA_ALPHA = 0.15
_RATE_ZERO_THRESHOLD = 5e-4  # %/s — suppress extrapolation below this


def _format_minutes(minutes: float | None) -> str | None:
    if minutes is None or minutes <= 0:
        return None
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


class BMSWorker(QThread):
    soc_updated = pyqtSignal(dict)
    cell_range_updated = pyqtSignal(dict)
    temp_updated = pyqtSignal(dict)
    mosfet_updated = pyqtSignal(dict)
    errors_updated = pyqtSignal(list)
    connection_changed = pyqtSignal(str)
    estimates_updated = pyqtSignal(dict)

    def __init__(
        self,
        mac: str,
        db_path: str = "bms_log.db",
        name: str = "",
    ):
        super().__init__()
        self._mac = mac
        self._name = name or mac
        self._logger = DataLogger(db_path, device_id=mac)
        self._stop_event = None
        self._loop = None
        self._bms = None
        self._soc_history: deque = deque(maxlen=120)
        self._last_mode: str | None = None
        self._last_temp: dict | None = None
        self._remaining_capacity_ah: float | None = None
        self._rate_ema: float | None = None
        self._consecutive_reconnects: int = 0
        self._last_estimates: dict = {}

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        self._logger.open()
        try:
            self._loop.run_until_complete(self._poll_loop())
        finally:
            self._logger.close()

    def stop(self):
        if self._loop and self._stop_event:
            self._loop.call_soon_threadsafe(self._stop_event.set)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect_with_retry(self):
        self._bms = DalyBMSBluetooth()
        if self._consecutive_reconnects >= 3:
            delay = 30
        else:
            delay = 2
        while not self._stop_event.is_set():
            try:
                self.connection_changed.emit("connecting")
                await self._bms.connect(self._mac)
                self.connection_changed.emit("connected")
                return
            except Exception as e:
                self.connection_changed.emit(f"reconnecting ({e})")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _reconnect(self):
        self.connection_changed.emit("reconnecting")
        try:
            await self._bms.disconnect()
        except Exception:
            pass
        await self._connect_with_retry()

    # ------------------------------------------------------------------
    # Poll loop
    # ------------------------------------------------------------------

    async def _poll_loop(self):
        await self._connect_with_retry()
        if self._stop_event.is_set():
            return

        slow_counter = 0
        record: dict = {}

        while not self._stop_event.is_set():
            try:
                # --- fast tier (every 1s) ---
                soc = await self._bms.get_soc()
                if soc:
                    voltage = soc.get("total_voltage")
                    current = soc.get("current")
                    soc_pct = soc.get("soc_percent")
                    power = (
                        voltage * current
                        if voltage is not None and current is not None
                        else None
                    )
                    payload = dict(soc, device_id=self._mac, power=power)
                    self.soc_updated.emit(payload)
                    record["ts"] = time.time()
                    record["voltage"] = voltage
                    record["current"] = current
                    record["soc"] = soc_pct
                    record["power"] = power
                    if soc_pct is not None:
                        self._soc_history.append((record["ts"], soc_pct))

                cell_range = await self._bms.get_cell_voltage_range()
                if cell_range:
                    self.cell_range_updated.emit(dict(cell_range, device_id=self._mac))
                    record["cell_min"] = cell_range.get("lowest_voltage")
                    record["cell_max"] = cell_range.get("highest_voltage")

                if record.get("ts"):
                    self._logger.insert(record)

                self._emit_estimates(record.get("soc"), record.get("current"))

                # --- slow tier (every 5s) ---
                slow_counter += 1
                if slow_counter >= 5:
                    slow_counter = 0

                    temp = await self._bms.get_temperature_range()
                    if temp:
                        self._last_temp = temp
                        self.temp_updated.emit(dict(temp, device_id=self._mac))
                        record["temp_min"] = temp.get("lowest_temperature")
                        record["temp_max"] = temp.get("highest_temperature")

                    mosfet = await self._bms.get_mosfet_status()
                    if mosfet:
                        self._last_mode = mosfet.get("mode")
                        self._remaining_capacity_ah = mosfet.get("capacity_ah")
                        self.mosfet_updated.emit(dict(mosfet, device_id=self._mac))
                        record["mode"] = self._last_mode
                        record["capacity_ah"] = self._remaining_capacity_ah

                    errors = await self._bms.get_errors()
                    if errors is not False:
                        err_list = errors if errors else []
                        self.errors_updated.emit(err_list)
                        record["errors"] = err_list

            except Exception:
                self._consecutive_reconnects += 1
                self.connection_changed.emit("reconnecting")
                await self._reconnect()
                slow_counter = 0
                continue

            self._consecutive_reconnects = 0
            await asyncio.sleep(1)

        self.connection_changed.emit("disconnected")
        try:
            await self._bms.disconnect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def _emit_estimates(self, soc_pct: float | None, current: float | None):
        null_result = {"rate_pct_per_min": None, "time_to_full_min": None,
                       "time_to_20_min": None, "time_to_empty_min": None}

        # -- Primary: current-based coulomb counting --
        if (current is not None and soc_pct is not None and soc_pct > 0
                and self._remaining_capacity_ah is not None):
            remaining = self._remaining_capacity_ah
            rated = remaining / (soc_pct / 100.0)

            raw_rate = (current / rated) * 100.0 / 60.0  # %/min
            if self._rate_ema is None:
                self._rate_ema = raw_rate
            else:
                self._rate_ema = _EMA_ALPHA * raw_rate + (1 - _EMA_ALPHA) * self._rate_ema

            rate = self._rate_ema
            result: dict = {"rate_pct_per_min": rate}
            if abs(rate / 60.0) < _RATE_ZERO_THRESHOLD:
                result.update({"time_to_full_min": None, "time_to_20_min": None,
                               "time_to_empty_min": None})
            elif rate > 0:  # charging
                result.update({"time_to_full_min": (rated - remaining) / current * 60.0,
                               "time_to_20_min": None, "time_to_empty_min": None})
            else:  # discharging
                abs_c = abs(current)
                result.update({
                    "time_to_full_min": None,
                    "time_to_20_min": (
                        (remaining - 0.2 * rated) / abs_c * 60.0 if soc_pct > 20 else None
                    ),
                    "time_to_empty_min": remaining / abs_c * 60.0,
                })
            self._last_estimates = result
            self.estimates_updated.emit(result)
            return

        # -- Fallback: SOC linear regression (before mosfet data arrives) --
        self._rate_ema = None
        hist = self._soc_history
        if len(hist) < _ESTIMATE_MIN_SAMPLES or soc_pct is None:
            self._last_estimates = null_result
            self.estimates_updated.emit(null_result)
            return

        ts_arr = np.array([h[0] for h in hist])
        soc_arr = np.array([h[1] for h in hist])
        slope_per_s = np.polyfit(ts_arr - ts_arr[0], soc_arr, 1)[0]  # %/s

        if abs(slope_per_s) < _ESTIMATE_SLOPE_THRESHOLD:
            zero_result = {"rate_pct_per_min": 0.0, "time_to_full_min": None,
                           "time_to_20_min": None, "time_to_empty_min": None}
            self._last_estimates = zero_result
            self.estimates_updated.emit(zero_result)
            return

        rate = slope_per_s * 60.0  # %/min
        if rate > 0:  # charging
            result = {"rate_pct_per_min": rate,
                      "time_to_full_min": (100.0 - soc_pct) / rate,
                      "time_to_20_min": None, "time_to_empty_min": None}
        else:  # discharging
            dr = -rate
            result = {"rate_pct_per_min": rate, "time_to_full_min": None,
                      "time_to_20_min": (soc_pct - 20.0) / dr if soc_pct > 20 else None,
                      "time_to_empty_min": soc_pct / dr}
        self._last_estimates = result
        self.estimates_updated.emit(result)
