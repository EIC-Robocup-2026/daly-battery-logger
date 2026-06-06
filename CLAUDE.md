# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py

# Debug BLE device discovery (no GUI)
uv run python discover.py

# Debug raw BLE GATT characteristics
uv run python raw_test.py
```

Requires Python 3.11 (see `.python-version`). The `dalybms` dependency is sourced from a local fork at `./dalybms/` via the `[tool.uv.sources]` path in `pyproject.toml`.

## Architecture

The app is a PyQt5 desktop monitor for a Daly BMS (Battery Management System) connected over Bluetooth LE.

**Threading model:** The GUI runs on the main Qt thread. `BMSWorker` (`bms_worker.py`) is a `QThread` that owns its own `asyncio` event loop for all BLE I/O. It communicates back to the GUI exclusively via `pyqtSignal` emissions — never by touching Qt widgets directly.

**Polling tiers in `BMSWorker._poll_loop()`:**
- Fast tier (every ~1 s): SOC + cell voltage range → logged to SQLite each cycle
- Slow tier (every ~5 s): temperature, MOSFET status, errors

**Signal fan-out:** `MainWindow._start_worker()` wires each `BMSWorker` signal to the relevant tab update methods. `soc_updated` feeds both `DashboardTab` and `LiveChartsTab`; `temp_updated` feeds both as well.

**Persistence:** `DataLogger` (`data_logger.py`) wraps a SQLite connection in autocommit mode. `query_range()` (same file) returns a `pandas.DataFrame` for the History tab's date-range queries and CSV export.

**GUI tabs** (all in `gui_app.py`):
- `DashboardTab` — live text readouts for all BMS values
- `LiveChartsTab` — scrolling matplotlib charts backed by `deque(maxlen=900)` ring buffers; window size controlled by a slider (60–900 s)
- `HistoryTab` — date-range picker that queries the database and renders four stacked matplotlib subplots with CSV export

**BLE device selection:** On startup, `ConnectDialog` appears and auto-starts a 5-second BLE scan (`DeviceScanThread` → `BleakScanner`). The selected MAC is passed to `BMSWorker`. "Change Device" in the status bar re-opens this dialog.

**`dalybms/` library:** A vendored fork of `python-daly-bms`. The relevant async API used throughout is `DalyBMSBluetooth` with methods `connect()`, `disconnect()`, `get_soc()`, `get_cell_voltage_range()`, `get_temperature_range()`, `get_mosfet_status()`, `get_errors()`.
