import asyncio
import json
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyqtgraph as pg

from PyQt5.QtCore import Qt, QFileSystemWatcher, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QFrame,
    QSizePolicy,
)

from daly_logger.bms_worker import BMSWorker
from daly_logger.data_logger import query_range, list_devices

DB_PATH = "bms_log.db"
DEVICES_FILE = "devices.json"


def _load_saved_devices() -> list[dict]:
    try:
        return json.loads(Path(DEVICES_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_devices(devices: list[dict]) -> None:
    try:
        Path(DEVICES_FILE).write_text(json.dumps(devices, indent=2))
    except OSError:
        pass


pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")

# (key, label, db_column, rgb)
_SERIES = [
    ("soc",     "SOC (%)",     "soc",      (0,   114, 189)),
    ("voltage", "Voltage (V)", "voltage",  (217,  83,  25)),
    ("current", "Current (A)", "current",  (32,  134,  48)),
    ("power",   "Power (W)",   "power",    (126,  47, 142)),
    ("temp",    "Temp (°C)",   "temp_max", (163,  31,  52)),
]

# Colors cycled per device when multiple devices are shown on the same plot
_DEVICE_COLORS = [
    (31,  119, 180),
    (255, 127,  14),
    (44,  160,  44),
    (214,  39,  40),
    (148, 103, 189),
    (140,  86,  75),
    (227, 119, 194),
    (127, 127, 127),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_label(text: str, large: bool = False) -> QLabel:
    lbl = QLabel(text)
    if large:
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        lbl.setFont(f)
    return lbl


def _val_label(large: bool = False) -> QLabel:
    lbl = QLabel("—")
    lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    if large:
        f = QFont()
        f.setPointSize(20)
        f.setBold(True)
        lbl.setFont(f)
    return lbl


def _soc_color(pct):
    if pct is None:
        return ""
    if pct < 20:
        return "color: red;"
    if pct < 50:
        return "color: orange;"
    return "color: green;"


def _fmt_time(minutes: float | None) -> str:
    if minutes is None:
        return "—"
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m:02d}m" if h else f"{m}m"


def _toggle_btn_style(r: int, g: int, b: int) -> str:
    dim = f"rgb({r//2},{g//2},{b//2})"
    return (
        f"QPushButton:checked  {{ background: rgb({r},{g},{b}); color: white; "
        f"border: none; padding: 3px 10px; border-radius: 3px; }}"
        f"QPushButton:!checked {{ background: {dim}; color: #bbb; "
        f"border: none; padding: 3px 10px; border-radius: 3px; }}"
    )


# ---------------------------------------------------------------------------
# Shared chart container: toggle buttons + linked PlotWidgets + crosshair
# ---------------------------------------------------------------------------

class _ChartContainer(QWidget):
    """
    Series-toggle buttons + vertically stacked x-linked PyQtGraph PlotWidgets.
    Supports both single-device (set_data) and multi-device (set_multi_data) modes.
    """

    def __init__(self, x_label: str = "s"):
        super().__init__()
        root = QVBoxLayout(self)
        root.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._toggle_btns: dict[str, QPushButton] = {}
        for key, label, _, rgb in _SERIES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(_toggle_btn_style(*rgb))
            btn.setFixedHeight(24)
            self._toggle_btns[key] = btn
            btn_row.addWidget(btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        chart_area = QWidget()
        self._chart_layout = QVBoxLayout(chart_area)
        self._chart_layout.setSpacing(2)
        self._chart_layout.setContentsMargins(0, 0, 0, 0)
        root.addWidget(chart_area)

        self._plot_widgets: list[pg.PlotWidget] = []
        self._pw_map: dict[str, pg.PlotWidget] = {}
        self._curves: dict[str, pg.PlotDataItem] = {}       # default single-device curve
        self._extra_curves: dict[str, list] = {}             # multi-device curves
        self._legends: dict[str, pg.LegendItem] = {}
        self._vlines: list[pg.InfiniteLine] = []
        self._hlines: list[pg.InfiniteLine] = []

        anchor_pw: pg.PlotWidget | None = None
        for i, (key, label, _, rgb) in enumerate(_SERIES):
            pw = pg.PlotWidget()
            pw.setMinimumHeight(80)
            pw.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setLabel("bottom", x_label)
            pw.setTitle(label, size="9pt")
            color = pg.mkColor(*rgb)
            curve = pw.plot(pen=pg.mkPen(color=color, width=1.5))
            self._curves[key] = curve
            self._pw_map[key] = pw
            self._plot_widgets.append(pw)
            self._chart_layout.addWidget(pw, stretch=1)

            legend = pw.addLegend(offset=(-10, 10))
            legend.hide()
            self._legends[key] = legend

            if anchor_pw is None:
                anchor_pw = pw
            else:
                pw.setXLink(anchor_pw)

            vl = pg.InfiniteLine(angle=90, movable=False,
                                 pen=pg.mkPen("gray", style=Qt.DashLine))
            hl = pg.InfiniteLine(angle=0,  movable=False,
                                 pen=pg.mkPen("gray", style=Qt.DashLine))
            pw.addItem(vl, ignoreBounds=True)
            pw.addItem(hl, ignoreBounds=True)
            self._vlines.append(vl)
            self._hlines.append(hl)

            pw.scene().sigMouseMoved.connect(
                lambda pos, idx=i: self._on_mouse_move(idx, pos)
            )
            self._toggle_btns[key].toggled.connect(
                lambda checked, w=pw: w.setVisible(checked)
            )

    def _on_mouse_move(self, pw_index: int, pos):
        pw = self._plot_widgets[pw_index]
        vb = pw.plotItem.vb
        if pw.sceneBoundingRect().contains(pos):
            mp = vb.mapSceneToView(pos)
            for vl in self._vlines:
                vl.setPos(mp.x())
            self._hlines[pw_index].setPos(mp.y())

    # ------------------------------------------------------------------
    # Public data API
    # ------------------------------------------------------------------

    def set_data(self, key: str, x, y):
        """Single-device mode: one curve per series using the series color."""
        self._clear_extra_curves(key)
        self._curves[key].show()
        self._curves[key].setData(x, y)
        self._legends[key].hide()

    def set_multi_data(self, key: str, datasets: list[tuple]):
        """
        Multi-device mode.
        datasets: [(x_arr, y_arr, display_name, rgb_tuple), ...]
        Each device gets its own colored curve; a legend is shown.
        """
        self._clear_extra_curves(key)
        self._curves[key].setData([], [])   # clear default so it doesn't appear
        pw = self._pw_map[key]
        legend = self._legends[key]
        legend.clear()
        extras = []
        for x, y, name, rgb in datasets:
            c = pw.plot(x, y, pen=pg.mkPen(color=pg.mkColor(*rgb), width=1.5), name=name)
            extras.append(c)
        self._extra_curves[key] = extras
        if extras:
            legend.show()
        else:
            legend.hide()

    def _clear_extra_curves(self, key: str):
        pw = self._pw_map[key]
        for c in self._extra_curves.pop(key, []):
            pw.plotItem.removeItem(c)
        self._legends[key].clear()


# ---------------------------------------------------------------------------
# Tab 1 — Dashboard
# ---------------------------------------------------------------------------

class DashboardTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        soc_group = QGroupBox("State of Charge")
        soc_layout = QHBoxLayout(soc_group)
        self.lbl_soc = _val_label(large=True)
        soc_layout.addWidget(_make_label("SOC", large=True))
        soc_layout.addWidget(self.lbl_soc)
        layout.addWidget(soc_group)

        grid_group = QGroupBox("Pack")
        grid = QGridLayout(grid_group)
        self.lbl_voltage = _val_label()
        self.lbl_current = _val_label()
        self.lbl_power = _val_label()
        self.lbl_cell_min = _val_label()
        self.lbl_cell_max = _val_label()
        self.lbl_temp_min = _val_label()
        self.lbl_temp_max = _val_label()
        self.lbl_mode = _val_label()
        self.lbl_capacity = _val_label()

        rows = [
            ("Voltage (V)", self.lbl_voltage),
            ("Current (A)", self.lbl_current),
            ("Power (W)", self.lbl_power),
            ("Cell Min (V)", self.lbl_cell_min),
            ("Cell Max (V)", self.lbl_cell_max),
            ("Temp Min (°C)", self.lbl_temp_min),
            ("Temp Max (°C)", self.lbl_temp_max),
            ("Mode", self.lbl_mode),
            ("Capacity (Ah)", self.lbl_capacity),
        ]
        for i, (name, widget) in enumerate(rows):
            grid.addWidget(QLabel(name), i, 0)
            grid.addWidget(widget, i, 1)
        layout.addWidget(grid_group)

        est_group = QGroupBox("Estimates")
        est_grid = QGridLayout(est_group)
        self.lbl_rate = _val_label()
        self.lbl_to_full = _val_label()
        self.lbl_to_20 = _val_label()
        self.lbl_to_empty = _val_label()
        est_rows = [
            ("Rate (%/min)", self.lbl_rate),
            ("Time to 100%", self.lbl_to_full),
            ("Time to 20%", self.lbl_to_20),
            ("Time to 0%", self.lbl_to_empty),
        ]
        for i, (name, widget) in enumerate(est_rows):
            est_grid.addWidget(QLabel(name), i, 0)
            est_grid.addWidget(widget, i, 1)
        layout.addWidget(est_group)

        err_group = QGroupBox("Active Errors")
        err_layout = QVBoxLayout(err_group)
        self.error_list = QListWidget()
        self.error_list.setMaximumHeight(100)
        err_layout.addWidget(self.error_list)
        layout.addWidget(err_group)

        layout.addStretch()

    def update_soc(self, data: dict):
        v = data.get("soc_percent")
        self.lbl_soc.setText(f"{v:.1f} %" if v is not None else "—")
        self.lbl_soc.setStyleSheet(_soc_color(v))
        v2 = data.get("total_voltage")
        self.lbl_voltage.setText(f"{v2:.2f}" if v2 is not None else "—")
        c = data.get("current")
        self.lbl_current.setText(f"{c:+.2f}" if c is not None else "—")
        p = data.get("power")
        self.lbl_power.setText(f"{p:+.1f}" if p is not None else "—")

    def update_cell_range(self, data: dict):
        lo = data.get("lowest_voltage")
        hi = data.get("highest_voltage")
        self.lbl_cell_min.setText(f"{lo:.3f}" if lo is not None else "—")
        self.lbl_cell_max.setText(f"{hi:.3f}" if hi is not None else "—")

    def update_temp(self, data: dict):
        lo = data.get("lowest_temperature")
        hi = data.get("highest_temperature")
        self.lbl_temp_min.setText(f"{lo}" if lo is not None else "—")
        self.lbl_temp_max.setText(f"{hi}" if hi is not None else "—")

    def update_mosfet(self, data: dict):
        self.lbl_mode.setText(data.get("mode", "—"))
        cap = data.get("capacity_ah")
        self.lbl_capacity.setText(f"{cap:.1f}" if cap is not None else "—")

    def update_errors(self, errors: list):
        self.error_list.clear()
        if errors:
            for e in errors:
                self.error_list.addItem(e)
        else:
            self.error_list.addItem("No errors")

    def update_estimates(self, data: dict):
        rate = data.get("rate_pct_per_min")
        self.lbl_rate.setText(f"{rate:+.3f}" if rate is not None else "—")
        self.lbl_to_full.setText(_fmt_time(data.get("time_to_full_min")))
        self.lbl_to_20.setText(_fmt_time(data.get("time_to_20_min")))
        self.lbl_to_empty.setText(_fmt_time(data.get("time_to_empty_min")))


# ---------------------------------------------------------------------------
# Tab 2 — Live Charts
# ---------------------------------------------------------------------------

WINDOW_DEFAULT = 300  # seconds


class LiveChartsTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        # Window slider
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Window:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(60)
        self.slider.setMaximum(900)
        self.slider.setValue(WINDOW_DEFAULT)
        self.slider.setTickInterval(60)
        self.slider_lbl = QLabel(f"{WINDOW_DEFAULT}s")
        self.slider.valueChanged.connect(lambda v: self.slider_lbl.setText(f"{v}s"))
        ctrl.addWidget(self.slider)
        ctrl.addWidget(self.slider_lbl)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self._charts = _ChartContainer(x_label="s")
        layout.addWidget(self._charts)

        self._ts: deque = deque(maxlen=900)
        self._bufs: dict[str, deque] = {k: deque(maxlen=900) for k, *_ in _SERIES}
        self._last_temp = None

        self.slider.valueChanged.connect(self._redraw)

    def update_soc(self, data: dict):
        now = time.time()
        self._ts.append(now)
        self._bufs["soc"].append(data.get("soc_percent"))
        self._bufs["voltage"].append(data.get("total_voltage"))
        self._bufs["current"].append(data.get("current"))
        self._bufs["power"].append(data.get("power"))
        self._bufs["temp"].append(self._last_temp)
        self._redraw()

    def update_temp(self, data: dict):
        self._last_temp = data.get("highest_temperature")

    def _redraw(self):
        window = self.slider.value()
        ts = np.array(self._ts)
        if len(ts) == 0:
            return
        cutoff = ts[-1] - window
        mask = ts >= cutoff
        if not mask.any():
            return
        t_rel = ts[mask] - ts[mask][0]

        for key, _, _, _ in _SERIES:
            raw = np.array(self._bufs[key], dtype=object)[mask]
            y = raw.astype(float)
            self._charts.set_data(key, t_rel, y)


# ---------------------------------------------------------------------------
# Tab 3 — History
# ---------------------------------------------------------------------------

class HistoryTab(QWidget):
    def __init__(self, db_path: str):
        super().__init__()
        self._db_path = db_path
        layout = QVBoxLayout(self)

        # Controls row
        ctrl = QHBoxLayout()
        now = datetime.now()
        self.dt_start = QDateTimeEdit(now - timedelta(hours=24))
        self.dt_end = QDateTimeEdit(now)
        for w in (self.dt_start, self.dt_end):
            w.setDisplayFormat("yyyy-MM-dd HH:mm")
            w.setCalendarPopup(True)
        self.device_combo = QComboBox()
        self.device_combo.addItem("All devices", userData=None)
        self.btn_load = QPushButton("Load")
        self.btn_export = QPushButton("Export CSV")
        ctrl.addWidget(QLabel("From:"))
        ctrl.addWidget(self.dt_start)
        ctrl.addWidget(QLabel("To:"))
        ctrl.addWidget(self.dt_end)
        ctrl.addWidget(QLabel("Device:"))
        ctrl.addWidget(self.device_combo)
        ctrl.addWidget(self.btn_load)
        ctrl.addWidget(self.btn_export)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.status_lbl = QLabel("")
        layout.addWidget(self.status_lbl)

        self._charts = _ChartContainer(x_label="min")
        layout.addWidget(self._charts)

        self.btn_load.clicked.connect(self._load)
        self.btn_export.clicked.connect(self._export)
        self._df = None
        self._device_info: list[dict] = []   # [{"mac": ..., "name": ...}]

    def refresh_devices(self, devices: list[dict]):
        """devices: [{"mac": "AA:BB:...", "name": "Pack A"}, ...]"""
        self._device_info = devices
        current = self.device_combo.currentData()
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("All devices", userData=None)
        for d in devices:
            self.device_combo.addItem(d["name"], userData=d["mac"])
        for i in range(self.device_combo.count()):
            if self.device_combo.itemData(i) == current:
                self.device_combo.setCurrentIndex(i)
                break
        self.device_combo.blockSignals(False)

    def _load(self):
        start_ts = float(self.dt_start.dateTime().toSecsSinceEpoch())
        end_ts = float(self.dt_end.dateTime().toSecsSinceEpoch())
        device_id = self.device_combo.currentData()
        # Always load all devices in range; filter per-device in _render if needed
        self._df = query_range(self._db_path, start_ts, end_ts, device_id=device_id)
        if self._df.empty:
            self.status_lbl.setText("No data in selected range.")
            return
        self.status_lbl.setText(
            f"{len(self._df)} rows — drag to zoom, right-click to reset."
        )
        self._render()

    def _render(self):
        df = self._df
        device_id = self.device_combo.currentData()

        if device_id is not None:
            # Single-device: one line per series using series color
            ts = df["ts"].values
            t_min = (ts - ts[0]) / 60.0 if len(ts) else np.array([])
            for key, _, db_col, _ in _SERIES:
                y = df[db_col].values.astype(float) if db_col in df.columns \
                    else np.full(len(t_min), np.nan)
                self._charts.set_data(key, t_min, y)
        else:
            # All devices: one line per device, colored by device
            name_map = {d["mac"]: d["name"] for d in self._device_info}
            macs = df["device_id"].unique() if "device_id" in df.columns else []
            t0 = df["ts"].values[0] if len(df) else 0
            for key, _, db_col, _ in _SERIES:
                datasets = []
                for i, mac in enumerate(macs):
                    sub = df[df["device_id"] == mac]
                    t_min = (sub["ts"].values - t0) / 60.0
                    y = sub[db_col].values.astype(float) if db_col in sub.columns \
                        else np.full(len(t_min), np.nan)
                    name = name_map.get(mac, mac)
                    rgb = _DEVICE_COLORS[i % len(_DEVICE_COLORS)]
                    datasets.append((t_min, y, name, rgb))
                self._charts.set_multi_data(key, datasets)

    def _export(self):
        if self._df is None or self._df.empty:
            self.status_lbl.setText("Load data first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "bms_export.csv", "CSV (*.csv)"
        )
        if path:
            self._df.to_csv(path, index=False)
            self.status_lbl.setText(f"Exported to {path}")


# ---------------------------------------------------------------------------
# Device detail widget (dashboard + live charts for one BMS)
# ---------------------------------------------------------------------------

class DeviceDetailWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        self.dashboard = DashboardTab()
        self.live_charts = LiveChartsTab()
        tabs.addTab(self.dashboard, "Dashboard")
        tabs.addTab(self.live_charts, "Live Charts")


# ---------------------------------------------------------------------------
# Device card (compact overview row)
# ---------------------------------------------------------------------------

class DeviceCard(QFrame):
    view_details_clicked = pyqtSignal(str)
    remove_requested = pyqtSignal(str)
    robot_toggled = pyqtSignal(str, bool)

    def __init__(self, mac: str, name: str):
        super().__init__()
        self._mac = mac
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        root = QVBoxLayout(self)
        root.setSpacing(3)
        root.setContentsMargins(8, 6, 8, 6)

        # Row 1: name + action buttons
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        name_lbl = QLabel(f"<b>{name}</b>")
        row1.addWidget(name_lbl)
        row1.addStretch()
        view_btn = QPushButton("View")
        view_btn.setFixedWidth(52)
        view_btn.setFixedHeight(22)
        view_btn.clicked.connect(lambda: self.view_details_clicked.emit(self._mac))
        row1.addWidget(view_btn)
        self.btn_robot = QPushButton("Robot")
        self.btn_robot.setCheckable(True)
        self.btn_robot.setFixedWidth(52)
        self.btn_robot.setFixedHeight(22)
        self.btn_robot.setStyleSheet(
            "QPushButton:checked  { background: #1565c0; color: white; border: none;"
            " padding: 2px 6px; border-radius: 3px; }"
            "QPushButton:!checked { background: #bbb; color: #555; border: none;"
            " padding: 2px 6px; border-radius: 3px; }"
            "QPushButton:disabled { background: #ddd; color: #aaa; border: none;"
            " padding: 2px 6px; border-radius: 3px; }"
        )
        self.btn_robot.clicked.connect(
            lambda checked: self.robot_toggled.emit(self._mac, checked)
        )
        row1.addWidget(self.btn_robot)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(26)
        remove_btn.setFixedHeight(22)
        remove_btn.setToolTip("Remove device")
        remove_btn.setStyleSheet(
            "QPushButton { color: #c00; border: 1px solid #c00; border-radius: 3px; }"
            "QPushButton:hover { background: #fee; }"
        )
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._mac))
        row1.addWidget(remove_btn)
        root.addLayout(row1)

        # Row 2: MAC + status
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        mac_lbl = QLabel(f"<small>{mac}</small>")
        mac_lbl.setStyleSheet("color: #666;")
        row2.addWidget(mac_lbl)
        self.lbl_status = QLabel("connecting…")
        self.lbl_status.setStyleSheet("color: #888; font-style: italic;")
        row2.addWidget(self.lbl_status)
        row2.addStretch()
        root.addLayout(row2)

        # Row 3: live readings
        row3 = QHBoxLayout()
        row3.setSpacing(12)
        self.lbl_soc = QLabel("SOC: —")
        self.lbl_voltage = QLabel("—V")
        self.lbl_power = QLabel("—W")
        for lbl in (self.lbl_soc, self.lbl_voltage, self.lbl_power):
            row3.addWidget(lbl)
        row3.addStretch()
        root.addLayout(row3)

    def update_connection(self, state: str):
        self.lbl_status.setText(state)
        connected = state == "connected"
        self.lbl_status.setStyleSheet(
            "color: #090; font-style: normal;" if connected
            else "color: #888; font-style: italic;"
        )

    def update_soc(self, data: dict):
        soc = data.get("soc_percent")
        self.lbl_soc.setText(f"SOC: {soc:.1f}%" if soc is not None else "SOC: —")
        v = data.get("total_voltage")
        self.lbl_voltage.setText(f"{v:.2f}V" if v is not None else "—V")
        p = data.get("power")
        self.lbl_power.setText(f"{p:+.1f}W" if p is not None else "—W")

    def set_robot_selected(self, is_robot: bool):
        self.btn_robot.blockSignals(True)
        self.btn_robot.setChecked(is_robot)
        self.btn_robot.blockSignals(False)


# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------

class OverviewTab(QWidget):
    add_device_requested = pyqtSignal()
    view_device_requested = pyqtSignal(str)
    remove_device_requested = pyqtSignal(str)
    robot_toggled = pyqtSignal(str, bool)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("<b>Connected BMS Devices</b>"))
        top.addStretch()
        self.btn_add = QPushButton("+ Add Device")
        self.btn_add.clicked.connect(self.add_device_requested.emit)
        top.addWidget(self.btn_add)
        layout.addLayout(top)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._cards_widget = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_widget)
        self._cards_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(self._cards_widget)
        layout.addWidget(scroll)

        self._cards: dict[str, DeviceCard] = {}

    def add_card(self, mac: str, name: str) -> DeviceCard:
        card = DeviceCard(mac, name)
        card.view_details_clicked.connect(self.view_device_requested.emit)
        card.remove_requested.connect(self.remove_device_requested.emit)
        card.robot_toggled.connect(self.robot_toggled)
        self._cards[mac] = card
        self._cards_layout.addWidget(card)
        return card

    def remove_card(self, mac: str):
        card = self._cards.pop(mac, None)
        if card:
            self._cards_layout.removeWidget(card)
            card.deleteLater()

    def get_card(self, mac: str) -> DeviceCard | None:
        return self._cards.get(mac)


# ---------------------------------------------------------------------------
# BLE scanner
# ---------------------------------------------------------------------------

class DeviceScanThread(QThread):
    device_found = pyqtSignal(str, str, int)
    scan_finished = pyqtSignal()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._scan())

    async def _scan(self):
        from bleak import BleakScanner
        devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
        for dev, adv in devices.values():
            self.device_found.emit(dev.name or "(unknown)", dev.address, adv.rssi or 0)
        self.scan_finished.emit()


class ConnectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Bluetooth Device")
        self.setMinimumWidth(440)
        self.selected_mac = None
        self.selected_name = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Scan for nearby Bluetooth devices and select your BMS:"))

        self.device_list = QListWidget()
        self.device_list.setMinimumHeight(200)
        self.device_list.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self.device_list)

        self.status_lbl = QLabel("Press Scan to discover devices.")
        layout.addWidget(self.status_lbl)

        btn_row = QHBoxLayout()
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self._start_scan)
        btn_row.addWidget(self.scan_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        self.ok_btn = btns.button(QDialogButtonBox.Ok)
        self.ok_btn.setEnabled(False)
        layout.addWidget(btns)

        self.device_list.itemSelectionChanged.connect(
            lambda: self.ok_btn.setEnabled(bool(self.device_list.selectedItems()))
        )
        self._scanner = None

    def _start_scan(self):
        self.device_list.clear()
        self.ok_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.status_lbl.setText("Scanning… (5s)")
        self._scanner = DeviceScanThread()
        self._scanner.device_found.connect(self._add_device)
        self._scanner.scan_finished.connect(self._scan_done)
        self._scanner.start()

    def _add_device(self, name: str, address: str, rssi: int):
        item = QListWidgetItem(f"{name}  —  {address}  ({rssi} dBm)")
        item.setData(Qt.UserRole, (address, name))
        self.device_list.addItem(item)

    def _scan_done(self):
        count = self.device_list.count()
        self.status_lbl.setText(f"Found {count} device(s). Select one and press OK.")
        self.scan_btn.setEnabled(True)

    def _accept(self):
        items = self.device_list.selectedItems()
        if not items:
            return
        address, name = items[0].data(Qt.UserRole)
        self.selected_mac = address
        self.selected_name = name
        self.accept()


# ---------------------------------------------------------------------------
# Notification Settings Dialog
# ---------------------------------------------------------------------------

class NotificationSettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Notification Settings")
        self.setMinimumWidth(400)
        self._settings = dict(settings)

        layout = QVBoxLayout(self)

        self.chk_enabled = QCheckBox("Enable Notifications")
        self.chk_enabled.setChecked(self._settings.get("enabled", True))
        layout.addWidget(self.chk_enabled)

        vol_group = QGroupBox("Volume")
        vol_layout = QHBoxLayout(vol_group)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setMinimum(0)
        self.vol_slider.setMaximum(100)
        self.vol_slider.setValue(self._settings.get("volume", 80))
        self.vol_label = QLabel(f"{self.vol_slider.value()}%")
        self.vol_slider.valueChanged.connect(
            lambda v: self.vol_label.setText(f"{v}%")
        )
        vol_layout.addWidget(self.vol_slider)
        vol_layout.addWidget(self.vol_label)
        layout.addWidget(vol_group)

        notif_group = QGroupBox("Notification Types")
        notif_layout = QVBoxLayout(notif_group)
        notifs = self._settings.get("notifications", {})
        self.chk_low_battery = QCheckBox("Low Battery (Robot, <20%)")
        self.chk_low_battery.setChecked(notifs.get("low_battery", True))
        self.chk_fully_charged = QCheckBox("Fully Charged")
        self.chk_fully_charged.setChecked(notifs.get("fully_charged", True))
        self.chk_charging = QCheckBox("Charging Started")
        self.chk_charging.setChecked(notifs.get("charging_started", True))
        notif_layout.addWidget(self.chk_low_battery)
        notif_layout.addWidget(self.chk_fully_charged)
        notif_layout.addWidget(self.chk_charging)
        layout.addWidget(notif_group)

        threshold_group = QGroupBox("Low Battery Thresholds")
        threshold_layout = QGridLayout(threshold_group)
        threshold_layout.addWidget(QLabel("Alert at:"), 0, 0)
        self.spn_threshold = QSpinBox()
        self.spn_threshold.setRange(5, 50)
        self.spn_threshold.setValue(self._settings.get("low_battery_threshold", 20))
        self.spn_threshold.setSuffix("%")
        threshold_layout.addWidget(self.spn_threshold, 0, 1)
        threshold_layout.addWidget(QLabel("Lowest bound:"), 1, 0)
        self.spn_min = QSpinBox()
        self.spn_min.setRange(1, 15)
        self.spn_min.setValue(self._settings.get("low_battery_min", 5))
        self.spn_min.setSuffix("%")
        threshold_layout.addWidget(self.spn_min, 1, 1)
        layout.addWidget(threshold_group)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._save_and_accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _save_and_accept(self):
        self._settings["enabled"] = self.chk_enabled.isChecked()
        self._settings["volume"] = self.vol_slider.value()
        self._settings["notifications"] = {
            "low_battery": self.chk_low_battery.isChecked(),
            "fully_charged": self.chk_fully_charged.isChecked(),
            "charging_started": self.chk_charging.isChecked(),
        }
        self._settings["low_battery_threshold"] = self.spn_threshold.value()
        self._settings["low_battery_min"] = self.spn_min.value()
        self.accept()

    def get_settings(self) -> dict:
        return self._settings


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Daly BMS Monitor")
        self.resize(1100, 720)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self._top_tabs = QTabWidget()
        root_layout.addWidget(self._top_tabs)

        # Devices tab: splitter — overview list | detail stack
        devices_widget = QWidget()
        devices_layout = QVBoxLayout(devices_widget)
        devices_layout.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)
        devices_layout.addWidget(splitter)

        self._overview = OverviewTab()
        self._overview.setMinimumWidth(480)
        splitter.addWidget(self._overview)

        self._detail_stack = QStackedWidget()
        splitter.addWidget(self._detail_stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        placeholder = QLabel("Select a device on the left to view details.")
        placeholder.setAlignment(Qt.AlignCenter)
        self._detail_stack.addWidget(placeholder)

        self._top_tabs.addTab(devices_widget, "Devices")

        self._history_tab = HistoryTab(DB_PATH)
        self._top_tabs.addTab(self._history_tab, "History")

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._status_bar = status_bar

        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Settings")
        self._action_notifications = QAction("Notifications...", self)
        self._action_notifications.triggered.connect(self._open_notification_settings)
        settings_menu.addAction(self._action_notifications)

        self._overview.add_device_requested.connect(self._on_add_device)
        self._overview.view_device_requested.connect(self._show_device_detail)
        self._overview.remove_device_requested.connect(self._remove_device)
        self._overview.robot_toggled.connect(self._on_robot_toggled)

        self.workers: dict[str, BMSWorker] = {}
        self._details: dict[str, DeviceDetailWidget] = {}
        self._device_names: dict[str, str] = {}
        self._robot_mac: str | None = None

        from daly_logger.notification_settings import load_settings
        from daly_logger.sound_manager import SoundManager
        self._notif_settings = load_settings()
        self._sound_manager = SoundManager()
        self._sound_manager.set_volume(self._notif_settings.get("volume", 80))
        self._notif_state: dict[str, dict] = {}

        self._file_watcher = QFileSystemWatcher()
        if Path(DEVICES_FILE).exists():
            self._file_watcher.addPath(DEVICES_FILE)
        self._file_watcher.fileChanged.connect(self._on_devices_file_changed)

        saved = _load_saved_devices()
        if saved:
            QTimer.singleShot(0, lambda: self._autoconnect(saved))
        else:
            QTimer.singleShot(0, lambda: self._show_connect_dialog(startup=True))

        self._device_refresh_timer = QTimer()
        self._device_refresh_timer.setInterval(30_000)
        self._device_refresh_timer.timeout.connect(self._refresh_history_devices)
        self._device_refresh_timer.start()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def _show_connect_dialog(self, startup: bool = False):
        dlg = ConnectDialog(self)
        if startup:
            dlg._start_scan()
        if dlg.exec_() == QDialog.Accepted and dlg.selected_mac:
            mac = dlg.selected_mac
            name = dlg.selected_name or mac
            if mac in self.workers:
                self._status_bar.showMessage(f"{mac} is already connected.")
                return
            self._add_device(mac, name)
        elif startup:
            self.close()

    def _autoconnect(self, saved: list[dict]):
        for dev in saved:
            mac = dev.get("mac", "")
            name = dev.get("name", mac)
            if mac and mac not in self.workers:
                self._add_device(mac, name)
        for dev in saved:
            mac = dev.get("mac", "")
            if dev.get("robot") and mac in self.workers:
                self._on_robot_toggled(mac, True)

    def _on_add_device(self):
        self._show_connect_dialog(startup=False)

    def _open_notification_settings(self):
        from daly_logger.notification_settings import save_settings
        dlg = NotificationSettingsDialog(self._notif_settings, self)
        if dlg.exec_() == QDialog.Accepted:
            self._notif_settings = dlg.get_settings()
            save_settings(self._notif_settings)
            self._sound_manager.set_volume(self._notif_settings.get("volume", 80))

    def _persist_devices(self):
        _save_devices([
            {"mac": mac, "name": self._device_names.get(mac, mac),
             "robot": mac == self._robot_mac}
            for mac in self.workers
        ])
        # Re-watch after write (atomic replace changes the inode)
        if DEVICES_FILE not in self._file_watcher.files():
            self._file_watcher.addPath(DEVICES_FILE)

    def _add_device(self, mac: str, name: str):
        card = self._overview.add_card(mac, name)

        detail = DeviceDetailWidget()
        self._details[mac] = detail
        self._detail_stack.addWidget(detail)

        worker = BMSWorker(mac, DB_PATH, name=name)

        worker.soc_updated.connect(detail.dashboard.update_soc)
        worker.soc_updated.connect(detail.live_charts.update_soc)
        worker.soc_updated.connect(card.update_soc)
        worker.cell_range_updated.connect(detail.dashboard.update_cell_range)
        worker.temp_updated.connect(detail.dashboard.update_temp)
        worker.temp_updated.connect(detail.live_charts.update_temp)
        worker.mosfet_updated.connect(detail.dashboard.update_mosfet)
        worker.errors_updated.connect(detail.dashboard.update_errors)
        worker.estimates_updated.connect(detail.dashboard.update_estimates)
        worker.connection_changed.connect(
            lambda state, m=mac: self._on_connection(m, state)
        )

        self._notif_state[mac] = {
            "last_soc": None,
            "last_mode": None,
            "notified_low": False,
            "notified_charging": False,
            "notified_full": False,
        }
        worker.soc_updated.connect(lambda data, m=mac: self._check_low_battery(m, data))
        worker.soc_updated.connect(lambda data, m=mac: self._check_fully_charged(m, data))
        worker.mosfet_updated.connect(lambda data, m=mac: self._check_charging_state(m, data))

        self._device_names[mac] = name
        self.workers[mac] = worker
        worker.start()
        self._persist_devices()
        self._refresh_history_devices()

    def _remove_device(self, mac: str):
        if self._robot_mac == mac:
            self._robot_mac = None
        worker = self.workers.pop(mac, None)
        if worker:
            worker.stop()
            worker.wait(5000)
        self._device_names.pop(mac, None)
        self._notif_state.pop(mac, None)
        self._overview.remove_card(mac)
        detail = self._details.pop(mac, None)
        if detail:
            self._detail_stack.removeWidget(detail)
            detail.deleteLater()
        self._persist_devices()

    def _show_device_detail(self, mac: str):
        detail = self._details.get(mac)
        if detail:
            self._detail_stack.setCurrentWidget(detail)

    def _on_connection(self, mac: str, state: str):
        card = self._overview.get_card(mac)
        if card:
            card.update_connection(state)
        name = self._device_names.get(mac, mac)
        self._status_bar.showMessage(f"{name}: {state}")

    def _check_low_battery(self, mac: str, data: dict):
        if mac != self._robot_mac:
            return
        if not self._notif_settings.get("enabled", True):
            return
        if not self._notif_settings.get("notifications", {}).get("low_battery", True):
            return

        soc = data.get("soc_percent")
        state = self._notif_state.get(mac, {})
        threshold = self._notif_settings.get("low_battery_threshold", 20)
        min_bound = self._notif_settings.get("low_battery_min", 5)

        if soc is None:
            return

        if soc < threshold and soc > min_bound and not state.get("notified_low", False):
            time_to_min = data.get("time_to_empty_min")
            name = self._device_names.get(mac, mac)
            if time_to_min is not None:
                h = int(time_to_min) // 60
                m = int(time_to_min) % 60
                time_str = f"{h}h {m:02d}m" if h else f"{m}m"
                msg = f"[NOTIFICATION] Low battery on {name} ({soc:.1f}%). Time to {min_bound}%: {time_str}"
                print(msg)
                self._sound_manager.speak(
                    f"Battery low on {name}. Estimated time to {min_bound} percent: {time_str}."
                )
            else:
                msg = f"[NOTIFICATION] Low battery on {name} ({soc:.1f}%)"
                print(msg)
                self._sound_manager.speak(f"Battery low on {name}.")
            state["notified_low"] = True
        elif soc >= threshold + 5:
            state["notified_low"] = False

    def _check_fully_charged(self, mac: str, data: dict):
        if not self._notif_settings.get("enabled", True):
            return
        if not self._notif_settings.get("notifications", {}).get("fully_charged", True):
            return

        soc = data.get("soc_percent")
        state = self._notif_state.get(mac, {})

        if soc is None:
            return

        if soc >= 100 and not state.get("notified_full", False):
            name = self._device_names.get(mac, mac)
            msg = f"[NOTIFICATION] Battery fully charged on {name}"
            print(msg)
            self._sound_manager.speak(f"Battery fully charged on {name}.")
            state["notified_full"] = True
        elif soc < 95:
            state["notified_full"] = False

    def _check_charging_state(self, mac: str, data: dict):
        if not self._notif_settings.get("enabled", True):
            return
        if not self._notif_settings.get("notifications", {}).get("charging_started", True):
            return

        mode = data.get("mode")
        state = self._notif_state.get(mac, {})
        last_mode = state.get("last_mode")

        if mode is None:
            return

        state["last_mode"] = mode

        if mode == "Charging" and last_mode != "Charging" and not state.get("notified_charging", False):
            name = self._device_names.get(mac, mac)
            worker = self.workers.get(mac)
            time_str = ""
            if worker and worker._remaining_capacity_ah is not None:
                estimates = data
                time_to_full = estimates.get("time_to_full_min")
                if time_to_full is not None:
                    h = int(time_to_full) // 60
                    m = int(time_to_full) % 60
                    time_str = f" {h}h {m:02d}m" if h else f" {m}m"
            msg = f"[NOTIFICATION] Charging started on {name}{time_str}"
            print(msg)
            self._sound_manager.speak(f"Battery now charging on {name}.{time_str}")
            state["notified_charging"] = True
        elif mode != "Charging":
            state["notified_charging"] = False

    def _on_devices_file_changed(self, path: str):
        # Re-add after external atomic write (inode replaced)
        if path not in self._file_watcher.files():
            self._file_watcher.addPath(path)

        saved = _load_saved_devices()
        new_robot_mac = next((d["mac"] for d in saved if d.get("robot")), None)
        if new_robot_mac == self._robot_mac:
            return

        # Deselect old robot without persisting (avoids write loop)
        if self._robot_mac:
            old_card = self._overview.get_card(self._robot_mac)
            if old_card:
                old_card.set_robot_selected(False)

        self._robot_mac = new_robot_mac

        # Select new robot
        if new_robot_mac:
            card = self._overview.get_card(new_robot_mac)
            if card:
                card.set_robot_selected(True)

    def _on_robot_toggled(self, mac: str, is_robot: bool):
        if is_robot:
            if self._robot_mac and self._robot_mac != mac:
                old_card = self._overview.get_card(self._robot_mac)
                if old_card:
                    old_card.set_robot_selected(False)
            self._robot_mac = mac
            card = self._overview.get_card(mac)
            if card:
                card.set_robot_selected(True)
        else:
            if self._robot_mac == mac:
                self._robot_mac = None
            card = self._overview.get_card(mac)
            if card:
                card.set_robot_selected(False)
        self._persist_devices()

    def _refresh_history_devices(self):
        try:
            mac_list = list_devices(DB_PATH)
        except Exception:
            mac_list = []
        # Build name map: saved devices + currently connected (connected takes priority)
        name_map = {d["mac"]: d["name"] for d in _load_saved_devices()}
        name_map.update(self._device_names)
        device_info = [{"mac": m, "name": name_map.get(m, m)} for m in mac_list]
        self._history_tab.refresh_devices(device_info)

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._device_refresh_timer.stop()
        for mac in list(self.workers):
            self._remove_device(mac)
        event.accept()
