# Daly BMS Monitor

A desktop and web monitor for [Daly BMS](https://www.dalybms.com/) (Battery Management System) units connected over Bluetooth LE. Logs data to SQLite, shows live charts, and exposes a web UI over a local WebSocket server.

## Features

- **Live dashboard** — SOC, voltage, current, power, temperature, MOSFET state, and error flags
- **Live charts** — scrolling matplotlib plots with a configurable time window (60–900 s)
- **History tab** — date-range query against the SQLite database with CSV export
- **Web UI** — mirrors the desktop layout; streams live data via WebSocket (no device management)
- **Charge/discharge rate estimates** — coulomb-counting with EMA smoothing; falls back to SOC linear regression on startup
- **Multi-device support** — multiple BMS units can be connected simultaneously, each logged with its own `device_id`

## Requirements

- Python 3.11
- A Daly BMS with Bluetooth LE
- Linux with BlueZ (or macOS with CoreBluetooth via Bleak)

## Installation

```bash
# Install uv if you don't have it
pip install uv

# Install dependencies
uv sync
```

The `dalybms` library is sourced from a Git fork declared in `pyproject.toml` — `uv sync` fetches it automatically.

## Usage

```bash
# Launch desktop GUI
uv run python main.py

# Start web server only (no GUI)
uv run python web_server.py
```

On first launch, a dialog auto-scans for BLE devices for 5 seconds. Select your Daly BMS and click Connect. Use **Change Device** in the status bar to reconnect to a different unit.

## Architecture

```
main.py
└── MainWindow (Qt main thread)
    ├── ConnectDialog / DeviceScanThread  — BLE scan UI
    ├── BMSWorker (QThread + asyncio loop) — BLE polling
    │   ├── Fast tier ~1 s: SOC, cell voltage range → SQLite
    │   └── Slow tier ~5 s: temperature, MOSFET, errors
    ├── DashboardTab   — live text readouts
    ├── LiveChartsTab  — scrolling matplotlib charts
    └── HistoryTab     — date-range query + CSV export

web_server.py (FastAPI + uvicorn)
└── WebSocket broadcast ← BMSWorker signals
    └── static/index.html — browser UI
```

**Data flow:** `BMSWorker` emits Qt signals; `MainWindow` wires them to the GUI tabs and the WebSocket broadcaster. The desktop GUI and web UI share the same live data stream.

**Persistence:** `DataLogger` wraps SQLite in autocommit mode. Schema migrations (new columns) run automatically on startup — existing databases are upgraded in place.

## Project Structure

```
src/daly_logger/
├── bms_worker.py   — BLE polling thread, signal definitions
├── data_logger.py  — SQLite persistence, pandas query helper
├── gui_app.py      — all Qt windows and tabs
├── web_server.py   — FastAPI server, WebSocket broadcast
└── static/
    └── index.html  — browser UI
dalybms/            — vendored fork of python-daly-bms
main.py             — entry point (desktop GUI)
web_server.py       — entry point (web server only)
```

## Data Logged

Each fast-tier poll (~1 s) writes a row to `bms_log.db`:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 UTC |
| `device_id` | BLE MAC address |
| `soc` | State of charge (%) |
| `voltage` | Pack voltage (V) |
| `current` | Current (A, negative = charging) |
| `power` | `voltage × current` (W) |
| `capacity_ah` | Remaining capacity (Ah) |
| `min_cell_v` / `max_cell_v` | Cell voltage range (V) |
