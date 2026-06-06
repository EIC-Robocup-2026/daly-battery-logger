"""
Daly BMS web visualization server.

Run with:  uv run python web_server.py
Then open: http://localhost:8765
"""

import asyncio
import json
import math
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

_MAC_RE = re.compile(r'^([0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}$')

def _valid_mac(mac: str) -> bool:
    return bool(_MAC_RE.match(mac or ""))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from daly_logger.data_logger import (
    list_devices,
    query_latest,
    query_new_since,
    query_range,
)

DB_PATH = "bms_log.db"
DEVICES_FILE = "devices.json"
STATIC_DIR = Path(__file__).parent / "static"

# ---------------------------------------------------------------------------
# Device / robot helpers
# ---------------------------------------------------------------------------

def _load_saved() -> list[dict]:
    try:
        return json.loads(Path(DEVICES_FILE).read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _save_saved(devices: list[dict]) -> None:
    try:
        Path(DEVICES_FILE).write_text(json.dumps(devices, indent=2))
    except OSError:
        pass


def _load_robot_mac() -> str | None:
    for d in _load_saved():
        if d.get("robot"):
            return d.get("mac")
    return None


def _set_robot_mac(mac: str | None) -> None:
    saved = _load_saved()
    for d in saved:
        d["robot"] = (d.get("mac") == mac)
    _save_saved(saved)


def _get_devices() -> list[dict]:
    saved = _load_saved()
    name_map = {d["mac"]: d["name"] for d in saved}
    robot_mac = next((d["mac"] for d in saved if d.get("robot")), None)
    try:
        macs = list_devices(DB_PATH)
    except Exception:
        macs = []
    return [{"mac": m, "name": name_map.get(m, m), "robot": m == robot_mac}
            for m in macs if _valid_mac(m)]


# ---------------------------------------------------------------------------
# WebSocket connection manager + background poller
# ---------------------------------------------------------------------------

class _ConnectionManager:
    def __init__(self):
        self._clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self._clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self._clients.discard(ws)

    async def broadcast(self, payload: str):
        dead = set()
        for ws in self._clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.add(ws)
        self._clients -= dead


manager = _ConnectionManager()


def _sanitize(rows: list[dict]) -> list[dict]:
    """Replace NaN/Inf with None so json.dumps doesn't choke."""
    out = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        out.append(clean)
    return out


async def _poll_loop():
    last_ts = time.time() - 5.0
    last_robot_mac = _load_robot_mac()
    while True:
        try:
            rows = [r for r in query_new_since(DB_PATH, last_ts)
                    if _valid_mac(r.get("device_id", ""))]
            if rows:
                last_ts = rows[-1]["ts"]
                await manager.broadcast(json.dumps(_sanitize(rows)))
            current_robot = _load_robot_mac()
            if current_robot != last_robot_mac:
                last_robot_mac = current_robot
                await manager.broadcast(json.dumps({"type": "robot", "mac": current_robot}))
        except Exception:
            pass
        await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_poll_loop())
    yield
    task.cancel()


app = FastAPI(title="Daly BMS Monitor", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/devices")
async def api_devices():
    return _get_devices()


@app.get("/api/latest")
async def api_latest():
    try:
        rows = [r for r in query_latest(DB_PATH)
                if _valid_mac(r.get("device_id", ""))]
    except Exception:
        rows = []
    saved = _load_saved()
    name_map = {d["mac"]: d["name"] for d in saved}
    for row in rows:
        row["name"] = name_map.get(row.get("device_id", ""), row.get("device_id", ""))
    return _sanitize(rows)


@app.get("/api/robot")
async def api_robot():
    return {"mac": _load_robot_mac()}


class _RobotRequest(BaseModel):
    mac: str | None = None


@app.post("/api/robot")
async def api_set_robot(req: _RobotRequest):
    _set_robot_mac(req.mac)
    await manager.broadcast(json.dumps({"type": "robot", "mac": req.mac}))
    return {"ok": True}


@app.get("/api/history")
async def api_history(
    start: float,
    end: float,
    device_id: str | None = None,
):
    try:
        df = query_range(DB_PATH, start, end, device_id=device_id or None)
        if df.empty:
            return []
        rows = df.to_dict(orient="records")
    except Exception:
        rows = []
    return _sanitize(rows)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await manager.connect(websocket)
    # Send the latest row immediately on connect so the UI isn't blank
    try:
        latest = [r for r in query_latest(DB_PATH)
                  if _valid_mac(r.get("device_id", ""))]
        if latest:
            await websocket.send_text(json.dumps(_sanitize(latest)))
    except Exception:
        pass
    try:
        while True:
            await websocket.receive_text()   # keep-alive; client pings
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
