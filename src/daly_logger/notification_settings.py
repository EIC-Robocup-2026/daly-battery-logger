import json
from pathlib import Path

SETTINGS_FILE = "notification_settings.json"

DEFAULT_SETTINGS = {
    "volume": 80,
    "enabled": True,
    "notifications": {
        "low_battery": True,
        "fully_charged": True,
        "charging_started": True,
    },
    "low_battery_threshold": 20,
    "low_battery_min": 5,
}


def load_settings() -> dict:
    try:
        saved = json.loads(Path(SETTINGS_FILE).read_text())
        merged = dict(DEFAULT_SETTINGS)
        merged.update(saved)
        if "notifications" in saved:
            merged["notifications"] = dict(DEFAULT_SETTINGS["notifications"])
            merged["notifications"].update(saved["notifications"])
        return merged
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict) -> None:
    try:
        Path(SETTINGS_FILE).write_text(json.dumps(settings, indent=2))
    except OSError:
        pass
