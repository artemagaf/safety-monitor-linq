"""Загрузка config.yaml. Относительные пути считаются от папки, где лежит конфиг."""
import json
from pathlib import Path

import yaml

PATH_KEYS = {
    "model": ["path"],
    "linq": ["model_path", "head_onnx", "tail_onnx", "names_from", "fallback_path"],
    "events": ["db"],
    "reports": ["dir"],
    "segments": ["dir"],
    "fire": ["weights_dir"],
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = path.parent

    for section, keys in PATH_KEYS.items():
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                cfg[section][key] = str(_resolve(base, value))

    if cfg.get("behavior", {}).get("pose_model"):
        cfg["behavior"]["pose_model"] = str(_resolve(base, cfg["behavior"]["pose_model"]))
    for part in cfg["model"].get("parts") or []:   # составной бэкенд: у каждой модели свой путь
        part["path"] = str(_resolve(base, part["path"]))

    # enabled: false — камера выключена, но остаётся в списке: её можно включить в приложении.
    # Что включил или выключил оператор, лежит в data/cameras.json и важнее config.yaml.
    cfg["cameras_file"] = str(_resolve(base, cfg.get("cameras_file", "data/cameras.json")))
    states = {}
    if Path(cfg["cameras_file"]).exists():
        states = json.loads(Path(cfg["cameras_file"]).read_text(encoding="utf-8"))
    for cam in cfg.get("cameras", []):
        cam["enabled"] = bool(states.get(cam["id"], cam.get("enabled", True)))

    for cam in cfg.get("cameras", []):
        src = str(cam["source"])
        # номер веб-камеры и сетевые потоки оставляем как есть
        if not src.isdigit() and "://" not in src:
            cam["source"] = str(_resolve(base, src))
        cam.setdefault("zones", [])
        cam.setdefault("fps", 10)
        cam.setdefault("loop", False)

    cfg["model"]["class_map"] = {k.lower(): v for k, v in cfg["model"].get("class_map", {}).items()}

    # зоны, нарисованные в приложении, лежат отдельно (config.yaml с комментариями не перезаписываем)
    cfg["zones_file"] = str(_resolve(base, cfg.get("zones_file", "data/zones.json")))
    if Path(cfg["zones_file"]).exists():
        overlay = json.loads(Path(cfg["zones_file"]).read_text(encoding="utf-8"))
        for cam in cfg.get("cameras", []):
            if cam["id"] in overlay:
                cam["zones"] = overlay[cam["id"]]
    return cfg


def save_zones(cfg: dict):
    """Сохраняет текущие зоны всех камер в zones_file — вызывается из редактора зон."""
    data = {cam["id"]: cam["zones"] for cam in cfg.get("cameras", [])}
    Path(cfg["zones_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["zones_file"]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def active_cameras(cfg: dict) -> list[dict]:
    return [cam for cam in cfg.get("cameras", []) if cam.get("enabled", True)]


def save_camera_states(cfg: dict):
    """Сохраняет, какие камеры включены, — вызывается из приложения при переключении."""
    data = {cam["id"]: bool(cam.get("enabled", True)) for cam in cfg.get("cameras", [])}
    Path(cfg["cameras_file"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["cameras_file"]).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def _resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p
