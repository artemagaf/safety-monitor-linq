"""Десктопное приложение «Пост контроля ОТ».

    python run_desktop.py                      # config.yaml рядом
    python run_desktop.py --config my.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    (ROOT / "data").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "data" / "app.log", encoding="utf-8")])

    from PySide6.QtWidgets import QApplication

    from app.config import load_config
    from desktop.main_window import MainWindow
    from desktop.runtime import Runtime
    from desktop.theme import QSS

    app = QApplication(sys.argv)
    app.setApplicationName("Пост контроля ОТ")
    app.setStyleSheet(QSS)
    rt = Runtime(load_config(args.config))
    win = MainWindow(rt)
    rt.start()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
