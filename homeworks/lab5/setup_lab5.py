"""
Установка окружения для lab05_dqn.

Что делает:
1. pip install -r requirements.txt
2. AutoROM --accept-license  (скачивает ROM-файлы Atari)
3. Качает вспомогательные .py из репо yandexdataschool/Practical_RL
   (atari_wrappers.py, utils.py, replay_buffer.py, framebuffer.py, analysis.py,
    tests/compute_td_loss.py)

Запуск:
    python setup_lab5.py

Если на каком-то шаге упадёт — следующие всё равно выполнятся.
В конце увидишь сводку, что прошло, что нет.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
results: list[tuple[str, bool, str]] = []  # (этап, успех, комментарий)


def step(title: str, fn) -> None:
    print(f"\n=== {title} ===")
    try:
        fn()
        results.append((title, True, "OK"))
    except Exception as e:
        print(f"!!! Шаг провалился: {e}")
        results.append((title, False, str(e)))


def run_cmd(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.check_call(cmd)


# -----------------------------------------------------------------------------
# Шаг 1. Установка Python-пакетов
# -----------------------------------------------------------------------------
def install_requirements() -> None:
    req = ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"Не нашёл {req}")
    run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req)])


# -----------------------------------------------------------------------------
# Шаг 2. Скачивание Atari ROMs через AutoROM
# -----------------------------------------------------------------------------
def install_atari_roms() -> None:
    # AutoROM ставится как отдельная утилита pip-ом; здесь вызываем модуль через sys.executable
    run_cmd([sys.executable, "-m", "AutoROM", "--accept-license"])


# -----------------------------------------------------------------------------
# Шаг 3. Скачивание вспомогательных .py из Practical_RL
# -----------------------------------------------------------------------------
BASE_URL = (
    "https://raw.githubusercontent.com/yandexdataschool/"
    "Practical_RL/master/week04_approx_rl"
)

# Маппинг "куда сохранить локально" -> "где взять на GitHub".
# Внимание: compute_td_loss.py в репо лежит в test_td_loss/, а локально нужен в tests/
# (потому что ноутбук импортит `from tests.compute_td_loss`).
# framebuffer.py из старого Colab-скрипта не существует в репо и в коде не используется.
HELPER_FILES = {
    "dqn/atari_wrappers.py": f"{BASE_URL}/dqn/atari_wrappers.py",
    "dqn/utils.py": f"{BASE_URL}/dqn/utils.py",
    "dqn/replay_buffer.py": f"{BASE_URL}/dqn/replay_buffer.py",
    "dqn/analysis.py": f"{BASE_URL}/dqn/analysis.py",
    "tests/compute_td_loss.py": f"{BASE_URL}/test_td_loss/compute_td_loss.py",
}


def download_helpers() -> None:
    failures: list[str] = []
    for rel_path, url in HELPER_FILES.items():
        local = ROOT / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local)
            print(f"  OK   {rel_path}")
        except Exception as e:
            print(f"  FAIL {rel_path}: {e}")
            failures.append(rel_path)

    # Делаем dqn/ и tests/ нормальными Python-пакетами
    (ROOT / "dqn" / "__init__.py").touch()
    (ROOT / "tests" / "__init__.py").touch()

    if failures:
        raise RuntimeError(f"Не скачались: {failures}")


# -----------------------------------------------------------------------------
# Запуск
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Рабочая папка: {ROOT}")
    step("1. pip install -r requirements.txt", install_requirements)
    step("2. AutoROM --accept-license (Atari ROMs)", install_atari_roms)
    step("3. Скачиваем dqn/* и tests/* из Practical_RL", download_helpers)

    print("\n=== Сводка ===")
    for name, ok, msg in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name}: {msg}")

    n_failed = sum(1 for _, ok, _ in results if not ok)
    if n_failed:
        print(f"\n{n_failed} шаг(ов) упало. Посмотри логи выше.")
        sys.exit(1)
    else:
        print("\nВсё установлено. Перезапусти Jupyter kernel в VS Code и работай.")


if __name__ == "__main__":
    main()
