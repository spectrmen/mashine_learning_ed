"""
GPU-установка окружения для lab05_dqn под RTX 4080 (CUDA 12.9 driver, Python 3.12).

Что делает:
1. Сносит CPU-версию torch если стоит
2. Ставит PyTorch с CUDA 12.8 wheel (backward-compatible с CUDA 12.9 driver)
3. pip install -r requirements.txt (без torch — он уже стоит)
4. AutoROM --accept-license (Atari ROMs)
5. Если каких-то helper-файлов вдруг нет (dqn/, tests/, utils.py, replay_buffer.py) —
   качает их с GitHub курса. Обычно они уже в репо после git pull.

Запуск:
    python setup_lab5_gpu.py

Если на каком-то шаге упадёт — следующие всё равно выполнятся.
"""

from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
results: list[tuple[str, bool, str]] = []


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
# Шаг 1. Снести любой существующий torch (CPU-версию или старую)
# -----------------------------------------------------------------------------
def uninstall_torch() -> None:
    # -y чтобы не спрашивал, --quiet чтобы меньше шума.
    # Не падаем, если torch не установлен.
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


# -----------------------------------------------------------------------------
# Шаг 2. Установить torch с CUDA 12.8 (cu128)
# -----------------------------------------------------------------------------
def install_torch_cuda() -> None:
    # PyTorch собран под CUDA 12.8, но runtime forward-compatible с 12.9 driver.
    # RTX 4080 — Ada Lovelace (sm_89), поддерживается стабильно.
    run_cmd([
        sys.executable, "-m", "pip", "install",
        "torch",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ])


# -----------------------------------------------------------------------------
# Шаг 3. Остальные пакеты (без torch — он уже стоит).
# -----------------------------------------------------------------------------
def install_other_requirements() -> None:
    """Ставим всё из requirements.txt, пропуская torch (он уже из cu128 wheel)."""
    req = ROOT / "requirements.txt"
    if not req.exists():
        raise FileNotFoundError(f"Не нашёл {req}")

    # Читаем requirements, отфильтровываем строчку с torch
    lines = req.read_text(encoding="utf-8").splitlines()
    other_pkgs: list[str] = []
    for ln in lines:
        ln_stripped = ln.strip()
        if not ln_stripped or ln_stripped.startswith("#"):
            continue
        # Не дублируем torch (его уже поставили из cu128 wheel)
        if ln_stripped.lower().startswith("torch"):
            continue
        other_pkgs.append(ln_stripped)

    if not other_pkgs:
        return
    run_cmd([sys.executable, "-m", "pip", "install", *other_pkgs])


# -----------------------------------------------------------------------------
# Шаг 4. Atari ROMs через AutoROM
# -----------------------------------------------------------------------------
def install_atari_roms() -> None:
    run_cmd([sys.executable, "-m", "AutoROM", "--accept-license"])


# -----------------------------------------------------------------------------
# Шаг 5. Подстраховка — helper-файлы (если репо без них)
# -----------------------------------------------------------------------------
BASE_URL = (
    "https://raw.githubusercontent.com/yandexdataschool/"
    "Practical_RL/master/week04_approx_rl"
)
HELPER_FILES = {
    "dqn/atari_wrappers.py": f"{BASE_URL}/dqn/atari_wrappers.py",
    "dqn/utils.py": f"{BASE_URL}/dqn/utils.py",
    "dqn/replay_buffer.py": f"{BASE_URL}/dqn/replay_buffer.py",
    "dqn/analysis.py": f"{BASE_URL}/dqn/analysis.py",
    "tests/compute_td_loss.py": f"{BASE_URL}/test_td_loss/compute_td_loss.py",
}


def download_helpers_if_missing() -> None:
    missing: list[tuple[str, str]] = []
    for rel_path, url in HELPER_FILES.items():
        local = ROOT / rel_path
        if not local.exists():
            missing.append((rel_path, url))

    if not missing:
        print("Все helper-файлы уже на месте.")
        return

    print(f"Не хватает {len(missing)} файлов, докачиваю...")
    for rel_path, url in missing:
        local = ROOT / rel_path
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(url, local)
            print(f"  OK   {rel_path}")
        except Exception as e:
            print(f"  FAIL {rel_path}: {e}")

    (ROOT / "dqn" / "__init__.py").touch()
    (ROOT / "tests" / "__init__.py").touch()


# -----------------------------------------------------------------------------
# Шаг 6. Smoke-test что torch с CUDA реально работает
# -----------------------------------------------------------------------------
def verify_cuda() -> None:
    code = """
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    x = torch.zeros(3, 3, device="cuda")
    y = x + 1
    print("Tensor on CUDA OK, sum =", y.sum().item())
else:
    raise RuntimeError("torch.cuda.is_available() == False")
"""
    run_cmd([sys.executable, "-c", code])


# -----------------------------------------------------------------------------
# Запуск
# -----------------------------------------------------------------------------
def main() -> None:
    print(f"Рабочая папка: {ROOT}")
    print(f"Python: {sys.version.split()[0]}")

    step("1. Снос старого torch", uninstall_torch)
    step("2. Установка torch с CUDA 12.8 (cu128)", install_torch_cuda)
    step("3. Установка остальных пакетов из requirements.txt", install_other_requirements)
    step("4. AutoROM --accept-license (Atari ROMs)", install_atari_roms)
    step("5. Проверка helper-файлов (dqn/, tests/)", download_helpers_if_missing)
    step("6. Smoke-test torch+CUDA", verify_cuda)

    print("\n=== Сводка ===")
    for name, ok, msg in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name}: {msg}")

    n_failed = sum(1 for _, ok, _ in results if not ok)
    if n_failed:
        print(f"\n{n_failed} шаг(ов) упало. Посмотри логи выше.")
        sys.exit(1)
    else:
        print("\nВсё установлено. Перезапусти Jupyter kernel в VS Code.")
        print("Не забудь положить last_state_dict.pt в lab5/ (он в репо после git pull).")


if __name__ == "__main__":
    main()
