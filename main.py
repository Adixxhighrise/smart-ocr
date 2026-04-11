"""
main.py - Smart OCR & Handwriting Recognition Desktop Application
Entry point: configures environment, validates dependencies, launches GUI.
"""

import os
import sys
import warnings
import logging

# ─────────────────────────────────────────────────────────────────────────────
# SUPPRESS ALL WARNINGS BEFORE ANY OTHER IMPORT
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"]          = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

for noisy_logger in (
    "easyocr", "torch", "transformers", "PIL",
    "urllib3", "requests", "tensorflow",
):
    logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from utils import get_logger
logger = get_logger("main")

import config


# ─────────────────────────────────────────────────────────────────────────────
# DEPENDENCY CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_dependencies() -> list:
    """
    Check that required packages are importable.
    Returns a list of missing package names.
    """
    required = {
        "cv2":          "opencv-python",
        "PIL":          "Pillow",
        "numpy":        "numpy",
        "pytesseract":  "pytesseract",
    }
    optional = {
        "easyocr":      "easyocr",
        "torch":        "torch",
        "transformers": "transformers",
        "autocorrect":  "autocorrect",
        "textdistance": "textdistance",
    }

    missing_required = []
    missing_optional = []

    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append(pkg)
            logger.error(f"Required package missing: {pkg}")

    for module, pkg in optional.items():
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(pkg)
            logger.warning(f"Optional package missing: {pkg}")

    if missing_optional:
        logger.warning(
            f"Optional packages not found (some features may be limited): "
            f"{', '.join(missing_optional)}"
        )

    return missing_required


def check_tesseract() -> bool:
    """Check if Tesseract OCR executable is accessible."""
    import subprocess
    try:
        result = subprocess.run(
            [config.TESSERACT_CMD, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            version_line = result.stdout.split("\n")[0]
            logger.info(f"Tesseract found: {version_line}")
            return True
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as exc:
        logger.warning(f"Tesseract not found at '{config.TESSERACT_CMD}': {exc}")
        return False


def show_tesseract_install_instructions():
    """Print platform-specific Tesseract installation instructions."""
    import platform
    plat = platform.system().lower()

    print("\n" + "=" * 60)
    print("  Tesseract OCR not found — Installation Instructions")
    print("=" * 60)

    if plat == "windows":
        print("""
  Windows:
  ─────────
  1. Download installer from:
     https://github.com/UB-Mannheim/tesseract/wiki
  2. Run the installer (default path:
     C:\\Program Files\\Tesseract-OCR\\tesseract.exe)
  3. Add Tesseract to your PATH, or update TESSERACT_PATHS
     in config.py to point to your installation.
""")
    elif plat == "darwin":
        print("""
  macOS:
  ───────
  Using Homebrew:
    brew install tesseract

  Or using MacPorts:
    sudo port install tesseract
""")
    else:
        print("""
  Linux (Debian/Ubuntu):
  ───────────────────────
    sudo apt-get update
    sudo apt-get install tesseract-ocr

  Linux (Fedora/RHEL):
    sudo dnf install tesseract

  Linux (Arch):
    sudo pacman -S tesseract
""")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP CHECKS & WARNINGS DIALOG
# ─────────────────────────────────────────────────────────────────────────────

def run_startup_checks() -> bool:
    """
    Run all startup checks.
    Returns True if the app can proceed (even with warnings).
    """
    logger.info(f"Starting {config.APP_TITLE}")
    logger.info(f"Python {sys.version}")
    logger.info(f"Base directory: {BASE_DIR}")

    missing = check_dependencies()
    if missing:
        msg = (
            f"The following required packages are not installed:\n"
            f"  {', '.join(missing)}\n\n"
            f"Please run:\n"
            f"  pip install -r requirements.txt"
        )
        logger.critical(msg)
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing Dependencies", msg)
            root.destroy()
        except Exception:
            print(f"\nERROR: {msg}\n")
        return False

    tess_ok = check_tesseract()
    if not tess_ok:
        show_tesseract_install_instructions()
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            proceed = messagebox.askyesno(
                "Tesseract Not Found",
                "Tesseract OCR was not found on this system.\n\n"
                "Printed text recognition (Tesseract) will be unavailable.\n"
                "EasyOCR and TrOCR will still work for handwritten text.\n\n"
                "Continue anyway?",
            )
            root.destroy()
            if not proceed:
                return False
        except Exception:
            logger.warning("Continuing without Tesseract (no GUI available for prompt).")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not run_startup_checks():
        sys.exit(1)

    logger.info("Launching GUI…")
    from gui.interface import SmartOCRApp

    app = SmartOCRApp()
    app.run()


if __name__ == "__main__":
    main()
