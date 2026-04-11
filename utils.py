"""
utils.py - Shared utility functions for Smart OCR Application
"""

import os
import sys
import warnings
import logging
import time
import re
import io
import traceback
from functools import wraps

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# SUPPRESS WARNINGS GLOBALLY
# ─────────────────────────────────────────────────────────────────────────────

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress easyocr/torch prints
logging.getLogger("easyocr").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger("utils")

# ─────────────────────────────────────────────────────────────────────────────
# PILLOW COMPATIBILITY (ANTIALIAS REMOVED IN PILLOW >=10)
# ─────────────────────────────────────────────────────────────────────────────

def get_resampling_filter():
    """Return correct Pillow resampling filter regardless of version."""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.ANTIALIAS  # type: ignore[attr-defined]


RESAMPLING = get_resampling_filter()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CONVERSION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    import cv2
    img = pil_image.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR ndarray → PIL Image."""
    import cv2
    if len(cv2_image.shape) == 2:
        # Grayscale
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_image(path: str) -> Image.Image:
    """Load an image from disk, returning RGBA-safe PIL Image."""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
        return background
    return img.convert("RGB")


def pil_to_bytes(pil_image: Image.Image, fmt: str = "PNG") -> bytes:
    """Encode PIL Image to bytes."""
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UPSCALING (MANDATORY FOR SMALL IMAGES)
# ─────────────────────────────────────────────────────────────────────────────

def ensure_min_dimension(image: Image.Image, min_dim: int = 1200) -> Image.Image:
    """
    Upscale image so its largest side is at least min_dim pixels.
    Uses high-quality Lanczos resampling.
    """
    w, h = image.size
    largest = max(w, h)
    if largest >= min_dim:
        return image
    scale = min_dim / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info(f"Upscaling image from {w}x{h} → {new_w}x{new_h}")
    return image.resize((new_w, new_h), RESAMPLING)


def ensure_min_dimension_cv2(img: np.ndarray, min_dim: int = 1200) -> np.ndarray:
    """Upscale OpenCV image so its largest side is at least min_dim pixels."""
    import cv2
    h, w = img.shape[:2]
    largest = max(w, h)
    if largest >= min_dim:
        return img
    scale = min_dim / largest
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs into one; strip leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_words(text: str) -> int:
    return len(text.split())


def text_confidence_score(text: str) -> float:
    """
    Heuristic text quality score in [0, 1].
    Penalises high ratio of non-alphabetic characters.
    """
    if not text:
        return 0.0
    alpha  = sum(c.isalpha() for c in text)
    total  = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return 0.0
    return min(alpha / total, 1.0)


def choose_better_text(text_a: str, text_b: str) -> str:
    """Return whichever text has higher confidence + length heuristic."""
    score_a = text_confidence_score(text_a) * (1 + 0.01 * count_words(text_a))
    score_b = text_confidence_score(text_b) * (1 + 0.01 * count_words(text_b))
    return text_a if score_a >= score_b else text_b


# ─────────────────────────────────────────────────────────────────────────────
# TIMING DECORATOR
# ─────────────────────────────────────────────────────────────────────────────

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"{fn.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# SAFE EXECUTE WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

def safe_run(fn, *args, fallback=None, **kwargs):
    """Run fn(*args, **kwargs), returning fallback on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        logger.warning(f"safe_run caught exception in {fn.__qualname__}: {exc}")
        logger.debug(traceback.format_exc())
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# FILE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def save_text(text: str, filepath: str) -> bool:
    """Save text to file, creating parent dirs if needed."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as exc:
        logger.error(f"Failed to save text: {exc}")
        return False


def make_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)
