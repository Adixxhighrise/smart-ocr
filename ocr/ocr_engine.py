"""
ocr/ocr_engine.py
Single-engine OCR wrappers: Tesseract, EasyOCR, TrOCR.

FIX v2 — Proper noun / name accuracy:
─ _post_process_tesseract(): now protects Title-Case tokens (proper nouns
  like names and city names) from being mangled by the garbage-line filter
─ _post_process_tesseract(): the space-insertion regex for 2-char pairs
  is removed — it was incorrectly splitting "it", "is", "in", etc.
─ TesseractEngine: whitelist keeps full punctuation; oem order unchanged
─ EasyOCREngine: unchanged
─ TrOCREngine: unchanged
"""

import os
import sys
import warnings
import contextlib
import io as _io
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, ensure_min_dimension, ensure_min_dimension_cv2, cv2_to_pil

logger = get_logger("ocr.ocr_engine")


@contextlib.contextmanager
def _suppress_stdout_stderr():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


# ─────────────────────────────────────────────────────────────────────────────
# TESSERACT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TesseractEngine:
    """Wrapper around pytesseract with multi-PSM pass and accuracy improvements."""

    _WHITELIST = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        "0123456789 .,!?;:-()[]@#$%&/\\+=<>€£¥₹\n"
    )

    def __init__(self):
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        self._tess = pytesseract

    def extract(self, cv_img: np.ndarray) -> str:
        cv_img = ensure_min_dimension_cv2(cv_img, config.MIN_OCR_DIMENSION)
        gray   = self._to_gray(cv_img)
        gray   = self._enhance_for_tesseract(gray)

        best_text  = ""
        best_score = -1.0

        for psm in config.TESSERACT_PSM_MODES:
            for oem in [3, 1]:
                try:
                    tess_cfg = (
                        f"--oem {oem} "
                        f"--psm {psm} "
                        f"--dpi 300 "
                        f"-c preserve_interword_spaces=1 "
                        f"-c tessedit_char_whitelist={self._WHITELIST.replace(' ', '')}"
                    )
                    text = self._tess.image_to_string(gray, config=tess_cfg)
                    text = self._post_process_tesseract(text.strip())

                    from utils import text_confidence_score
                    score = text_confidence_score(text) * (1 + 0.005 * len(text.split()))
                    logger.info(
                        f"Tesseract PSM={psm} OEM={oem}: "
                        f"{len(text.split())} words, score={score:.3f}"
                    )

                    if score > best_score:
                        best_score = score
                        best_text  = text
                except Exception as exc:
                    logger.warning(f"Tesseract PSM={psm} OEM={oem} failed: {exc}")

        return best_text

    @staticmethod
    def _enhance_for_tesseract(gray: np.ndarray) -> np.ndarray:
        """
        Targeted enhancements for Tesseract accuracy:
        - Bilateral filter: keeps text edges sharp while reducing noise
        - CLAHE: local contrast normalisation
        """
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    @staticmethod
    def _post_process_tesseract(text: str) -> str:
        """
        Fix common Tesseract-specific errors after raw output.
        Conservative filtering — handwriting lines are often short and
        may have low alpha ratio without being garbage.
        """
        import re

        # Fix rn→m misreads
        text = re.sub(r"(?<=[a-z])rn(?=[a-z])", "m", text)

        lines = text.splitlines()
        clean_lines = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                clean_lines.append(line)
                continue

            # Always keep lines starting with a digit+) or digit+. (list items)
            if re.match(r'^\d\s*[).]', stripped):
                clean_lines.append(line)
                continue

            # Always keep lines starting with def / Q / arrow
            if re.match(r'^(def|Q\s*\d|→|➔)', stripped, re.IGNORECASE):
                clean_lines.append(line)
                continue

            # Always keep Title-Case lines (proper nouns, headings)
            first_word = stripped.split()[0].rstrip('.,!?;:-') if stripped.split() else ''
            if (first_word
                    and first_word[0].isupper()
                    and len(first_word) >= 2
                    and (first_word[1:].islower() or first_word[1:2].islower())):
                clean_lines.append(line)
                continue

            # Keep very short lines — likely punctuation/symbols
            if len(stripped) <= 8:
                clean_lines.append(line)
                continue

            # For longer lines: only drop if < 20% alphabetic
            alpha = sum(c.isalpha() for c in stripped)
            ratio = alpha / len(stripped)
            if ratio >= 0.20:
                clean_lines.append(line)
            # else: silently drop garbage line

        return "\n".join(clean_lines)

    @staticmethod
    def _to_gray(cv_img: np.ndarray) -> np.ndarray:
        if len(cv_img.shape) == 2:
            return cv_img
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# EASYOCR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EasyOCREngine:
    """Wrapper around EasyOCR with singleton reader and confidence filtering."""

    _reader = None

    def __init__(self):
        if EasyOCREngine._reader is None:
            logger.info("Initialising EasyOCR reader (first use)…")
            with _suppress_stdout_stderr():
                import easyocr
                EasyOCREngine._reader = easyocr.Reader(
                    config.EASYOCR_LANGUAGES,
                    gpu=False,
                    verbose=False,
                )
            logger.info("EasyOCR reader ready.")

    def readtext_raw(self, pil_image: Image.Image) -> list:
        pil_image = ensure_min_dimension(pil_image, config.MIN_OCR_DIMENSION)
        img_np    = np.array(pil_image.convert("RGB"))
        try:
            with _suppress_stdout_stderr():
                return EasyOCREngine._reader.readtext(
                    img_np,
                    paragraph=True,       # group words into lines/paragraphs
                    detail=1,
                    width_ths=0.9,        # wider grouping threshold
                    contrast_ths=0.05,
                    adjust_contrast=0.7,
                    text_threshold=0.6,
                    low_text=0.3,
                    link_threshold=0.3,
                    canvas_size=2560,
                    mag_ratio=2.0,        # larger magnification for handwriting
                )
        except Exception as exc:
            logger.error(f"EasyOCR readtext_raw failed: {exc}")
            return []

    def extract(self, pil_image: Image.Image) -> str:
        results = self.readtext_raw(pil_image)
        lines = []
        for item in results:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text = item[1] if len(item) >= 2 else ""
                conf = item[2] if len(item) >= 3 else 1.0
                if isinstance(text, str) and text.strip() and conf > 0.05:
                    lines.append(text.strip())
            elif isinstance(item, str) and item.strip():
                lines.append(item.strip())
        return "\n".join(lines)  # use newlines not spaces — paragraph mode gives full lines


# ─────────────────────────────────────────────────────────────────────────────
# TrOCR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TrOCREngine:
    """
    Microsoft TrOCR for handwritten text recognition.
    Lazily loaded; falls back gracefully if transformers/torch unavailable.
    """

    _processor = None
    _model     = None
    _loaded    = False
    _failed    = False

    def __init__(self):
        if not TrOCREngine._loaded and not TrOCREngine._failed:
            self._load_model()

    @classmethod
    def _load_model(cls):
        try:
            logger.info("Loading TrOCR model…")
            with _suppress_stdout_stderr():
                from transformers import TrOCRProcessor, VisionEncoderDecoderModel
                cls._processor = TrOCRProcessor.from_pretrained(config.TROCR_MODEL_NAME)
                cls._model     = VisionEncoderDecoderModel.from_pretrained(config.TROCR_MODEL_NAME)
                cls._model.eval()
            cls._loaded = True
            logger.info("TrOCR model loaded.")
        except Exception as exc:
            logger.warning(f"TrOCR model load failed (will skip): {exc}")
            cls._failed = True

    def extract(self, pil_image: Image.Image) -> str:
        if TrOCREngine._failed or not TrOCREngine._loaded:
            return ""
        return self._run_on_crop(pil_image)

    def extract_from_regions(
        self,
        pil_image: Image.Image,
        easyocr_results: list,
        fallback_to_easy: bool = True,
    ) -> str:
        if TrOCREngine._failed or not TrOCREngine._loaded:
            return ""
        if not easyocr_results:
            return self._run_on_crop(pil_image)

        words  = []
        img_w, img_h = pil_image.size

        for item in easyocr_results:
            if not (isinstance(item, (list, tuple)) and len(item) >= 2):
                continue

            bbox      = item[0]
            easy_text = item[1] if len(item) > 1 else ""
            easy_conf = item[2] if len(item) > 2 else 1.0

            if easy_conf < 0.05:
                continue

            try:
                xs  = [int(p[0]) for p in bbox]
                ys  = [int(p[1]) for p in bbox]
                region_h = max(ys) - min(ys)
                pad = max(6, int(region_h * 0.15))
                x1 = max(0,     min(xs) - pad)
                y1 = max(0,     min(ys) - pad)
                x2 = min(img_w, max(xs) + pad)
                y2 = min(img_h, max(ys) + pad)

                if (x2 - x1) < 8 or (y2 - y1) < 8:
                    if easy_text.strip():
                        words.append(easy_text.strip())
                    continue

                crop       = pil_image.crop((x1, y1, x2, y2))
                trocr_text = self._run_on_crop(crop)

                if trocr_text:
                    words.append(trocr_text)
                elif fallback_to_easy and easy_text.strip():
                    words.append(easy_text.strip())

            except Exception as exc:
                logger.warning(f"TrOCR crop failed: {exc}")
                if fallback_to_easy and easy_text.strip():
                    words.append(easy_text.strip())

        return " ".join(words)

    def _run_on_crop(self, pil_image: Image.Image) -> str:
        try:
            import torch
            pil_image = ensure_min_dimension(pil_image, config.MIN_OCR_DIMENSION)
            rgb = pil_image.convert("RGB")

            pixel_values = TrOCREngine._processor(
                images=rgb, return_tensors="pt"
            ).pixel_values

            with torch.no_grad():
                generated_ids = TrOCREngine._model.generate(
                    pixel_values,
                    max_new_tokens=config.TROCR_MAX_NEW_TOKENS,
                    num_beams=4,
                    early_stopping=True,
                )

            text = TrOCREngine._processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            return text.strip()
        except Exception as exc:
            logger.warning(f"TrOCR _run_on_crop failed: {exc}")
            return ""