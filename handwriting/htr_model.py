"""
handwriting/htr_model.py
High-level handwriting recognition facade that coordinates
EasyOCR + TrOCR and applies line segmentation for improved accuracy.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, ensure_min_dimension, ensure_min_dimension_cv2, cv2_to_pil

logger = get_logger("handwriting.htr_model")


class HandwritingRecogniser:
    """
    Handwriting-specific text recogniser.
    Performs:
    1. Line segmentation (horizontal projection profile)
    2. Per-line EasyOCR + TrOCR inference
    3. Confidence-weighted combination
    """

    def __init__(self):
        self._easy  = None
        self._trocr = None

    # ── LAZY LOAD ─────────────────────────────────────────────────────────────

    def _get_easy(self):
        if self._easy is None:
            from ocr.ocr_engine import EasyOCREngine
            self._easy = EasyOCREngine()
        return self._easy

    def _get_trocr(self):
        if self._trocr is None:
            from ocr.ocr_engine import TrOCREngine
            self._trocr = TrOCREngine()
        return self._trocr

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def recognise(self, image) -> str:
        """
        Recognise handwriting in the given image.
        Returns extracted text string.
        """
        if isinstance(image, Image.Image):
            pil_img = image
        else:
            pil_img = cv2_to_pil(image)

        pil_img = ensure_min_dimension(pil_img, config.MIN_OCR_DIMENSION)
        lines   = self._segment_lines(pil_img)

        if len(lines) <= 1:
            # No line segmentation possible: use full image
            return self._recognise_region(pil_img)

        logger.info(f"Line segmentation: {len(lines)} lines detected.")
        line_texts = []
        for i, line_img in enumerate(lines):
            text = self._recognise_region(line_img)
            logger.info(f"  Line {i+1}: '{text[:60]}{'…' if len(text)>60 else ''}'")
            if text.strip():
                line_texts.append(text.strip())

        return "\n".join(line_texts)

    # ── LINE SEGMENTATION ─────────────────────────────────────────────────────

    def _segment_lines(self, pil_img: Image.Image) -> list:
        """
        Segment image into individual text lines using horizontal
        projection profile analysis.
        Returns list of PIL Images (one per line).
        """
        gray = np.array(pil_img.convert("L"))
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Horizontal projection (sum of ink pixels per row)
        h_proj = np.sum(binary, axis=1)

        # Smooth to avoid jagged cuts
        kernel = np.ones(5) / 5.0
        h_proj_smooth = np.convolve(h_proj, kernel, mode="same")

        threshold = h_proj_smooth.max() * 0.05  # 5% of max density

        # Find row bands above threshold
        in_line    = False
        line_bounds = []
        start_row   = 0
        margin      = 4  # extra pixels above/below each line

        for row_idx, val in enumerate(h_proj_smooth):
            if not in_line and val > threshold:
                in_line   = True
                start_row = max(0, row_idx - margin)
            elif in_line and val <= threshold:
                in_line = False
                end_row = min(gray.shape[0], row_idx + margin)
                if end_row - start_row > 8:
                    line_bounds.append((start_row, end_row))

        # Handle still-open line at image bottom
        if in_line:
            line_bounds.append((start_row, gray.shape[0]))

        if len(line_bounds) < 2:
            return [pil_img]

        lines = []
        for (top, bot) in line_bounds:
            line_pil = pil_img.crop((0, top, pil_img.width, bot))
            # Re-upscale narrow strips
            line_pil = ensure_min_dimension(line_pil, 300)
            lines.append(line_pil)
        return lines

    # ── REGION RECOGNITION ────────────────────────────────────────────────────

    def _recognise_region(self, pil_img: Image.Image) -> str:
        """Run EasyOCR and TrOCR on a region; return best result."""
        easy_text  = ""
        trocr_text = ""

        try:
            easy_text = self._get_easy().extract(pil_img)
        except Exception as exc:
            logger.warning(f"EasyOCR region failed: {exc}")

        try:
            trocr_text = self._get_trocr().extract(pil_img)
        except Exception as exc:
            logger.warning(f"TrOCR region failed: {exc}")

        from utils import choose_better_text
        return choose_better_text(easy_text, trocr_text)

    # ── STROKE ANALYSIS ───────────────────────────────────────────────────────

    @staticmethod
    def estimate_pen_pressure(cv_img: np.ndarray) -> float:
        """
        Estimate writing pressure from stroke darkness.
        Returns float in [0, 1] — higher = heavier pen pressure.
        """
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink_pixels = np.sum(binary > 0)
        total      = binary.size
        return float(ink_pixels) / total if total > 0 else 0.0

    @staticmethod
    def estimate_slant(cv_img: np.ndarray) -> float:
        """
        Estimate writing slant angle (degrees) using Hough line transform.
        Returns angle; positive = right-slant, negative = left-slant.
        """
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY) if len(cv_img.shape) == 3 else cv_img
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        if lines is None:
            return 0.0
        angles = [np.degrees(line[0][1]) - 90 for line in lines if -45 < np.degrees(line[0][1]) - 90 < 45]
        return float(np.median(angles)) if angles else 0.0
