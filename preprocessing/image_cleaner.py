"""
preprocessing/image_cleaner.py
Multi-pass intelligent image preprocessing pipeline for OCR.

FIX v2 — Notebook paper support:
─ Added Variant 7: 'notebook' — specifically designed for lined notebook
  paper with handwritten text. Removes horizontal ruled lines using
  morphological processing, boosts blue-ink contrast, applies aggressive
  CLAHE before thresholding.
─ _quality_score(): now also rewards images with good text density
  (0.05–0.30 dark pixel ratio) — lined paper images were being penalised
  because the ruled lines inflated dark pixel count.
─ generate_variants(): notebook variant added as variant 7.
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
from utils import (
    get_logger, pil_to_cv2, cv2_to_pil,
    ensure_min_dimension, ensure_min_dimension_cv2,
)

logger = get_logger("preprocessing.image_cleaner")


class ImageCleaner:
    """
    Intelligent image preprocessing with multi-pass variant generation.
    Selects the best preprocessed variant based on a quality heuristic.
    """

    def __init__(self):
        pass

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def preprocess(self, image) -> np.ndarray:
        """
        Main entry point. Accepts PIL Image or numpy ndarray.
        Returns the best preprocessed BGR ndarray.
        """
        if isinstance(image, Image.Image):
            cv_img = pil_to_cv2(image)
        else:
            cv_img = image.copy()

        cv_img = ensure_min_dimension_cv2(cv_img, config.MIN_OCR_DIMENSION)
        # Remove red margin line before any further processing
        cv_img = self._remove_red_margin(cv_img)
        best = self._select_best_variant(cv_img)
        logger.info("Preprocessing complete.")
        return best

    def generate_variants(self, cv_img: np.ndarray) -> list:
        """
        Generate multiple preprocessing variants of the same image.
        Returns list of (name, ndarray) tuples.
        """
        gray = self._to_gray(cv_img)
        variants = []

        # HSV-based notebook variant is best for blue-ink on ruled paper
        variants.append(("notebook_hsv",   self._notebook_hsv(cv_img)))
        variants.append(("notebook_morph", self._notebook_paper(gray)))
        variants.append(("clahe_otsu",     self._clahe_otsu(gray)))
        variants.append(("adaptive",       self._adaptive_threshold(gray)))
        variants.append(("otsu",           self._otsu(gray)))
        variants.append(("sauvola",        self._sauvola_approx(gray)))
        variants.append(("denoised",       self._denoised_threshold(gray)))
        variants.append(("morph",          self._morph_clean(gray)))

        return variants

    # ── SELECTION ─────────────────────────────────────────────────────────────

    def _select_best_variant(self, cv_img: np.ndarray) -> np.ndarray:
        """Select the variant with the most foreground/background separation."""
        variants = self.generate_variants(cv_img)
        best_name:  str             = ""
        best_img:   np.ndarray      = None
        best_score: float           = -1.0

        for name, variant in variants:
            score = self._quality_score(variant)
            logger.info(f"  Variant '{name}' quality score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_name  = name
                best_img   = variant

        logger.info(f"Selected variant: '{best_name}' (score={best_score:.4f})")
        if len(best_img.shape) == 2:
            best_img = cv2.cvtColor(best_img, cv2.COLOR_GRAY2BGR)
        return best_img

    def _quality_score(self, gray_or_binary: np.ndarray) -> float:
        """
        Heuristic: high sharpness + good text density.
        Ideal dark pixel ratio for handwriting on white: 3%–25%.
        """
        if len(gray_or_binary.shape) == 3:
            img = cv2.cvtColor(gray_or_binary, cv2.COLOR_BGR2GRAY)
        else:
            img = gray_or_binary

        lap_var = cv2.Laplacian(img, cv2.CV_64F).var()

        text_ratio = np.sum(img < 127) / img.size
        if 0.03 <= text_ratio <= 0.25:
            density_bonus = 1.0
        else:
            center = 0.12
            density_bonus = max(1.0 - abs(text_ratio - center) * 3, 0.05)

        return float(lap_var * density_bonus)

    # ── VARIANTS ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(cv_img: np.ndarray) -> np.ndarray:
        if len(cv_img.shape) == 2:
            return cv_img
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _otsu(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _adaptive_threshold(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11, C=8,
        )
        return thresh

    @staticmethod
    def _sauvola_approx(gray: np.ndarray) -> np.ndarray:
        gray_f = gray.astype(np.float32) / 255.0
        w = 15
        mean    = cv2.blur(gray_f, (w * 2 + 1, w * 2 + 1))
        mean_sq = cv2.blur(gray_f ** 2, (w * 2 + 1, w * 2 + 1))
        std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
        k = 0.2
        R = 0.5
        threshold = mean * (1 + k * (std / R - 1))
        binary = (gray_f >= threshold).astype(np.uint8) * 255
        return binary

    @staticmethod
    def _clahe_otsu(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _denoised_threshold(gray: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _morph_clean(gray: np.ndarray) -> np.ndarray:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel2)
        return cleaned

    # ── RED MARGIN REMOVAL ────────────────────────────────────────────────────

    @staticmethod
    def _remove_red_margin(cv_img: np.ndarray) -> np.ndarray:
        """
        White-out the red vertical margin line using HSV color masking.
        Finds the rightmost column of red content and blanks left of it.
        """
        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, np.array([0,  70, 70]),  np.array([10,  255, 255]))
        red2 = cv2.inRange(hsv, np.array([165, 70, 70]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red1, red2)

        col_sums = np.sum(red_mask, axis=0)
        red_cols = np.where(col_sums > cv_img.shape[0] * 0.04)[0]

        if len(red_cols) == 0:
            return cv_img

        margin_col = min(int(red_cols.max()) + 12, cv_img.shape[1] - 1)
        result = cv_img.copy()
        result[:, :margin_col] = 255
        logger.info(f"Red margin removed up to column {margin_col}")
        return result

    # ── NOTEBOOK HSV VARIANT ──────────────────────────────────────────────────

    @staticmethod
    def _notebook_hsv(cv_img: np.ndarray) -> np.ndarray:
        """
        HSV/LAB-based approach for blue-ink handwriting on blue-ruled paper.

        Key insight: ruled lines are LIGHTER blue, ink strokes are DARKER blue.
        In LAB color space, the L channel separates them by intensity.

        Pipeline:
        1. LAB decomposition — L channel gives intensity, b channel gives blue
        2. Combine inverted-L (dark=ink) with blue indicator
        3. Otsu threshold to binary
        4. Detect and remove long horizontal lines (ruled lines)
        5. Clean noise, slightly dilate to connect strokes
        """
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        L, a, b_ch = cv2.split(lab)

        # CLAHE on L to normalise uneven lighting
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        L_eq = clahe.apply(L)

        # b_ch < 128 = blue; invert so blue areas are bright
        b_inv = 255 - b_ch

        # Combine: dark on L (ink) + blue on b_inv
        combined = cv2.addWeighted(255 - L_eq, 0.65, b_inv, 0.35, 0)

        # Threshold
        _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Remove horizontal ruled lines
        h, w = binary.shape[:2]
        lw = max(w // 7, 50)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (lw, 1))
        lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
        erase_k = cv2.getStructuringElement(cv2.MORPH_RECT, (lw, 4))
        lines_fat = cv2.dilate(lines, erase_k, iterations=1)
        binary[lines_fat > 0] = 0

        # Invert so ink = dark on white background
        result = cv2.bitwise_not(binary)

        # Remove tiny noise specks
        noise_k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, noise_k)

        return result

    @staticmethod
    def _notebook_paper(gray: np.ndarray) -> np.ndarray:
        """
        Morphology-based fallback for notebook paper.
        Pipeline:
        1. Bilateral filter (edge-preserving denoise)
        2. Strong CLAHE
        3. Remove horizontal ruled lines via morphological opening
        4. Adaptive threshold
        5. Light dilation to connect strokes
        """
        bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)

        h_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (enhanced.shape[1] // 8, 1)
        )
        detected_lines = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN,
                                           h_kernel, iterations=2)
        without_lines = cv2.add(enhanced, detected_lines)

        binary = cv2.adaptiveThreshold(
            without_lines, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21, C=12,
        )

        ink_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.dilate(binary, ink_kernel, iterations=1)
        return binary

    # ── DESKEW ────────────────────────────────────────────────────────────────

    @staticmethod
    def deskew(gray: np.ndarray) -> np.ndarray:
        """Correct image skew using Hough line transform."""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        if lines is None:
            return gray

        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return gray

        median_angle = np.median(angles)
        if abs(median_angle) < 0.5:
            return gray

        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        logger.info(f"Deskewed by {median_angle:.2f}°")
        return rotated

    @staticmethod
    def remove_border(gray: np.ndarray, margin: int = 10) -> np.ndarray:
        """Crop a fixed margin from each edge to remove scan borders."""
        h, w = gray.shape
        return gray[margin: h - margin, margin: w - margin]

    @staticmethod
    def stretch_contrast(gray: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(gray, (2, 98))
        if p98 == p2:
            return gray
        stretched = np.clip((gray.astype(np.float32) - p2) / (p98 - p2) * 255, 0, 255)
        return stretched.astype(np.uint8)