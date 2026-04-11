"""
ocr/document_classifier.py
Rule-based document type classification (no ML model required).
Uses: contour height variance, edge density, connectivity ratio.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger

logger = get_logger("ocr.document_classifier")


class DocumentClassifier:
    """
    Classifies a document image as: 'printed', 'handwritten', or 'mixed'.

    Features used:
    ─ Edge density         : ratio of edge pixels to total pixels
    ─ Height variance      : normalised variance of contour heights
    ─ Connectivity ratio   : avg connected-component size / total area
    ─ Stroke width variation: CoV of stroke widths (printed = low)
    """

    DOC_TYPES = ("printed", "handwritten", "mixed")

    def classify(self, cv_img: np.ndarray) -> tuple:
        """
        Returns (doc_type: str, confidence: float).
        doc_type in {'printed', 'handwritten', 'mixed'}
        confidence in [0, 1]
        """
        gray = self._to_gray(cv_img)

        edge_density      = self._edge_density(gray)
        height_variance   = self._contour_height_variance(gray)
        connectivity      = self._connectivity_ratio(gray)
        stroke_cov        = self._stroke_width_cov(gray)

        logger.info(
            f"Classifier features: edge_density={edge_density:.4f}, "
            f"height_var={height_variance:.4f}, "
            f"connectivity={connectivity:.4f}, "
            f"stroke_cov={stroke_cov:.4f}"
        )

        doc_type, confidence = self._decision(
            edge_density, height_variance, connectivity, stroke_cov
        )
        logger.info(f"Document classified as '{doc_type}' (conf={confidence:.2f})")
        return doc_type, confidence

    # ── FEATURES ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(cv_img: np.ndarray) -> np.ndarray:
        if len(cv_img.shape) == 2:
            return cv_img
        return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _edge_density(gray: np.ndarray) -> float:
        """Ratio of Canny-edge pixels to total pixels."""
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges   = cv2.Canny(blurred, 50, 150)
        return float(np.count_nonzero(edges)) / gray.size

    @staticmethod
    def _contour_height_variance(gray: np.ndarray) -> float:
        """
        Normalised coefficient of variation of bounding-box heights.
        Printed text → low variance (consistent line height).
        Handwritten  → high variance.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        heights = []
        img_h = gray.shape[0]
        for cnt in contours:
            _, _, w, h = cv2.boundingRect(cnt)
            # Filter noise: height between 0.5% and 20% of image height
            if 0.005 * img_h < h < 0.20 * img_h and w > 3:
                heights.append(h)

        if len(heights) < 5:
            return 0.5  # uncertain

        arr  = np.array(heights, dtype=np.float32)
        mean = arr.mean()
        if mean == 0:
            return 0.5
        cov = arr.std() / mean  # coefficient of variation
        return float(min(cov, 1.0))

    @staticmethod
    def _connectivity_ratio(gray: np.ndarray) -> float:
        """
        Average connected-component area / total image area.
        Printed: small uniform components; handwritten: large irregular.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, stats, *_ = cv2.connectedComponentsWithStats(binary, connectivity=8)[:3]

        if num_labels <= 1:
            return 0.0

        # stats[:,4] = CC_STAT_AREA; skip label 0 (background)
        areas      = stats[1:, cv2.CC_STAT_AREA]
        total_area = gray.size
        mean_area  = float(np.mean(areas)) if len(areas) > 0 else 0.0
        return mean_area / total_area

    @staticmethod
    def _stroke_width_cov(gray: np.ndarray) -> float:
        """
        Estimate stroke width variation using distance transform.
        Printed text → consistent stroke widths → low CoV.
        Handwritten  → variable strokes → high CoV.
        """
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Only consider foreground pixels with meaningful distance
        widths = dist[dist > 0.5]
        if len(widths) < 10:
            return 0.5
        mean = widths.mean()
        if mean == 0:
            return 0.5
        cov = widths.std() / mean
        return float(min(cov, 2.0) / 2.0)  # normalise to [0, 1]

    # ── DECISION LOGIC ────────────────────────────────────────────────────────

    def _decision(
        self,
        edge_density: float,
        height_variance: float,
        connectivity: float,
        stroke_cov: float,
    ) -> tuple:
        """
        Weighted voting across four features.
        Returns (doc_type, confidence).
        """
        # ── SCORES per class ─────────────────────────────────────────────────
        # Printed signals: low edge density variance, low height var, low stroke cov
        # Handwritten signals: high height var, high stroke cov, low edge density
        # Mixed: intermediate values across features

        printed_score = 0.0
        handwritten_score = 0.0

        # Feature 1: edge_density
        if edge_density < 0.05:
            handwritten_score += 1.0   # sparse edges → handwriting
        elif edge_density < 0.10:
            printed_score += 0.5
            handwritten_score += 0.5
        else:
            printed_score += 1.0       # dense regular edges → print

        # Feature 2: height_variance
        if height_variance < 0.20:
            printed_score += 1.2       # very uniform heights → print
        elif height_variance < 0.40:
            printed_score += 0.6
            handwritten_score += 0.6
        else:
            handwritten_score += 1.2   # irregular heights → handwritten

        # Feature 3: connectivity
        if connectivity < 0.005:
            printed_score += 0.8       # small tight components → print
        elif connectivity < 0.015:
            printed_score += 0.4
            handwritten_score += 0.4
        else:
            handwritten_score += 0.8   # large blobs → handwritten

        # Feature 4: stroke_cov
        if stroke_cov < 0.25:
            printed_score += 1.0       # uniform strokes → print
        elif stroke_cov < 0.50:
            printed_score += 0.5
            handwritten_score += 0.5
        else:
            handwritten_score += 1.0   # variable strokes → handwritten

        total = printed_score + handwritten_score
        if total == 0:
            return "mixed", 0.5

        p_norm = printed_score     / total
        h_norm = handwritten_score / total

        # Classify
        if p_norm > 0.65:
            doc_type   = "printed"
            confidence = p_norm
        elif h_norm > 0.65:
            doc_type   = "handwritten"
            confidence = h_norm
        else:
            doc_type   = "mixed"
            confidence = 1.0 - abs(p_norm - h_norm)

        confidence = max(confidence, config.MIN_CLASSIFICATION_CONFIDENCE)
        return doc_type, float(min(confidence, 1.0))
