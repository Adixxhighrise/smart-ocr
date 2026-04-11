"""
ocr/hybrid_ocr_engine.py
Hybrid OCR ensemble: selects engines based on document type,
combines results using confidence + length heuristics.

FIX v2 — Handwriting accuracy improvements:
─ _run_handwritten(): now also runs Tesseract (PSM 6 + PSM 4) as a
  third candidate alongside EasyOCR and TrOCR. For structured
  handwritten documents like academic notes, Tesseract often outperforms
  EasyOCR because the text is laid out in neat lines. Best of 3 is kept.
─ _preprocess_for_notebook(): new preprocessing specifically for
  lined notebook paper — removes horizontal ruled lines, boosts contrast
  for blue ink on white paper. Called before EasyOCR on handwritten docs.
─ _run_handwritten(): falls back gracefully if any engine fails.
─ _pick_best_of_three(): picks the result with the highest
  confidence×word-count score rather than just using choose_better_text().
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, cv2_to_pil, pil_to_cv2, text_confidence_score, choose_better_text

# Import post processor
try:
    from postprocessing.ocr_post_processor import OCRPostProcessor
except ImportError:
    from ocr_post_processor import OCRPostProcessor

logger = get_logger("ocr.hybrid_ocr_engine")


class HybridOCREngine:
    """
    Routes OCR to the appropriate engine(s) based on document type:
    ─ printed     → Tesseract (multi-pass PSM)
    ─ handwritten → EasyOCR + TrOCR + Tesseract, take best of three
    ─ mixed       → Tesseract + EasyOCR ensemble, weighted merge
    """

    def __init__(self):
        self._tess_engine    = None
        self._easy_engine    = None
        self._trocr_engine   = None
        self._post_processor = OCRPostProcessor()

    # ── LAZY LOADERS ─────────────────────────────────────────────────────────

    def _apply_postprocessing(self, raw_text: str) -> dict:
        try:
            result = self._post_processor.process(raw_text)
            return result
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return {
                "text": raw_text,
                "confidence_pct": 0.0,
                "doc_type": "unknown",
            }

    def _get_tesseract(self):
        if self._tess_engine is None:
            from ocr.ocr_engine import TesseractEngine
            self._tess_engine = TesseractEngine()
        return self._tess_engine

    def _get_easyocr(self):
        if self._easy_engine is None:
            from ocr.ocr_engine import EasyOCREngine
            self._easy_engine = EasyOCREngine()
        return self._easy_engine

    def _get_trocr(self):
        if self._trocr_engine is None:
            from ocr.ocr_engine import TrOCREngine
            self._trocr_engine = TrOCREngine()
        return self._trocr_engine

    # ── MAIN INTERFACE ────────────────────────────────────────────────────────

    def extract(self, image, doc_type: str = "printed") -> dict:
        """
        Extract text from image using appropriate engine(s).

        Parameters
        ----------
        image    : PIL.Image or numpy ndarray
        doc_type : 'printed' | 'handwritten' | 'mixed'

        Returns
        -------
        dict with 'text', 'confidence_pct', 'doc_type' keys
        """
        pil_img, cv_img = self._normalise_input(image)
        doc_type = (doc_type or "printed").lower().strip()

        logger.info(f"Running OCR pipeline for doc_type='{doc_type}'")

        if doc_type == "printed":
            return self._run_printed(cv_img)
        elif doc_type == "handwritten":
            return self._run_handwritten(pil_img, cv_img)
        else:
            return self._run_mixed(pil_img, cv_img)

    # ── ENGINE PIPELINES ─────────────────────────────────────────────────────

    def _run_printed(self, cv_img: np.ndarray) -> dict:
        logger.info("Using Tesseract for printed document.")
        tess = self._get_tesseract()
        text = tess.extract(cv_img)
        logger.info(f"Tesseract extracted {len(text.split())} words.")
        return self._apply_postprocessing(text)

    def _run_handwritten(self, pil_img: Image.Image, cv_img: np.ndarray) -> dict:
        """
        FIX v2: Run EasyOCR, TrOCR, AND Tesseract. Pick best of three.

        Tesseract with PSM 6 (single uniform block) often handles neatly
        written academic notes better than EasyOCR, which tends to produce
        garbage (single letters, fragmented words) on notebook paper.

        Preprocessing: notebook-paper-specific cleanup is applied before
        EasyOCR to improve its accuracy on lined paper.
        """
        easy_results = []
        easy_text    = ""
        trocr_text   = ""
        tess_text    = ""

        # ── Step 1: Notebook-paper preprocessing ─────────────────────────────
        cleaned_pil = self._preprocess_for_notebook(pil_img)
        cleaned_cv  = pil_to_cv2(cleaned_pil)

        # ── Step 2: Tesseract (NEW for handwritten) ───────────────────────────
        # Tesseract PSM 6 works well for single-column academic notes
        try:
            tess = self._get_tesseract()
            tess_text = tess.extract(cleaned_cv)
            logger.info(f"Tesseract (handwritten): {len(tess_text.split())} words.")
        except Exception as exc:
            logger.warning(f"Tesseract (handwritten) failed: {exc}")

        # ── Step 3: EasyOCR (get raw results for bbox cropping) ──────────────
        try:
            easy = self._get_easyocr()
            easy_results = easy.readtext_raw(cleaned_pil)
            easy_text    = easy.extract(cleaned_pil)
            logger.info(f"EasyOCR extracted {len(easy_text.split())} words.")
        except Exception as exc:
            logger.warning(f"EasyOCR failed: {exc}")

        # ── Step 4: TrOCR on EasyOCR bounding-box crops ──────────────────────
        try:
            trocr = self._get_trocr()
            if easy_results:
                trocr_text = trocr.extract_from_regions(
                    cleaned_pil, easy_results, fallback_to_easy=True
                )
            else:
                trocr_text = trocr.extract(cleaned_pil)
            logger.info(f"TrOCR extracted {len(trocr_text.split())} words.")
        except Exception as exc:
            logger.warning(f"TrOCR failed: {exc}")

        # ── Step 5: Pick best of three ────────────────────────────────────────
        best = self._pick_best_of_three(tess_text, easy_text, trocr_text)
        logger.info(f"Handwritten OCR best result: {len(best.split())} words.")
        return self._apply_postprocessing(best)

    def _run_mixed(self, pil_img: Image.Image, cv_img: np.ndarray) -> dict:
        logger.info("Using Tesseract + EasyOCR ensemble for mixed document.")
        tess_text = ""
        easy_text = ""
        try:
            tess_text = self._get_tesseract().extract(cv_img)
            logger.info(f"Tesseract (mixed): {len(tess_text.split())} words.")
        except Exception as exc:
            logger.warning(f"Tesseract (mixed) failed: {exc}")
        try:
            easy_text = self._get_easyocr().extract(pil_img)
            logger.info(f"EasyOCR (mixed): {len(easy_text.split())} words.")
        except Exception as exc:
            logger.warning(f"EasyOCR (mixed) failed: {exc}")
        merged = self._weighted_merge(tess_text, easy_text)
        logger.info(f"Mixed OCR merged result: {len(merged.split())} words.")
        return self._apply_postprocessing(merged)

    # ── NOTEBOOK PREPROCESSING ────────────────────────────────────────────────

    @staticmethod
    def _preprocess_for_notebook(pil_img: Image.Image) -> Image.Image:
        """
        Preprocessing tuned for lined notebook paper with handwritten text.

        Steps:
        1. Convert to grayscale
        2. CLAHE for contrast boost (helps blue ink on white paper)
        3. Remove horizontal ruled lines using morphological opening
        4. Adaptive threshold to get clean binary
        5. Return as PIL RGB image
        """
        try:
            import cv2
            cv_img = pil_to_cv2(pil_img)
            gray   = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            # CLAHE — boost local contrast
            clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Remove horizontal ruled lines:
            # A horizontal line is a very wide, thin connected component.
            # Use a wide horizontal structuring element to detect & remove them.
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            detected_lines    = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN,
                                                  horizontal_kernel, iterations=2)
            # Subtract lines from image (add them to whitened background)
            cleaned = cv2.add(enhanced, detected_lines)

            # Adaptive threshold for clean black-on-white
            binary = cv2.adaptiveThreshold(
                cleaned, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=15, C=10,
            )

            # Convert back to 3-channel PIL
            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return cv2_to_pil(binary_bgr)

        except Exception as exc:
            logger.warning(f"Notebook preprocessing failed, using original: {exc}")
            return pil_img

    # ── BEST-OF-THREE SELECTION ───────────────────────────────────────────────

    @staticmethod
    def _pick_best_of_three(text_a: str, text_b: str, text_c: str) -> str:
        """
        Score each candidate by: confidence_score × (1 + 0.005 × word_count)
        Returns the candidate with the highest score.
        A blank candidate scores 0.
        """
        candidates = [t for t in (text_a, text_b, text_c) if t and t.strip()]
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        def _score(t: str) -> float:
            if not t or not t.strip():
                return 0.0
            return text_confidence_score(t) * (1 + 0.005 * len(t.split()))

        best = max(candidates, key=_score)
        logger.info(
            f"_pick_best_of_three scores: "
            f"A={_score(text_a):.3f} B={_score(text_b):.3f} C={_score(text_c):.3f}"
        )
        return best

    # ── ENSEMBLE MERGE ────────────────────────────────────────────────────────

    @staticmethod
    def _weighted_merge(text_a: str, text_b: str) -> str:
        if not text_a.strip():
            return text_b
        if not text_b.strip():
            return text_a

        score_a = text_confidence_score(text_a) * (1 + 0.005 * len(text_a.split()))
        score_b = text_confidence_score(text_b) * (1 + 0.005 * len(text_b.split()))

        w_a = config.HYBRID_TESSERACT_WEIGHT
        w_b = config.HYBRID_EASYOCR_WEIGHT

        weighted_a = score_a * w_a
        weighted_b = score_b * w_b

        if abs(weighted_a - weighted_b) < 0.05:
            return text_a + "\n" + text_b
        return text_a if weighted_a >= weighted_b else text_b

    # ── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_input(image):
        """Return (pil_image, cv2_image) regardless of input type."""
        if isinstance(image, Image.Image):
            pil_img = image
            cv_img  = pil_to_cv2(image)
        else:
            cv_img  = image
            pil_img = cv2_to_pil(image)
        return pil_img, cv_img