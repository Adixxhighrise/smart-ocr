# ocr/__init__.py
from ocr.document_classifier  import DocumentClassifier
from ocr.ocr_engine            import TesseractEngine, EasyOCREngine, TrOCREngine
from ocr.hybrid_ocr_engine     import HybridOCREngine

# Alias so any code doing `from ocr import OCREngine` still works
OCREngine = HybridOCREngine

__all__ = [
    "DocumentClassifier",
    "TesseractEngine",
    "EasyOCREngine",
    "TrOCREngine",
    "HybridOCREngine",
    "OCREngine",
]