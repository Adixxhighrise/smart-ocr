"""
config.py - Central configuration for Smart OCR Desktop Application
"""

import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TESSERACT_CANDIDATE_PATHS = {
    "win32": [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ],
    "darwin": [
        "/opt/homebrew/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/local/bin/tesseract",
    ],
    "linux": [
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ],
}

def get_tesseract_cmd():
    import shutil
    on_path = shutil.which("tesseract")
    if on_path:
        return on_path
    platform = sys.platform
    for key, paths in TESSERACT_CANDIDATE_PATHS.items():
        if platform.startswith(key):
            for path in paths:
                if os.path.isfile(path):
                    return path
    return "tesseract"

TESSERACT_CMD = get_tesseract_cmd()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

MIN_OCR_DIMENSION    = 800
UPSCALE_INTERPOLATION = 2     # cv2.INTER_CUBIC

# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED INPUT FORMATS
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_FORMATS = [
    ("All Supported", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.pdf"),
    ("PDF Documents", "*.pdf"),
    ("Image Files",   "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
    ("JPEG",          "*.jpg *.jpeg"),
    ("PNG",           "*.png"),
    ("BMP",           "*.bmp"),
    ("TIFF",          "*.tiff *.tif"),
    ("All Files",     "*.*"),
]

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT CLASSIFICATION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

PRINTED_EDGE_DENSITY_MIN       = 0.04
HANDWRITTEN_EDGE_DENSITY_MAX   = 0.12
PRINTED_HEIGHT_VARIANCE_MAX    = 0.35
HANDWRITTEN_HEIGHT_VARIANCE_MIN = 0.25
PRINTED_CONNECTIVITY_MAX       = 0.18
HANDWRITTEN_CONNECTIVITY_MIN   = 0.08
MIN_CLASSIFICATION_CONFIDENCE  = 0.50

# ─────────────────────────────────────────────────────────────────────────────
# OCR ENGINE SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

TESSERACT_PSM_MODES  = [6, 3, 4, 11, 12]

TESSERACT_OEM        = 3      # LSTM + Legacy
EASYOCR_LANGUAGES    = ["en"]
# TROCR_MODEL_NAME     = "microsoft/trocr-base-handwritten"
TROCR_MODEL_NAME = "microsoft/trocr-large-handwritten"
TROCR_MAX_NEW_TOKENS = 128

# Ensemble weights
HYBRID_TESSERACT_WEIGHT = 0.45
HYBRID_EASYOCR_WEIGHT   = 0.35
HYBRID_TROCR_WEIGHT     = 0.20

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING / CORRECTION SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

SPELL_CORRECTION_CONFIDENCE_THRESHOLD = 0.82

AMBIGUOUS_CHAR_MAP = {
    "0": "O",
    "1": "l",
    "5": "S",
}

NOISE_CHARS = r"[^\x00-\x7F]+"

# ─────────────────────────────────────────────────────────────────────────────
# ACCURACY IMPROVEMENT: Character confusion pairs
# FIX: Removed duplicate key "1" — Python dicts keep last value only.
#      Split into position-aware maps used by TextCleaner.
# ─────────────────────────────────────────────────────────────────────────────

# digit → letter  (used inside alphabetic tokens)
CHAR_CONFUSION_DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",   # leading position → capital I
    "5": "S",
    "6": "G",
    "8": "B",
}

# letter → digit  (used inside numeric tokens)
CHAR_CONFUSION_LETTER_TO_DIGIT = {
    "O": "0",
    "l": "1",
    "I": "1",
    "S": "5",
    "Z": "2",
    "B": "8",
}

# Legacy map kept for any code that imports CHAR_CONFUSION_MAP directly
CHAR_CONFUSION_MAP = {
    "0": "O",
    "1": "I",
    "5": "S",
    "6": "G",
    "8": "B",
    "O": "0",
    "l": "1",
    "S": "5",
    "Z": "2",
    "B": "8",
}

# CRITICAL: ! vs l/I disambiguation context
EXCLAMATION_CONTEXT_WORDS = frozenset(
    "great good wow yes amazing wonderful fantastic excellent brilliant"
    " perfect awesome incredible outstanding superb beautiful lovely"
    " terrible awful bad horrible dreadful".split()
)

# ─────────────────────────────────────────────────────────────────────────────
# ENTITY EXTRACTION PATTERNS
# FIX: Added ZIP/pincode pattern; broadened phone to accept Indian pincodes
#      embedded near phone numbers; kept all original patterns.
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_PATTERNS = {
    "email":    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",

    # FIX: phone pattern now also captures Indian mobile (10-digit starting 6-9)
    # and includes optional country code prefix (+91, 0, etc.)
    "phone":    r"(\+?(?:91|0)?[\s\-]?[6-9]\d{9}|\+?\d[\d\s\-().]{7,}\d)",

    "date":     r"\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}|\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}|"
                r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
                r"\s+\d{1,2},?\s+\d{4})\b",
    "url":      r"https?://[^\s<>\"{}|\\^`\[\]]+",
    "currency": r"(\$|€|£|¥|₹)\s?\d[\d,]*(?:\.\d{1,2})?|\d[\d,]*(?:\.\d{1,2})?\s*(?:USD|EUR|GBP|JPY|INR)",

    # FIX — NEW: ZIP / pincode extraction (Indian 6-digit + US 5-digit/ZIP+4)
    "zipcode":  r"\b(\d{6}|\d{5}(?:-\d{4})?)\b",
}

# ─────────────────────────────────────────────────────────────────────────────
# PDF SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

PDF_DPI           = 300
PDF_MAX_PAGES     = 50
PDF_IMAGE_FORMAT  = "PNG"

# ─────────────────────────────────────────────────────────────────────────────
# PDF TEXT RENDERING SETTINGS (used by pdf_loader._text_to_image)
# FIX: Increased page_width and font size for better spacing fidelity
# ─────────────────────────────────────────────────────────────────────────────

PDF_RENDER_PAGE_WIDTH   = 2480   # pixels — slightly wider for margin safety
PDF_RENDER_MARGIN       = 120    # px
PDF_RENDER_FONT_SIZE    = 28     # px — slightly larger for clearer OCR
PDF_RENDER_LINE_SPACING = 1.8    # multiplier — more generous line spacing
PDF_RENDER_PARA_GAP     = 20     # extra px for blank-line paragraph gaps

# ─────────────────────────────────────────────────────────────────────────────
# GUI SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

APP_TITLE   = "Smart OCR & Handwriting Recognition"
APP_WIDTH   = 1400
APP_HEIGHT  = 860
APP_MIN_W   = 1100
APP_MIN_H   = 700

THEME = {
    "bg":           "#0f1117",
    "bg_secondary": "#1a1d27",
    "bg_panel":     "#12151e",
    "bg_card":      "#1e2130",
    "accent":       "#4f9cf9",
    "accent_dim":   "#2d5fa8",
    "success":      "#3ecf8e",
    "warning":      "#f59e0b",
    "error":        "#ef4444",
    "text":         "#e2e8f0",
    "text_dim":     "#8892a4",
    "text_muted":   "#4a5568",
    "border":       "#2d3748",
    "btn_hover":    "#3b82f6",
    "tag_bg":       "#1e3a5f",
}

FONT_FAMILY = "Segoe UI" if sys.platform == "win32" else "SF Pro Display" if sys.platform == "darwin" else "Ubuntu"
FONT_MONO   = "Consolas"  if sys.platform == "win32" else "Menlo"         if sys.platform == "darwin" else "Ubuntu Mono"

FONT_TITLE   = (FONT_FAMILY, 18, "bold")
FONT_LABEL   = (FONT_FAMILY, 10)
FONT_LABEL_SM= (FONT_FAMILY, 9)
FONT_BUTTON  = (FONT_FAMILY, 10, "bold")
FONT_STATUS  = (FONT_FAMILY, 9)
FONT_TEXT    = (FONT_MONO,   10)
FONT_HEADING = (FONT_FAMILY, 11, "bold")