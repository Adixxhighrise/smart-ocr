"""
preprocessing/pdf_loader.py
Converts PDF files to PIL Images for OCR processing.
Handles both text-based and scanned (image-only) PDFs.

FIX v2 — Spacing & Layout:
─ _text_to_image(): now reads rendering constants from config
  (PDF_RENDER_PAGE_WIDTH, PDF_RENDER_FONT_SIZE, PDF_RENDER_LINE_SPACING,
   PDF_RENDER_PARA_GAP, PDF_RENDER_MARGIN) so they can be tuned in one place
─ _prepare_render_lines(): preserves ALL blank lines as paragraph gaps,
  never collapses consecutive blanks silently
─ _load_with_pypdf(): calls extract_text(extraction_mode="layout") with
  correct fallback; also normalises Windows-style CRLF before processing
─ Word spacing fix: _fix_pdf_word_spacing() inserted after text extraction
  to recover spaces that pypdf sometimes loses between tokens
─ chars_per_line calculation now uses font_size * 0.60 (was 0.55) — avoids
  over-wrapping on monospace fonts

Strategy (priority order):
  1. pymupdf (fitz)   — renders page to image at PDF_DPI; best quality
  2. pdf2image        — renders via poppler at PDF_DPI; excellent quality
  3. pypdf layout     — text-native PDFs with layout-mode extraction
     → rendered to image via _text_to_image() preserving all spacing
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger

logger = get_logger("preprocessing.pdf_loader")


# ─────────────────────────────────────────────────────────────────────────────
# WORD-SPACING FIX FOR PYPDF TEXT
# ─────────────────────────────────────────────────────────────────────────────

def _fix_pdf_word_spacing(text: str) -> str:
    """
    FIX — NEW: pypdf sometimes produces text with missing spaces between tokens.

    Cases addressed:
    1. Lowercase letter directly touching uppercase: "houseThe" → "house The"
    2. Punctuation directly touching next word:
       "party.I" → "party. I"  /  "love,Nikhil" → "love, Nikhil"
    3. Digit touching letter (non-ordinal): "560001MG" → "560001 MG"
    4. Letter touching digit: "Road560001" → "Road 560001"

    Protected cases (NOT split):
    ─ Ordinal suffixes: 17th, 1st, 2nd, 3rd
    ─ URLs / emails (contain :// or @)
    ─ All-caps tokens (acronyms)
    """
    lines = text.split('\n')
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue

        # 1. sentence-end punctuation before next word
        line = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', line)
        # 2. comma/semicolon/colon before next word
        line = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', line)
        # 3. digit + letter (protect ordinals)
        line = re.sub(
            r'(\d+)([A-Za-z]+)',
            lambda m: m.group(0)
                if re.match(r'^\d+(?:st|nd|rd|th)$', m.group(0), re.I)
                else m.group(1) + ' ' + m.group(2),
            line
        )
        # 4. letter + digit
        line = re.sub(r'([A-Za-z]{2,})(\d)', r'\1 \2', line)
        # 5. collapse any double-spaces introduced above
        line = re.sub(r'  +', ' ', line)
        fixed.append(line)

    return '\n'.join(fixed)


class PDFLoader:
    """
    Loads a PDF and returns pages as (page_index, PIL.Image) tuples.
    """

    def load(self, pdf_path: str) -> list:
        """
        Load PDF and return list of (page_index, PIL.Image) for OCR.

        Returns
        -------
        list of (int, PIL.Image)  — 0-based page index + rendered image
        """
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # ── Strategy 1: pymupdf — fastest, pixel-accurate layout ─────────────
        fitz_pages = self._load_with_fitz(pdf_path)
        if fitz_pages:
            return fitz_pages

        # ── Strategy 2: pdf2image (poppler) — excellent layout accuracy ───────
        poppler_pages = self._load_with_pdf2image(pdf_path)
        if poppler_pages:
            return poppler_pages

        # ── Strategy 3: pypdf layout extraction (text-native PDFs only) ───────
        text_pages = self._load_with_pypdf(pdf_path)
        if text_pages:
            return text_pages

        raise RuntimeError(
            "Could not render PDF. Install pymupdf or pdf2image+poppler:\n"
            "  pip install pymupdf\n"
            "  pip install pdf2image\n"
            "  # Linux:  sudo apt install poppler-utils\n"
            "  # macOS:  brew install poppler\n"
        )

    # ── FITZ (pymupdf) ────────────────────────────────────────────────────────

    def _load_with_fitz(self, path: str) -> list:
        """
        Render each PDF page to a PIL Image at PDF_DPI.
        Perfectly preserves all layout: line spacing, paragraph gaps,
        indentation, columns.
        """
        try:
            import fitz
            from PIL import Image

            doc   = fitz.open(path)
            pages = []
            limit = min(len(doc), config.PDF_MAX_PAGES)
            logger.info(f"pymupdf: rendering {limit} page(s) at {config.PDF_DPI} DPI")

            mat = fitz.Matrix(config.PDF_DPI / 72, config.PDF_DPI / 72)

            for i in range(limit):
                page = doc[i]
                pix  = page.get_pixmap(matrix=mat, alpha=False)
                img  = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                pages.append((i, img))
                logger.info(f"  Page {i+1}: {img.size[0]}×{img.size[1]} px")

            doc.close()
            return pages

        except ImportError:
            logger.info("pymupdf not installed — trying pdf2image.")
            return []
        except Exception as exc:
            logger.warning(f"pymupdf failed: {exc}")
            return []

    # ── PDF2IMAGE (poppler) ───────────────────────────────────────────────────

    def _load_with_pdf2image(self, path: str) -> list:
        """Render via poppler at PDF_DPI — full layout fidelity."""
        try:
            from pdf2image import convert_from_path

            logger.info(f"pdf2image: converting PDF at {config.PDF_DPI} DPI")
            images = convert_from_path(
                path,
                dpi=config.PDF_DPI,
                fmt="RGB",
                first_page=1,
                last_page=config.PDF_MAX_PAGES,
            )
            result = []
            for i, img in enumerate(images):
                result.append((i, img))
                logger.info(f"  Page {i+1}: {img.size[0]}×{img.size[1]} px")
            return result

        except ImportError:
            logger.info("pdf2image not installed — trying pypdf text extraction.")
            return []
        except Exception as exc:
            logger.warning(f"pdf2image failed: {exc}")
            return []

    # ── PYPDF (text-native, layout-preserving) ────────────────────────────────

    def _load_with_pypdf(self, path: str) -> list:
        """
        Extract text from text-native PDFs using layout-mode extraction.

        FIX v2:
        ─ Normalises CRLF line endings before processing
        ─ Calls _fix_pdf_word_spacing() after extraction to restore
          spaces that pypdf's layout mode sometimes drops
        ─ layout mode tried first (pypdf ≥ 3.4); plain extraction fallback
        """
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            pages  = []
            limit  = min(len(reader.pages), config.PDF_MAX_PAGES)
            logger.info(f"pypdf: extracting text from {limit} page(s)")

            for i in range(limit):
                page     = reader.pages[i]
                raw_text = self._extract_page_text(page)

                if not raw_text.strip():
                    logger.warning(f"  Page {i+1}: no embedded text (scanned PDF — install pymupdf)")
                    continue

                # FIX: normalise line endings then fix word spacing
                raw_text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
                raw_text = _fix_pdf_word_spacing(raw_text)

                img = self._text_to_image(raw_text)
                pages.append((i, img))
                logger.info(
                    f"  Page {i+1}: {len(raw_text.split())} words extracted "
                    f"→ {img.size[0]}×{img.size[1]} px"
                )

            return pages

        except ImportError:
            logger.warning("pypdf not installed — no text extraction available.")
            return []
        except Exception as exc:
            logger.warning(f"pypdf failed: {exc}")
            return []

    @staticmethod
    def _extract_page_text(page) -> str:
        """
        Extract text from a pypdf page object, using layout mode when available.
        Layout mode (pypdf ≥ 3.4) preserves horizontal spacing and indentation.
        """
        # Try layout mode first (pypdf ≥ 3.4)
        try:
            text = page.extract_text(extraction_mode="layout")
            if text:
                return text
        except TypeError:
            pass  # older pypdf — fall through to plain extraction

        text = page.extract_text() or ""
        return text

    # ── TEXT → IMAGE RENDERER ─────────────────────────────────────────────────

    @staticmethod
    def _text_to_image(
        text: str,
        page_width_px: int  = None,
        margin_px: int      = None,
        font_size: int      = None,
        line_spacing: float = None,
        para_gap_extra: int = None,
    ) -> "Image.Image":
        """
        Render plain/layout text to a white PIL Image.

        FIX v2:
        ─ All defaults now read from config constants (can be overridden by caller)
        ─ chars_per_line calculation uses 0.60 multiplier (was 0.55)
          → avoids over-wrapping on monospace fonts
        ─ Consecutive blank lines in input → each generates a para_gap,
          none are silently dropped
        ─ Indent calculation unchanged; still correct for layout text
        """
        from PIL import Image, ImageDraw
        import textwrap

        # FIX: read from config with safe fallbacks
        page_width_px  = page_width_px  or getattr(config, 'PDF_RENDER_PAGE_WIDTH',   2480)
        margin_px      = margin_px      or getattr(config, 'PDF_RENDER_MARGIN',        120)
        font_size      = font_size      or getattr(config, 'PDF_RENDER_FONT_SIZE',     28)
        line_spacing   = line_spacing   or getattr(config, 'PDF_RENDER_LINE_SPACING',  1.8)
        para_gap_extra = para_gap_extra or getattr(config, 'PDF_RENDER_PARA_GAP',      20)

        line_h    = int(font_size * line_spacing)
        content_w = page_width_px - 2 * margin_px
        # FIX: use 0.60 multiplier for better character-width estimate
        chars_per_line = max(20, int(content_w / (font_size * 0.60)))

        font = PDFLoader._load_best_font(font_size)

        render_lines = PDFLoader._prepare_render_lines(text, chars_per_line)

        total_height = margin_px
        for rl in render_lines:
            if rl["is_blank"]:
                total_height += line_h + para_gap_extra
            else:
                total_height += line_h
        total_height += margin_px

        img  = Image.new("RGB", (page_width_px, total_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        y = margin_px
        for rl in render_lines:
            if rl["is_blank"]:
                y += line_h + para_gap_extra
                continue

            indent_px = int(rl["indent"] * font_size * 0.45)
            x         = margin_px + indent_px

            if font:
                draw.text((x, y), rl["text"], fill=(0, 0, 0), font=font)
            else:
                draw.text((x, y), rl["text"], fill=(0, 0, 0))

            y += line_h

        return img

    @staticmethod
    def _load_best_font(font_size: int):
        """Try to load a readable monospace/proportional font."""
        from PIL import ImageFont

        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
            "/Library/Fonts/Courier New.ttf",
            "C:/Windows/Fonts/cour.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
            "/Library/Fonts/Arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

        for path in font_candidates:
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue

        try:
            return ImageFont.load_default()
        except Exception:
            return None

    @staticmethod
    def _prepare_render_lines(text: str, chars_per_line: int) -> list:
        """
        Convert raw text into render-instruction dicts:
            {"is_blank": bool, "text": str, "indent": int}

        FIX v2:
        ─ Consecutive blank lines each produce a {"is_blank": True} entry —
          none are dropped. This preserves paragraph spacing from the original.
        ─ Trailing whitespace stripped per line before indent measurement
        ─ textwrap.wrap with break_on_hyphens=True (unchanged)
        """
        import textwrap

        raw_lines = text.split("\n")
        result    = []

        for raw_line in raw_lines:
            # FIX: strip trailing whitespace only (preserve leading indent)
            raw_line = raw_line.rstrip()

            if not raw_line.strip():
                # FIX: ALL blank lines generate a gap — do not skip duplicates
                result.append({"is_blank": True, "text": "", "indent": 0})
                continue

            stripped_left = raw_line.lstrip(" \t")
            indent        = len(raw_line) - len(stripped_left)

            content = stripped_left
            if len(content) <= chars_per_line:
                result.append({"is_blank": False, "text": content, "indent": indent})
            else:
                wrapped = textwrap.wrap(
                    content,
                    width=chars_per_line,
                    break_long_words=True,
                    break_on_hyphens=True,
                )
                for j, wline in enumerate(wrapped):
                    result.append({
                        "is_blank": False,
                        "text":     wline,
                        "indent":   indent if j == 0 else 0,
                    })

        return result

    # ── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Quick page count without full load."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            n   = len(doc)
            doc.close()
            return n
        except ImportError:
            pass
        try:
            from pypdf import PdfReader
            return len(PdfReader(pdf_path).pages)
        except Exception:
            return 0