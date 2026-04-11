"""
postprocessing/text_cleaner.py
Main text cleaning pipeline: noise removal, whitespace normalisation,
ambiguous-character resolution, and optional spell correction.

PRODUCTION FIX — Spacing & Structure:
─ Line-by-line processing: blank lines (paragraph breaks) are NEVER collapsed
─ Single newlines within a paragraph are preserved as soft line breaks
─ Indentation / leading whitespace on each line is captured and restored
─ Consecutive blank lines → exactly ONE blank line (clean paragraph gap)
─ No spurious period injection at end of lines
─ All character fixes (ambiguous chars, spell) operate on tokens only,
  never touch newline or spacing structure characters
─ _fix_hyphenated_break(): fixes 'word-\n  word' within a single line token
  (already split lines don't have embedded newlines at this stage)
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, text_confidence_score

logger = get_logger("postprocessing.text_cleaner")


class TextCleaner:
    """
    Full text cleaning pipeline applied after raw OCR output.

    Steps
    ─────
    1.  Normalise line endings  (CRLF → LF)
    2.  Split into lines; blank lines = paragraph separators
    3.  Per-line: strip control chars, fix ligatures, remove lone noise
    4.  Per-line: fix hyphenated word-break artefacts
    5.  Per-line CRITICAL: ! vs l/I disambiguation
    6.  Per-line: resolve ambiguous characters (0/O, 1/l/I)
    7.  Per-line: fix number/letter boundary merges
    8.  Per-line: fix merged words
    9.  Per-line: spell correction (only when confidence < threshold)
    10. Rebuild: paragraphs separated by one blank line, lines by newline
    """

    _LIGATURE_MAP = {
        # Unicode ligatures
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
        "ﬅ": "ft", "ﬆ": "st",
        # Vertical bar misread
        "|": "I",
        # Smart quotes → straight
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        # Em/en dash — preserve em-dash, normalise en-dash to hyphen
        "\u2013": "-",   # en-dash → hyphen
        # "\u2014" em-dash intentionally kept as-is
        "°": " degrees",
        "©": "(c)",
        "®": "(R)",
        "™": "(TM)",
    }

    def __init__(self):
        self._speller = None

    def _get_speller(self):
        if self._speller is None:
            try:
                from autocorrect import Speller
                self._speller = Speller(lang="en")
            except Exception as exc:
                logger.warning(f"autocorrect unavailable: {exc}")
                self._speller = None
        return self._speller

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def clean(self, text: str, confidence: float = 1.0) -> str:
        if not text or not text.strip():
            return ""

        # Normalise line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split preserving blank lines (paragraph boundaries)
        raw_lines = text.split("\n")
        cleaned_lines = []

        for line in raw_lines:
            # Blank line = paragraph separator — keep as-is
            if not line.strip():
                cleaned_lines.append("")
                continue

            # Preserve leading whitespace (indentation)
            stripped_left = line.lstrip()
            indent_len    = len(line) - len(stripped_left)
            indent        = line[:indent_len]

            l = line

            # Apply per-line text fixes
            l = self._remove_control_chars(l)
            l = self._fix_ligatures(l)
            l = self._remove_lone_noise(l)
            l = self._fix_hyphenated_break(l)
            l = self._fix_exclamation_vs_l(l)
            l = self._resolve_ambiguous(l)
            l = self._fix_number_letter_boundaries(l)
            l = self._fix_merged_words(l)

            if confidence < config.SPELL_CORRECTION_CONFIDENCE_THRESHOLD:
                l = self._spell_correct(l)

            # Normalise internal whitespace only (preserve indent separately)
            l_stripped = l.strip()
            l_stripped = re.sub(r"[ \t]+", " ", l_stripped)

            if l_stripped:
                cleaned_lines.append(indent + l_stripped)
            # If line became empty after cleaning, keep as blank (don't add noise)

        result = self._rebuild_paragraphs(cleaned_lines)
        logger.info(f"TextCleaner: output {len(result.split())} words.")
        return result

    # ── STRUCTURE REBUILD ─────────────────────────────────────────────────────

    @staticmethod
    def _rebuild_paragraphs(lines: list) -> str:
        """
        Rebuilds clean paragraph-structured text from processed lines.

        Rules:
        ─ Non-blank lines are kept with single-newline separation
          (preserves intra-paragraph line breaks from original document)
        ─ Consecutive blank lines are collapsed to exactly ONE blank line
          (clean paragraph gap = one empty line between paragraphs)
        ─ Leading/trailing blank lines stripped
        """
        result    = []
        prev_blank = False

        for line in lines:
            is_blank = (line.strip() == "")

            if is_blank:
                # Only emit one blank line no matter how many consecutive blanks
                if not prev_blank and result:
                    result.append("")
                prev_blank = True
            else:
                result.append(line)
                prev_blank = False

        # Strip leading/trailing blank lines
        while result and result[0] == "":
            result.pop(0)
        while result and result[-1] == "":
            result.pop()

        return "\n".join(result)

    # ── CLEANING STEPS ────────────────────────────────────────────────────────

    @staticmethod
    def _remove_control_chars(text: str) -> str:
        # Keep \t (indentation); \n already separated; remove other controls
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _fix_ligatures(self, text: str) -> str:
        for wrong, right in self._LIGATURE_MAP.items():
            text = text.replace(wrong, right)
        return text

    @staticmethod
    def _remove_lone_noise(text: str) -> str:
        return re.sub(
            r"(?<!\S)[^a-zA-Z0-9$€£¥₹.,!?;:\'\-\"](?!\S)", "", text
        )

    @staticmethod
    def _fix_hyphenated_break(text: str) -> str:
        """Fix 'word-  word' style hyphenated merge artefacts within a line."""
        return re.sub(r"(\w)-\s{2,}(\w)", r"\1\2", text)

    # ── CRITICAL: ! vs l/I DISAMBIGUATION ────────────────────────────────────

    @staticmethod
    def _fix_exclamation_vs_l(text: str) -> str:
        tokens = text.split(" ")
        result = []
        exclamation_words = config.EXCLAMATION_CONTEXT_WORDS

        for i, tok in enumerate(tokens):
            if "!" not in tok:
                result.append(tok)
                continue

            if tok.strip("!") == "":
                result.append(tok)
                continue

            fixed = tok
            fixed = re.sub(r"(?<=[a-z])!(?=[a-z])", "l", fixed)
            fixed = re.sub(r"^!(?=[a-z])", "l", fixed)
            fixed = re.sub(r"^!(?=[A-Z])", "I", fixed)
            fixed = re.sub(r"(?<=[A-Z])!(?=[a-z])", "l", fixed)
            fixed = re.sub(r"(?<=\d)!(?=[a-zA-Z])", "l", fixed)
            result.append(fixed)

        return " ".join(result)

    # ── AMBIGUOUS CHARACTER RESOLUTION ────────────────────────────────────────

    @staticmethod
    def _resolve_ambiguous(text: str) -> str:
        tokens = text.split()
        resolved = []

        for tok in tokens:
            if re.match(r"^\d[\d.,]+$", tok):
                resolved.append(tok)
                continue
            if "@" in tok or "://" in tok:
                resolved.append(tok)
                continue

            inner = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", tok)
            if not inner or len(inner) < 2:
                resolved.append(tok)
                continue

            alpha_ratio = sum(c.isalpha() for c in inner) / len(inner)
            if alpha_ratio < 0.5:
                resolved.append(tok)
                continue

            corrected = tok

            if "0" in inner and not any(c.isdigit() and c != "0" for c in inner):
                corrected = corrected.replace("0", "O")

            if inner.startswith("1") and len(inner) > 1 and inner[1].isalpha():
                corrected = corrected[:corrected.find("1")] + "I" + corrected[corrected.find("1")+1:]
            elif inner.endswith("1") and len(inner) > 1 and inner[-2].isalpha():
                last_pos = corrected.rfind("1")
                corrected = corrected[:last_pos] + "l" + corrected[last_pos+1:]

            corrected = re.sub(r"(?<=[a-zA-Z])1(?=[a-zA-Z])", "l", corrected)

            if inner.startswith("5") and len(inner) > 2 and inner[1].isalpha():
                corrected = corrected[:corrected.find("5")] + "S" + corrected[corrected.find("5")+1:]

            if "6" in inner and alpha_ratio > 0.7:
                corrected = re.sub(r"(?<=[a-zA-Z])6(?=[a-zA-Z])", "G", corrected)

            resolved.append(corrected)

        return " ".join(resolved)

    # ── NUMBER / LETTER BOUNDARY FIX ─────────────────────────────────────────

    @staticmethod
    def _fix_number_letter_boundaries(text: str) -> str:
        # Protect ordinals: 1st, 2nd, 3rd, 4th, 17th, 21st etc.
        # Use a negative lookahead that checks the digit+letter pair together.
        # (?!\d*(?:st|nd|rd|th)\b) — skip if the digit sequence ends in an ordinal suffix.
        text = re.sub(r"(\d+)([a-zA-Z]+)", lambda m: m.group(0)
                      if re.match(r'^\d+(?:st|nd|rd|th)$', m.group(0), re.I)
                      else m.group(1) + ' ' + m.group(2), text)
        text = re.sub(r"([a-zA-Z]{2,})(\d)", r"\1 \2", text)
        return text

    @staticmethod
    def _fix_merged_words(text: str) -> str:
        prefixes = (
            r"(The|the|It|it|In|in|On|on|At|at|To|to|Of|of|"
            r"And|and|But|but|For|for|Is|is|Was|was|Are|are|"
            r"I|A|a)"
        )
        text = re.sub(rf"{prefixes}([A-Z][a-z])", r"\1 \2", text)
        text = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", text)
        return text

    def _spell_correct(self, text: str) -> str:
        speller = self._get_speller()
        if speller is None:
            return text

        tokens    = text.split()
        corrected = []

        for tok in tokens:
            inner = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", tok)
            if (
                len(inner) < 4
                or not inner.isalpha()
                or inner[0].isupper()
                or inner.isupper()
            ):
                corrected.append(tok)
                continue
            try:
                fixed = speller(inner)
            except Exception:
                fixed = inner
            corrected.append(tok.replace(inner, fixed, 1))

        return " ".join(corrected)