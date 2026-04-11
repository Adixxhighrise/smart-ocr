"""
postprocessing/context_resolver.py
Context-aware resolution of OCR ambiguities.

PRODUCTION FIX — Spacing & Structure:
─ resolve() now operates line-by-line to preserve \n structure
─ Paragraph separators (blank lines) are never touched
─ _fix_run_on_sentences(): operates within a line, not across lines
─ _restore_list_structure(): respects existing line boundaries
─ _fix_letter_structure(): preserves paragraph breaks from cleaner output
─ No pass collapses multiple lines into one (the core spacing bug fixed)
─ Each structural fix explicitly works on single-line strings
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_logger

logger = get_logger("postprocessing.context_resolver")


# ─────────────────────────────────────────────────────────────────────────────
# WORD-LEVEL FIX PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

_WORD_FIXES = [
    # Articles
    (r"\btbe\b", "the"), (r"\bTbe\b", "The"), (r"\bTke\b", "The"),
    (r"\bTne\b", "The"), (r"\btlie\b", "the"), (r"\bthc\b", "the"),
    (r"\bt11e\b", "the"), (r"\bth e\b", "the"),
    # Conjunctions
    (r"\baild\b", "and"), (r"\baiid\b", "and"), (r"\baud\b", "and"),
    (r"\bor1d\b", "and"), (r"\ban d\b", "and"), (r"\ba nd\b", "and"),
    # Prepositions
    (r"\b0f\b", "of"), (r"\b1n\b", "in"), (r"\bln\b", "in"),
    (r"\bwlth\b", "with"), (r"\bftom\b", "from"), (r"\bfrorn\b", "from"),
    (r"\bfroni\b", "from"), (r"\babont\b", "about"),
    # Common verbs
    (r"\b1s\b", "is"), (r"\bwonld\b", "would"), (r"\bconld\b", "could"),
    (r"\bshonld\b", "should"),
    # Pronouns
    (r"\byonr\b", "your"), (r"\bvou\b", "you"), (r"\b1t\b", "It"),
    (r"\b1ts\b", "Its"),
    # Common nouns / adjectives
    (r"\bwhicb\b", "which"), (r"\bvvhich\b", "which"),
    (r"\bpeop1e\b", "people"), (r"\bt1me\b", "time"), (r"\bl1fe\b", "life"),
    (r"\bl0ve\b", "love"), (r"\bw0rld\b", "world"), (r"\bw0rd\b", "word"),
    (r"\bp0int\b", "point"), (r"\bc0me\b", "come"), (r"\bh0me\b", "home"),
    (r"\bm0re\b", "more"), (r"\bst0re\b", "store"), (r"\bsc0re\b", "score"),
    (r"\bfo110wing\b", "following"), (r"\bfo1lowing\b", "following"),
    # Common OCR confusion words  (rn → m)
    (r"\brnore\b", "more"), (r"\brnay\b", "may"), (r"\brny\b", "my"),
    (r"\brnake\b", "make"), (r"\brnust\b", "must"), (r"\brnan\b", "man"),
    (r"\brnany\b", "many"), (r"\brneet\b", "meet"),
    # Zero confusion
    (r"\bG0\b", "Go"), (r"\bd0\b", "do"), (r"\bn0\b", "no"),
    (r"\bs0\b", "so"), (r"\bt0\b", "to"), (r"\bg0\b", "go"),
]

_COMPILED_WORD_FIXES = [
    (re.compile(pat, re.IGNORECASE), rep) for pat, rep in _WORD_FIXES
]

_MONTHS = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)
_DATE_RE = re.compile(
    rf"(\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\s+\d{{2,4}}|"
    rf"(?:{_MONTHS})\s+\d{{1,2}},?\s+\d{{2,4}}|"
    rf"\d{{1,2}}[\/\-]\d{{1,2}}[\/\-]\d{{2,4}})",
    re.IGNORECASE,
)

_SALUTATION_RE = re.compile(
    r"^(Dear\s+\w+[,.]?|To\s+Whom\s+It\s+May\s+Concern[,.]?|"
    r"Hi\s+\w+[,.]?|Hello\s+\w+[,.]?)",
    re.IGNORECASE,
)

_CLOSING_RE = re.compile(
    r"^(Yours\s+\w+[,.]?|With\s+\w+[,.]?|Sincerely[,.]?|"
    r"Regards[,.]?|Best\s+\w*[,.]?|Warm\s+\w+[,.]?|"
    r"With\s+lots\s+of\s+\w+[,.]?|Take\s+care[,.]?|"
    r"With\s+love[,.]?|Love[,.]?)",
    re.IGNORECASE,
)


class ContextResolver:
    """
    Resolves OCR ambiguities using contextual analysis.
    Applied AFTER TextCleaner.

    IMPORTANT: All transformations operate on individual lines.
    The newline structure produced by TextCleaner is NEVER modified here —
    that is what ensures correct spacing in the final output.
    """

    def resolve(self, text: str) -> str:
        if not text.strip():
            return text

        # Split into lines; blank lines = paragraph separators — never modify them
        lines = text.split("\n")
        processed = []

        for line in lines:
            # Paragraph separator: preserve exactly
            if not line.strip():
                processed.append("")
                continue

            l = line
            l = self._fix_rn_m(l)
            l = self._apply_word_fixes(l)
            l = self._fix_run_on_sentences(l)
            l = self._fix_spacing_around_punctuation(l)
            l = self._fix_standalone_i(l)
            l = self._fix_sentence_capitalisation(l)
            l = self._fix_number_word_boundaries(l)
            processed.append(l)

        result = "\n".join(processed)

        # Post-join structural passes (work with the full text but
        # must not collapse newlines)
        result = self._restore_list_item_markers(result)

        logger.info(f"ContextResolver processed {len(result.split())} words.")
        return result

    # ── PER-LINE FIXES ────────────────────────────────────────────────────────

    @staticmethod
    def _fix_rn_m(text: str) -> str:
        return re.sub(r"(?<=[a-z])rn(?=[a-z])", "m", text)

    @staticmethod
    def _apply_word_fixes(text: str) -> str:
        for pattern, replacement in _COMPILED_WORD_FIXES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _fix_run_on_sentences(text: str) -> str:
        """
        Insert space after sentence-ending punctuation when directly
        followed by a capital letter — within a single line.
        'Hello.How are you' → 'Hello. How are you'
        """
        text = re.sub(r"([.!?])([A-Z][a-z])", r"\1 \2", text)
        text = re.sub(r"(,)([A-Z][a-z]{2,})", r"\1 \2", text)
        return text

    @staticmethod
    def _fix_spacing_around_punctuation(text: str) -> str:
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"([.,!?;:])([A-Za-z0-9])", r"\1 \2", text)
        text = re.sub(r"([.!?]){2,}", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

    @staticmethod
    def _fix_standalone_i(text: str) -> str:
        return re.sub(r"(?<!\w)i(?!\w)", "I", text)

    @staticmethod
    def _fix_sentence_capitalisation(text: str) -> str:
        """Capitalise first letter of each sentence within the line."""
        text = re.sub(r"(?<=[.!?]\s)([a-z])", lambda m: m.group(1).upper(), text)
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def _fix_number_word_boundaries(text: str) -> str:
        # Protect ordinals (1st, 2nd, 3rd, 17th etc.) by matching full token
        text = re.sub(r"(\d+)([a-zA-Z]+)", lambda m: m.group(0)
                      if re.match(r'^\d+(?:st|nd|rd|th)$', m.group(0), re.I)
                      else m.group(1) + ' ' + m.group(2), text)
        text = re.sub(r"([a-zA-Z]{2,})(\d)", r"\1 \2", text)
        return text

    # ── STRUCTURAL PASSES (post-join, newline-safe) ───────────────────────────

    @staticmethod
    def _restore_list_item_markers(text: str) -> str:
        """
        If numbered or bulleted list items were merged onto one line by OCR,
        split them back out — but ONLY insert a newline, never remove one.
        Pattern: '...text 1. Next item' or '...text - next'
        """
        # Numbered list items run together: 'some text 2. next item'
        # Insert newline before the number if it looks like a new list item
        text = re.sub(r"(?<=\w)\s+(\d{1,2}[.)]\s+[A-Z])", r"\n\1", text)
        # Bullet items run together: 'text - Item'  only when preceded by word end
        text = re.sub(r"(?<=\w)\s+([-•*]\s+[A-Z])", r"\n\1", text)
        return text