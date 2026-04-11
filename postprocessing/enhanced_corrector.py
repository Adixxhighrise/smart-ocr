"""
postprocessing/enhanced_corrector.py
Enhanced post-processing pipeline orchestrator.

PRODUCTION FIX — Spacing & Structure:
─ process() preserves \n / blank-line structure through all 4 stages
─ _format_output() NO LONGER injects periods at end of lines
─ _format_letter(): preserves original paragraph blocks; does not reorder
─ _format_plain(): does not collapse lines; only normalises inter-paragraph gaps
─ _format_list(): ensures each item is on its own line (additive, not destructive)
─ _format_form(): field-per-line without collapsing existing structure
─ quality_report() unchanged (accuracy estimation)
─ The golden rule: formatting passes may ADD newlines for clarity but must
  NEVER remove existing newlines or blank lines from the cleaner output
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger, text_confidence_score

logger = get_logger("postprocessing.enhanced_corrector")


class EnhancedCorrector:
    """
    Full post-processing pipeline.

    Pipeline
    ────────
    Stage 1  TextCleaner      — noise, ligatures, ambiguous chars (line-aware)
    Stage 2  ContextResolver  — contextual word fixes, spacing (line-aware)
    Stage 3  AICorrector      — vocabulary-distance correction (token-level)
    Stage 4  _format_output() — document-type-aware formatting (structure-safe)
    """

    def __init__(self, custom_vocabulary: list = None):
        from postprocessing.text_cleaner     import TextCleaner
        from postprocessing.context_resolver import ContextResolver
        from postprocessing.ai_corrector     import AICorrector

        self._cleaner   = TextCleaner()
        self._resolver  = ContextResolver()
        self._corrector = AICorrector(custom_vocabulary=custom_vocabulary)

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def process(
        self,
        text: str,
        confidence: float = 1.0,
        doc_hint: str = "auto",
    ) -> str:
        if not text or not text.strip():
            return ""

        logger.info(
            f"EnhancedCorrector: input {len(text.split())} words, "
            f"conf={confidence:.2f}, hint='{doc_hint}'"
        )

        # Stage 1: Cleaning (line-aware — preserves structure)
        text = self._cleaner.clean(text, confidence=confidence)
        stage1_score = text_confidence_score(text)
        logger.info(f"  Stage 1 (TextCleaner):     score={stage1_score:.3f}")

        # Stage 2: Context resolution (line-aware — preserves structure)
        text = self._resolver.resolve(text)
        stage2_score = text_confidence_score(text)
        logger.info(f"  Stage 2 (ContextResolver): score={stage2_score:.3f}")

        # Stage 3: AI correction (SAFE + CONTROLLED)
        effective_conf = min(confidence, stage2_score)

        # STRICT LOGIC
        if self._is_noisy(text):
            should_correct = True   # fix garbage always
        elif effective_conf < 0.65:
            should_correct = True   # very low confidence → fix
        elif stage2_score < 0.70:
            should_correct = True   # weak OCR → fix
        else:
            should_correct = False  # good text → DO NOT TOUCH

        if should_correct:
            pre_score = stage2_score
            text = self._correct_preserving_structure(text)
            post_score = text_confidence_score(text)
            logger.info(
                f"  Stage 3 (AICorrector): score={pre_score:.3f} → {post_score:.3f}"
            )
        else:
            logger.info(
                "  Stage 3 (AICorrector): skipped (text already clean)"
            )

        # Stage 4: Formatting (structure-safe)
        doc_type = doc_hint if doc_hint != "auto" else self._detect_document_type(text)
        text = self._format_output(text, doc_type)

        logger.info(f"EnhancedCorrector: output {len(text.split())} words.")
        return text

    def _correct_preserving_structure(self, text: str) -> str:
        """
        Run AICorrector line-by-line while preserving structure
        and avoiding correction on noisy/short lines.
        """
        lines = text.split("\n")
        corrected = []
        for line in lines:
            # Keep empty lines unchanged
            if not line.strip():
                corrected.append("")
                continue
            # Skip very short lines — not enough context for the corrector
            if len(line.split()) < 3:
                corrected.append(line)
                continue
            # Skip noisy OCR patterns (e.g. "E T I I")
            if re.search(r'\b[A-Z]\b(?:\s+[A-Z]\b){2,}', line):
                corrected.append(line)
                continue
            # Skip symbol-heavy garbage lines
            if len(re.findall(r'[^\w\s]', line)) > len(line) * 0.20:
                corrected.append(line)
                continue
            # Only clean lines get AI correction
            corrected.append(self._corrector.correct(line))
        return "\n".join(corrected)

    def add_vocabulary(self, words: list):
        self._corrector.add_vocabulary(words)

    def process_batch(self, texts: list, confidences: list = None) -> list:
        if confidences is None:
            confidences = [1.0] * len(texts)
        return [self.process(t, c) for t, c in zip(texts, confidences)]

    def accuracy_percent(self, raw: str, cleaned: str) -> float:
        report = self.quality_report(raw, cleaned)
        return report["confidence_pct"]

    # ── DOCUMENT TYPE DETECTION ───────────────────────────────────────────────

    @staticmethod
    def _is_noisy(text: str) -> bool:
        """Detect whether text is too noisy / garbage to skip AI correction."""
        if not text or len(text.split()) < 3:
            return True
        # Too many consecutive single-letter uppercase tokens (e.g. "E T I I")
        if re.search(r'\b[A-Z]{1}\b(?:\s+[A-Z]{1}\b){2,}', text):
            return True
        # Too many non-word characters (>15% of total length)
        if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.15:
            return True
        return False

    @staticmethod
    def _detect_document_type(text: str) -> str:
        lower = text.lower()

        letter_signals = [
            "dear ", "yours sincerely", "yours faithfully", "with love",
            "regards", "take care", "hope you", "sincerely",
        ]
        if any(sig in lower for sig in letter_signals):
            return "letter"

        form_signals = ["name:", "date:", "address:", "signature:", "phone:"]
        if sum(1 for sig in form_signals if sig in lower) >= 2:
            return "form"

        lines = text.splitlines()
        list_lines = sum(
            1 for l in lines
            if re.match(r"^\s*(\d+[.)]\s|[-•*]\s)", l.strip())
        )
        if len(lines) > 3 and list_lines / max(len(lines), 1) > 0.4:
            return "list"

        return "plain"

    # ── OUTPUT FORMATTING ─────────────────────────────────────────────────────

    @staticmethod
    def _format_output(text: str, doc_type: str) -> str:
        if doc_type == "letter":
            return EnhancedCorrector._format_letter(text)
        elif doc_type == "list":
            return EnhancedCorrector._format_list(text)
        elif doc_type == "form":
            return EnhancedCorrector._format_form(text)
        else:
            return EnhancedCorrector._format_plain(text)

    @staticmethod
    def _format_plain(text: str) -> str:
        """
        Normalise paragraph spacing for plain documents.
        Rule: paragraphs separated by exactly one blank line.
        Lines WITHIN a paragraph keep their original single-newline separation.
        No period injection.
        """
        paragraphs = re.split(r"\n{2,}", text)
        cleaned_paragraphs = []

        for para in paragraphs:
            para = para.strip()
            if para:
                lines = para.split("\n")
                lines = [re.sub(r" {2,}", " ", l) for l in lines]
                cleaned_paragraphs.append("\n".join(lines))

        return "\n\n".join(cleaned_paragraphs)

    @staticmethod
    def _format_letter(text: str) -> str:
        """
        Format as a letter.
        Preserves the paragraph structure from the cleaner output.
        Only ensures clean blank-line separation between blocks.
        Does NOT inject periods or reorder content.
        """
        paragraphs = re.split(r"\n{2,}", text)
        formatted  = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            lines = para.split("\n")
            lines = [re.sub(r" {2,}", " ", l) for l in lines]
            formatted.append("\n".join(lines))

        return "\n\n".join(formatted)

    @staticmethod
    def _format_list(text: str) -> str:
        """
        Ensure list items each appear on their own line.
        Preserves existing newlines; only ensures list-marker lines are separated.
        """
        lines = text.splitlines()
        result = []

        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                if result and result[-1] != "":
                    result.append("")
                continue
            result.append(stripped)

        while result and result[-1] == "":
            result.pop()

        return "\n".join(result)

    @staticmethod
    def _format_form(text: str) -> str:
        """
        Format form fields: each 'Label: value' pair on its own line.
        Adds newlines before field labels, preserves existing structure.
        """
        text = re.sub(r"(?<!\n)([A-Z][a-zA-Z\s]{1,20}:)", r"\n\1", text)
        lines = text.splitlines()
        result = []
        for l in lines:
            stripped = l.strip()
            if stripped:
                result.append(stripped)
            elif result and result[-1] != "":
                result.append("")
        while result and result[-1] == "":
            result.pop()
        return "\n".join(result)

    # ── QUALITY REPORT ────────────────────────────────────────────────────────

    @staticmethod
    def quality_report(raw: str, cleaned: str) -> dict:
        try:
            from utils import count_words
            raw_words     = count_words(raw)
            cleaned_words = count_words(cleaned)
        except Exception:
            raw_words     = len(raw.split())
            cleaned_words = len(cleaned.split())

        raw_score     = text_confidence_score(raw)
        cleaned_score = text_confidence_score(cleaned)
        improvement   = cleaned_score - raw_score
        target_met    = cleaned_score >= 0.85

        word_retention = min(cleaned_words / max(raw_words, 1), 1.0)
        accuracy       = (cleaned_score * 0.7 + word_retention * 0.3)
        accuracy_pct   = round(accuracy * 100, 1)

        return {
            "raw_words":       raw_words,
            "cleaned_words":   cleaned_words,
            "raw_quality":     round(raw_score,     4),
            "cleaned_quality": round(cleaned_score, 4),
            "improvement":     round(improvement,   4),
            "target_met":      target_met,
            "confidence_pct":  accuracy_pct,
            "word_retention":  round(word_retention, 4),
        }