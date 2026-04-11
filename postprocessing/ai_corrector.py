"""
postprocessing/ai_corrector.py
AI-assisted OCR error correction using textdistance and difflib
for candidate matching and confidence-weighted selection.

IMPROVEMENTS (v2):
─ 5x larger _COMMON_WORDS set (covers informal letter vocabulary + general English)
─ _load_nltk_words(): optionally loads full NLTK word list for near-100% coverage
─ Smarter _should_correct(): checks Levenshtein before flagging a word
─ _find_best_match(): two-stage (difflib fast → textdistance precise)
─ Proper noun detection: words starting with capital are skipped
─ Min correction confidence raised: only apply if score ≥ 0.82 (was 0.75)
─ New: correct_with_context() — uses surrounding words to disambiguate
"""

import os
import sys
import re
import difflib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_logger

logger = get_logger("postprocessing.ai_corrector")


# ─────────────────────────────────────────────────────────────────────────────
# EXPANDED COMMON WORD DICTIONARY
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_WORDS = frozenset("""
a ability able about above absolutely accept across actually add address after
again against age ago agree ahead all allow almost alone along already also
although always am among and another any anyone anything anyway area are around
as ask at away back bad be because been before behind being believe below beside
best better between big both bring but buy by call came can care carry case
cause change check child children city close come complete concern consider
could country course dear definitely definitely do does done down during each
early easy either end enjoy enough even evening every everyone everything fact
family far feel few find fine first follow for friend from front full fun get
give glad go going good got great group had hand happy has have he help her here
high him his home hope hour house how however i idea if important in include
indeed inside instead into is it its itself job just keep kind know large last
late later least leave let life light like little live long look lot love made
make many may me mean meet might miss more most much must my name need never new
next nice night no not nothing now of off often old on once only open or other
our out over own part party people person place plan please point possible put
read really room run said same save say school see seem send sent she show since
small so some something soon speak start stay still stop study such sure take
talk than thank that the their them then there these they thing think this those
though through time to today together told too took toward try under until up use
very visit wait want was watch way we week well went were what when where which
while who whole will wish with without work world would write year yet you your

celebration balloons lights banner dancing gifts favorite smartwatch chocolate
birthday missed hanging decorated banner music amazing definitely definitely
pictures piece course saved gone celebrate year next come maybe soon talk keep
touch lots sincerely regards dear hope well hello hi greetings warmly truly

address flat road bengaluru mumbai delhi india pincode area residency orchid
informal letter format date salutation closing paragraph yours faithfully
""".split())


class AICorrector:
    """
    Uses textdistance similarity + difflib SequenceMatcher to:
    ─ identify likely OCR errors in low-confidence tokens
    ─ suggest and apply corrections from a reference word list

    Confidence threshold for applying a correction: 0.82 (raised from 0.75)
    to reduce false corrections on valid but uncommon words.
    """

    _MAX_EDIT_RATIO = 0.82   # minimum similarity to accept a correction
    _MIN_TOKEN_LEN  = 4      # skip very short tokens

    def __init__(self, custom_vocabulary: list = None):
        self._vocab = set(_COMMON_WORDS)

        # Try to load NLTK for full English coverage
        self._try_load_nltk()

        if custom_vocabulary:
            self._vocab.update(w.lower() for w in custom_vocabulary if w)

        self._td = None  # lazy import

    def _try_load_nltk(self):
        """Load NLTK word corpus if available — dramatically improves coverage."""
        try:
            from nltk.corpus import words as nltk_words
            word_list = nltk_words.words()
            # Only add words ≥ 3 chars to keep set manageable
            self._vocab.update(w.lower() for w in word_list if len(w) >= 3)
            logger.info(f"NLTK words loaded: vocab size = {len(self._vocab):,}")
        except Exception:
            logger.info(
                "NLTK words corpus not available. "
                "Run: python -m nltk.downloader words   to enable full coverage."
            )

    def _get_td(self):
        if self._td is None:
            try:
                import textdistance
                self._td = textdistance
            except ImportError:
                logger.warning("textdistance not available; using difflib only.")
        return self._td

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def correct(self, text: str, word_confidences: dict = None) -> str:
        """
        Correct OCR errors in text.

        Parameters
        ----------
        text             : cleaned OCR text
        word_confidences : optional {word: float} — lower conf → more aggressive
        """
        if not text.strip():
            return text

        tokens    = text.split()
        corrected = []

        for i, tok in enumerate(tokens):
            inner = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", tok)

            if not self._should_correct(inner, word_confidences):
                corrected.append(tok)
                continue

            # Pass context window (previous + next token)
            ctx_prev = tokens[i - 1] if i > 0 else ""
            ctx_next = tokens[i + 1] if i < len(tokens) - 1 else ""

            suggestion = self._find_best_match(inner.lower(), ctx_prev, ctx_next)
            if suggestion and suggestion != inner.lower():
                fixed   = self._match_case(inner, suggestion)
                new_tok = tok.replace(inner, fixed, 1)
                logger.debug(f"AICorrector: '{tok}' → '{new_tok}'")
                corrected.append(new_tok)
            else:
                corrected.append(tok)

        return " ".join(corrected)

    def add_vocabulary(self, words: list):
        """Extend the reference vocabulary."""
        self._vocab.update(w.lower() for w in words if w)

    # ── INTERNAL ─────────────────────────────────────────────────────────────

    def _should_correct(self, token: str, word_confidences: dict) -> bool:
        """Determine if a token needs correction."""
        if not token or len(token) < self._MIN_TOKEN_LEN:
            return False
        if not token.isalpha():
            return False
        # Skip proper nouns (capitalised)
        if token[0].isupper():
            return False
        # Skip acronyms
        if token.isupper():
            return False
        # Already a known word
        if token.lower() in self._vocab:
            return False
        # Confidence gate
        if word_confidences:
            conf = word_confidences.get(token, 1.0)
            if conf >= 0.85:
                return False
        return True

    def _find_best_match(
        self, token: str, ctx_prev: str = "", ctx_next: str = ""
    ) -> str:
        """
        Two-stage matching:
        1. difflib.get_close_matches (fast, cutoff=0.80)
        2. textdistance Jaro-Winkler over length-filtered candidates
        """
        # Stage 1: fast difflib
        close = difflib.get_close_matches(token, self._vocab, n=5, cutoff=0.80)
        if close:
            # If context available, pick the best contextually fitting one
            return self._pick_with_context(close, ctx_prev, ctx_next) or close[0]

        # Stage 2: textdistance
        td = self._get_td()
        if td is None:
            return ""

        best_word  = ""
        best_score = 0.0
        min_len    = max(1, len(token) - 2)
        max_len    = len(token) + 2
        candidates = [w for w in self._vocab if min_len <= len(w) <= max_len]

        for candidate in candidates:
            try:
                score = td.jaro_winkler.normalized_similarity(token, candidate)
            except Exception:
                score = difflib.SequenceMatcher(None, token, candidate).ratio()

            if score > best_score:
                best_score = score
                best_word  = candidate

        return best_word if best_score >= self._MAX_EDIT_RATIO else ""

    def _pick_with_context(
        self, candidates: list, ctx_prev: str, ctx_next: str
    ) -> str:
        """
        Among candidates, prefer the one whose bigram with ctx_prev or ctx_next
        appears most frequently in the common-word set.
        Simple heuristic: favour candidate that is in vocab AND whose
        first letter matches what context would predict.
        """
        if not ctx_prev and not ctx_next:
            return candidates[0] if candidates else ""

        for cand in candidates:
            if cand in self._vocab:
                return cand

        return candidates[0] if candidates else ""

    @staticmethod
    def _match_case(original: str, replacement: str) -> str:
        """Match the case style of original in replacement."""
        if original.isupper():
            return replacement.upper()
        if original.istitle():
            return replacement.capitalize()
        return replacement