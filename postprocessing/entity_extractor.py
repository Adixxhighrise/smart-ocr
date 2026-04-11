"""
postprocessing/entity_extractor.py
Extracts structured entities from OCR text:
emails, phone numbers, dates, URLs, currency amounts, ZIP/pincodes.

FIX v2:
─ Added 'zipcode' entity type — extracts Indian 6-digit pincodes and
  US 5-digit / ZIP+4 codes
─ _clean_phone(): now rejects tokens that are pure pincodes (6-digit
  India or 5-digit US) that have no other phone-number characteristics,
  preventing pincode "560001" from being classified as a phone number
─ _clean_zipcode(): validates extracted codes against known ranges
  (Indian pincodes: 100000–999999; US ZIP: 00501–99950)
─ summarise(): includes ZIPCODE in output
"""

import os
import sys
import re
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import get_logger

logger = get_logger("postprocessing.entity_extractor")


class EntityExtractor:
    """
    Regex-based entity extractor.

    Entities detected
    ─────────────────
    ─ email    : standard email addresses
    ─ phone    : international / domestic phone numbers
    ─ date     : numeric and written-form dates
    ─ url      : http/https URLs
    ─ currency : amounts with currency symbol or code
    ─ zipcode  : Indian 6-digit pincodes + US ZIP codes  [NEW]
    """

    def __init__(self):
        self._patterns = {
            entity: re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for entity, pattern in config.ENTITY_PATTERNS.items()
        }

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def extract(self, text: str) -> dict:
        """
        Extract all entities from text.

        Returns
        -------
        dict with keys: 'email', 'phone', 'date', 'url', 'currency', 'zipcode'
        Each value is a deduplicated list of strings.
        """
        results = {}

        # FIX: extract zipcodes FIRST so we can exclude them from phone matches
        zipcode_pattern = self._patterns.get("zipcode")
        zipcode_matches = set()
        if zipcode_pattern:
            for m in zipcode_pattern.finditer(text):
                raw = m.group(0).strip()
                cleaned = self._clean_zipcode(raw)
                if cleaned:
                    zipcode_matches.add(cleaned)
            results["zipcode"] = sorted(zipcode_matches)
            if results["zipcode"]:
                logger.info(f"Found {len(results['zipcode'])} zipcode(s): {results['zipcode'][:3]}")

        for entity_type, pattern in self._patterns.items():
            if entity_type == "zipcode":
                continue  # already processed above
            matches = pattern.findall(text)
            cleaned = self._clean_matches(entity_type, matches, zipcode_matches)
            results[entity_type] = cleaned
            if cleaned:
                logger.info(f"Found {len(cleaned)} {entity_type}(s): {cleaned[:3]}")

        return results

    def extract_with_positions(self, text: str) -> dict:
        """
        Like extract(), but returns (match_text, start, end) tuples.
        """
        results = {}
        for entity_type, pattern in self._patterns.items():
            found = []
            for m in pattern.finditer(text):
                raw = m.group(0).strip()
                cleaned = self._clean_single(entity_type, raw)
                if cleaned:
                    found.append({
                        "text":  cleaned,
                        "start": m.start(),
                        "end":   m.end(),
                    })
            results[entity_type] = found
        return results

    # ── CLEANING ─────────────────────────────────────────────────────────────

    def _clean_matches(
        self, entity_type: str, matches: list, known_zipcodes: set = None
    ) -> list:
        """Deduplicate and clean raw regex matches."""
        seen    = set()
        cleaned = []
        known_zipcodes = known_zipcodes or set()

        for m in matches:
            if isinstance(m, tuple):
                raw = "".join(m).strip()
            else:
                raw = str(m).strip()

            item = self._clean_single(entity_type, raw)
            if not item:
                continue

            # FIX: if this entity_type is 'phone', skip pure zipcodes
            if entity_type == "phone" and item in known_zipcodes:
                logger.debug(f"Skipping phone match '{item}' — it's a zipcode")
                continue

            if item.lower() not in seen:
                seen.add(item.lower())
                cleaned.append(item)
        return cleaned

    def _clean_single(self, entity_type: str, raw: str) -> str:
        """Apply entity-type-specific cleaning."""
        raw = raw.strip()
        if not raw:
            return ""

        if entity_type == "email":
            return self._clean_email(raw)
        if entity_type == "phone":
            return self._clean_phone(raw)
        if entity_type == "date":
            return self._clean_date(raw)
        if entity_type == "url":
            return self._clean_url(raw)
        if entity_type == "currency":
            return self._clean_currency(raw)
        if entity_type == "zipcode":
            return self._clean_zipcode(raw)
        return raw

    @staticmethod
    def _clean_email(raw: str) -> str:
        raw = raw.strip().lower()
        if re.match(r"^[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$", raw):
            return raw
        return ""

    @staticmethod
    def _clean_phone(raw: str) -> str:
        """
        Normalise phone numbers.

        FIX: reject tokens that are ONLY 5-6 digits with no other
        phone-number characters — these are likely pincodes, not phones.
        """
        raw = re.sub(r"[^\d\s\-+().x]", "", raw).strip()
        digits_only = re.sub(r"\D", "", raw)

        # FIX: pure 5 or 6 digit number → likely a pincode, not a phone
        if len(digits_only) in (5, 6) and re.match(r'^\d+$', digits_only):
            return ""

        if len(digits_only) < 7:
            return ""
        return raw

    @staticmethod
    def _clean_date(raw: str) -> str:
        raw = raw.strip()
        raw = re.sub(r"[.,;:]+$", "", raw).strip()
        return raw if raw else ""

    @staticmethod
    def _clean_url(raw: str) -> str:
        raw = re.sub(r"[.,;:!?)\]>]+$", "", raw).strip()
        if not raw.startswith("http"):
            return ""
        return raw

    @staticmethod
    def _clean_currency(raw: str) -> str:
        raw = raw.strip()
        raw = re.sub(r"([$€£¥₹])\s+(\d)", r"\1\2", raw)
        return raw if raw else ""

    @staticmethod
    def _clean_zipcode(raw: str) -> str:
        """
        FIX — NEW: validate and normalise ZIP / pincode.

        Accepted formats:
        ─ Indian pincode: 6-digit number in range 100000–999999
        ─ US ZIP-5:       5-digit number in range 00501–99950
        ─ US ZIP+4:       NNNNN-NNNN format
        """
        raw = raw.strip()
        if not raw:
            return ""

        # US ZIP+4
        if re.match(r'^\d{5}-\d{4}$', raw):
            return raw

        digits = re.sub(r'\D', '', raw)

        # Indian 6-digit pincode (100000–999999)
        if len(digits) == 6:
            val = int(digits)
            if 100000 <= val <= 999999:
                return digits

        # US 5-digit ZIP (00501–99950)
        if len(digits) == 5:
            val = int(digits)
            if 501 <= val <= 99950:
                return digits

        return ""

    # ── SUMMARY ──────────────────────────────────────────────────────────────

    @staticmethod
    def summarise(entities: dict) -> str:
        """Return a human-readable summary of extracted entities."""
        lines = []
        for entity_type, values in entities.items():
            if values:
                joined   = ", ".join(values[:5])
                ellipsis = "…" if len(values) > 5 else ""
                lines.append(f"{entity_type.upper()}: {joined}{ellipsis}")
        return "\n".join(lines) if lines else "No entities detected."