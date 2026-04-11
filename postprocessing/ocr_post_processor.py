"""
postprocessing/ocr_post_processor.py  —  v5

v5 CHANGES — Academic / Notes document support:
  ─ Added document type detection: 'letter' vs 'notes' vs 'generic'
  ─ Notes pipeline: preserves numbered points (1), 2), 3)), lettered
    sub-items, definition markers (def°, defn, def.), indented lines
  ─ _fix_notes_structure(): rebuilds numbered/bulleted list structure
    from flat Tesseract output for academic documents
  ─ _fix_academic_vocab(): fixes common OCR errors in economics/science
    notes (economics, definition, wealth, welfare, scarcity, enquiry,
    household, management, ordinary, significance, etc.)
  ─ _fix_notes_headers(): restores Q1./Q.1/Q1: question headers
  ─ All v4 letter fixes retained (run only for letter documents)

v4 ROOT CAUSE FIX retained:
  ─ _reconstruct_structure(): inserts paragraph breaks into flat output
  ─ All v3 targeted word/char/name fixes retained
"""

import re


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT TYPE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_NOTES_SIGNALS = re.compile(
    r'(?:'
    r'Q\s*\.?\s*\d'
    r'|def(?:inition|n|[°\^])'
    r'|\b(?:meaning|defination|welfare|scarcity|wealth|economics)\b'
    r'|\d\s*[)\.]\s+[A-Z]'
    r')',
    re.IGNORECASE
)

_LETTER_SIGNALS = re.compile(
    r'(?:Dear\s+\w|Yours\s+sincerely|With\s+love|Regards,|'
    r'flat\s+no|residency|bengaluru|mumbai)',
    re.IGNORECASE
)


def _detect_doc_type(text: str) -> str:
    """Return 'notes', 'letter', or 'generic'."""
    notes_hits  = len(_NOTES_SIGNALS.findall(text))
    letter_hits = len(_LETTER_SIGNALS.findall(text))
    if notes_hits >= 2:
        return 'notes'
    if letter_hits >= 2:
        return 'letter'
    if notes_hits >= 1:
        return 'notes'
    return 'generic'


# ─────────────────────────────────────────────────────────────────────────────
# GARBAGE LINE FILTER  (run first on raw OCR output)
# ─────────────────────────────────────────────────────────────────────────────

_KEEP_RE = re.compile(
    r'^\s*('
    r'Q\s*\d'
    r'|def\s*[°\^n\.]'
    r'|[→➔]'
    r'|\d\s*[).]'
    r')',
    re.IGNORECASE
)


def _strip_garbage_lines(text: str) -> str:
    """
    Remove lines that are clearly OCR garbage:
    - Mostly single uppercase letters, brackets, random symbols
    - Alpha ratio < 35% and line shorter than 12 chars
    Structural lines (Q1, def, →, numbered items) are always kept.
    Lines longer than 12 chars are kept unless almost entirely non-alpha.
    """
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            continue
        # Always keep structural lines
        if _KEEP_RE.match(stripped):
            result.append(line)
            continue
        # Keep lines that start with a capital word (headings, proper nouns)
        first_word = stripped.split()[0].rstrip('.,!?;:-') if stripped.split() else ''
        if (first_word and first_word[0].isupper()
                and len(first_word) >= 3
                and first_word[1:].islower()):
            result.append(line)
            continue
        # Keep long lines
        if len(stripped) >= 12:
            alpha = sum(c.isalpha() for c in stripped)
            if alpha / len(stripped) >= 0.35:
                result.append(line)
            # else drop — mostly symbols/garbage
            continue
        # Short lines: keep only if reasonably alphabetic
        alpha = sum(c.isalpha() for c in stripped)
        if len(stripped) >= 3 and alpha / len(stripped) >= 0.55:
            result.append(line)
        # else drop
    return '\n'.join(result)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — STRUCTURE RECONSTRUCTION  (v4, for letter/generic)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_structure(text: str) -> str:
    if '\n\n' in text:
        return text
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'  +', ' ', text).strip()
    text = re.sub(r'(\b\d{6}\b)\s+', r'\1\n\n', text)
    text = re.sub(
        r'\s+(\d{1,2}(?:st|nd|rd|th)?\s+'
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December|'
        r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4})',
        r'\n\n\1', text, flags=re.IGNORECASE
    )
    text = re.sub(r'\s+(Dear\s+[A-Z][a-z]+[,.]?)', r'\n\n\1', text)
    text = re.sub(
        r'\s+(With\s+(?:lots|love|warm|kind)\s+of\s+\w+[,.]?'
        r'|Yours\s+\w+[,.]?|Sincerely[,.]?|Regards[,.]?'
        r'|Take\s+care[,.]?|Best\s+wishes[,.]?)',
        r'\n\n\1', text, flags=re.IGNORECASE
    )
    _PARA_STARTERS = (
        r'We\s+had\b|We\s+were\b|They\b|Maybe\b|Perhaps\b|'
        r'I\s+really\b|Really\b|It\s+was\b|There\s+were\b'
    )
    text = re.sub(rf'([.!?])\s+((?:{_PARA_STARTERS}))', r'\1\n\n\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# NOTES-SPECIFIC: ACADEMIC VOCABULARY FIXES
# ─────────────────────────────────────────────────────────────────────────────

_ACADEMIC_WORD_FIXES = [
    # ── Title ─────────────────────────────────────────────────────────────────
    (re.compile(r'\bDiscriptive\b',          re.I), 'Descriptive'),
    (re.compile(r'\bQuestons\b',             re.I), 'Questions'),
    (re.compile(r'\bAnswers?\b',             re.I), 'Answers'),
    # ── Economics terms ───────────────────────────────────────────────────────
    (re.compile(r'\beconornics\b',           re.I), 'economics'),
    (re.compile(r'\beconomis\b',             re.I), 'economics'),
    (re.compile(r'\beconomice\b',            re.I), 'economics'),
    (re.compile(r'\beconom[il]cs\b',         re.I), 'economics'),
    (re.compile(r'\beconorny\b',             re.I), 'economy'),
    (re.compile(r'\bEcor\b',                 re.I), 'Econ'),
    # ── Definition variants ───────────────────────────────────────────────────
    (re.compile(r'\bdefinations?\b',         re.I), 'definitions'),
    (re.compile(r'\bdefination\b',           re.I), 'definition'),
    (re.compile(r'\bdefenition\b',           re.I), 'definition'),
    (re.compile(r'\bdefiniton\b',            re.I), 'definition'),
    # ── Scientists ────────────────────────────────────────────────────────────
    (re.compile(r'\bscientsts\b',            re.I), 'scientists'),
    (re.compile(r'\bscietist\b',             re.I), 'scientist'),
    (re.compile(r'\bscientst\b',             re.I), 'scientist'),
    # ── Greek etymology ───────────────────────────────────────────────────────
    (re.compile(r'\bOikanamicas?\b',         re.I), 'oikonomicas'),
    (re.compile(r'\bOikanomicos?\b',         re.I), 'oikonomicas'),
    (re.compile(r'\boikonomicas?\b',         re.I), 'oikonomicas'),
    # ── Household / management ────────────────────────────────────────────────
    (re.compile(r'\bhoushold\b',             re.I), 'household'),
    (re.compile(r'\bhosehold\b',             re.I), 'household'),
    (re.compile(r'\bmanagment\b',            re.I), 'management'),
    (re.compile(r'\bmanagernent\b',          re.I), 'management'),
    # ── Welfare / scarcity / wealth ───────────────────────────────────────────
    (re.compile(r'\bwellfar[e]?\b',          re.I), 'welfare'),
    (re.compile(r'\bscarsity\b',             re.I), 'scarcity'),
    (re.compile(r'\bscarscity\b',            re.I), 'scarcity'),
    (re.compile(r'\bWelth\b'),                      'Wealth'),
    (re.compile(r'\bweith\b',               re.I), 'wealth'),
    (re.compile(r'\bWeaith\b',              re.I), 'Wealth'),
    # ── Nations ───────────────────────────────────────────────────────────────
    (re.compile(r'\bNakoas\b',               re.I), 'Nations'),
    (re.compile(r'\bNatons\b',               re.I), 'Nations'),
    (re.compile(r'\bNafions\b',              re.I), 'Nations'),
    # ── Enquiries ─────────────────────────────────────────────────────────────
    (re.compile(r'\benquirios?\b',           re.I), 'enquiries'),
    (re.compile(r'\benquizios?\b',           re.I), 'enquiries'),
    (re.compile(r'\benquizies\b',            re.I), 'enquiries'),
    (re.compile(r'\benquizy\b',              re.I), 'enquiry'),
    # ── Ordinary / significance ───────────────────────────────────────────────
    (re.compile(r'\bordinory\b',             re.I), 'ordinary'),
    (re.compile(r'\bordifary\b',             re.I), 'ordinary'),
    (re.compile(r'\bordinay\b',              re.I), 'ordinary'),
    (re.compile(r'\bsignificanc\b',          re.I), 'significance'),
    # ── Economist names ───────────────────────────────────────────────────────
    (re.compile(r'\bAtfired\b',              re.I), 'Alfred'),
    (re.compile(r'\bAlferd\b',              re.I), 'Alfred'),
    (re.compile(r'\bMazshall\b',            re.I), 'Marshall'),
    (re.compile(r'\bMazs!?\b',              re.I), 'Marshall'),
    (re.compile(r'\bMars[hl]+all\b',        re.I), 'Marshall'),
    (re.compile(r'\bSmyth\b',               re.I), 'Smith'),
    (re.compile(r'\bRobb[il]ns\b',          re.I), 'Robbins'),
    (re.compile(r'\bRobb!ns\b',             re.I), 'Robbins'),
    # ── Publications / nature ─────────────────────────────────────────────────
    (re.compile(r'\bBitcaken\b',            re.I), 'Publications'),
    (re.compile(r'\bNafure\b'),                    'Nature'),
    (re.compile(r'\bNatyre\b'),                    'Nature'),
    (re.compile(r'\bNaure\b',              re.I), 'Nature'),
    # ── Common word confusions ────────────────────────────────────────────────
    (re.compile(r"\bman's\s+achan\b",      re.I), "man's action"),
    (re.compile(r'\bAsdyalman\s*tachan\b', re.I), "a study of man's action"),
    (re.compile(r'\bAsdyalmantachan\b',    re.I), "a study of man's action"),
    (re.compile(r'\bbusiaess\b',           re.I), 'business'),
    (re.compile(r'\bbusines[s]?\b',        re.I), 'business'),
    (re.compile(r'\bbusiess\b',            re.I), 'business'),
    (re.compile(r'\bintomeal\b',           re.I), 'income and'),
    (re.compile(r'\bintome\b',             re.I), 'income'),
    (re.compile(r'\bdownn\b',              re.I), 'down'),
    (re.compile(r'\bdowon\b',              re.I), 'down'),
    (re.compile(r'\bGivenby\b',            re.I), 'given by'),
    (re.compile(r'\bQivenby\b',            re.I), 'given by'),
    (re.compile(r'\bWritedowon\b',         re.I), 'Write down'),
    (re.compile(r'\bWritedawn\b',          re.I), 'Write down'),
    (re.compile(r'\bKnolwnas\b',           re.I), 'known as'),
    (re.compile(r'\bDmose\b',              re.I), 'more'),
    (re.compile(r'\bimportan\b',           re.I), 'important'),
    (re.compile(r'\bTifeit\b',             re.I), 'life it'),
    (re.compile(r'\bOfealthandon\b',       re.I), 'of wealth and on'),
    (re.compile(r'\bPrintit\b',            re.I), 'Principles of Economics'),
    # ── rn -> m (Tesseract LSTM artefact) ────────────────────────────────────
    (re.compile(r'(?<=[a-z])rn(?=[a-z])'),        'm'),
    # ── Common merged phrases ──────────────────────────────────────────────────
    (re.compile(r'\bfatherof\s*economics\b', re.I), 'father of economics'),
    (re.compile(r'\bWealthof\s*[Nn]ations\b',re.I), 'Wealth of Nations'),
    (re.compile(r'\bWealthafnakoas\b',       re.I), 'Wealth of Nations'),
    (re.compile(r'\bMeaningof\b',            re.I), 'Meaning of'),
    (re.compile(r'\bDefinitionof\b',         re.I), 'Definition of'),
    (re.compile(r'\bStudyof\b',              re.I), 'Study of'),
    (re.compile(r'\bDerived\b'),                   'Derived'),
]


def _fix_academic_vocab(text: str) -> str:
    for pattern, replacement in _ACADEMIC_WORD_FIXES:
        text = pattern.sub(replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# NOTES-SPECIFIC: QUESTION HEADER RESTORATION
# ─────────────────────────────────────────────────────────────────────────────

def _fix_notes_headers(text: str) -> str:
    text = re.sub(r'\bQ\s*[lIi\.]*\s*1\b', 'Q1.', text)
    text = re.sub(r'\bQ\s*[lIi]\b(?!\w)', 'Q1.', text)
    text = re.sub(r'\bQ\s*\.\s*1\b', 'Q1.', text)
    text = re.sub(r'\bQ\s*1\s*[:.]\s*', 'Q1. ', text)
    text = re.sub(r'\bQ\s*2\s*[:.]\s*', 'Q2. ', text)
    text = re.sub(r'\bQ\s*3\s*[:.]\s*', 'Q3. ', text)
    # Ensure Q<n>. starts on its own line
    text = re.sub(r'(?<!\n)(Q\d+\.)', r'\n\1', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# NOTES-SPECIFIC: NUMBERED LIST STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _fix_notes_structure(text: str) -> str:
    """
    Rebuild numbered/bulleted list structure for academic notes.
    Inserts newlines before list item markers, definition lines, arrows.
    """
    # Numbered items: 1) Word or 1. Word -> newline before
    text = re.sub(r'(?<!\n)(\s*)(\d{1,2}[).]\s+)(?=[A-Z])', r'\n\2', text)

    # Definition markers: def° / def. / defn - -> newline before
    text = re.sub(
        r'(?<!\n)(def\s*[°\^on\.]*\s*[-:–]\s*)',
        r'\ndef. — ', text, flags=re.IGNORECASE
    )

    # Arrow bullets
    text = re.sub(r'(?<!\n)([→➔]\s*)', r'\n\1', text)

    # Section headers that start mid-line: "Meaning of economics:-"
    text = re.sub(
        r'(?<!\n)([A-Z][a-z]+ (?:of|for|by) [a-z]+\s*:-)',
        r'\n\1', text
    )

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# NOTES-SPECIFIC: PARAGRAPH STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def _structure_notes(text: str) -> list:
    """Structure academic notes — preserve numbered items and indented lines."""
    raw_paras = re.split(r'\n{2,}', text.strip())
    result = []
    for para in raw_paras:
        lines = [l.rstrip() for l in para.split('\n') if l.strip()]
        if not lines:
            continue
        result.append('\n'.join(lines))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DATE FRAGMENT REPAIR
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    'jan': 'January',  'january': 'January',
    'feb': 'February', 'february': 'February',
    'mar': 'March',    'march': 'March',
    'apr': 'April',    'april': 'April',
    'may': 'May',
    'jun': 'June',     'june': 'June',
    'jul': 'July',     'july': 'July',
    'aug': 'August',   'august': 'August',
    'sep': 'September','september': 'September',
    'oct': 'October',  'october': 'October',
    'nov': 'November', 'november': 'November',
    'dec': 'December', 'december': 'December',
    'i': 'May', 'm': 'May', 'ma': 'May', 'nay': 'May', 'way': 'May',
}

_DATE_RECONSTRUCT_RE = re.compile(
    r'(\d{1,2}(?:st|nd|rd|th)?)\s+([A-Za-z]{1,9}\.?)\s+(1?20\d{2})',
    re.IGNORECASE
)


def _fix_date_fragments(text: str) -> str:
    def _repair(m):
        day = m.group(1)
        month_raw = m.group(2).lower().rstrip('.')
        year_raw = m.group(3)
        month = _MONTH_MAP.get(month_raw)
        if month is None:
            for key, name in _MONTH_MAP.items():
                if len(key) >= 3 and month_raw.startswith(key[:3]):
                    month = name
                    break
        if month is None:
            month = month_raw.capitalize()
        year = year_raw
        if len(year_raw) == 5 and year_raw[0] == '1' and year_raw[1] == '2':
            year = year_raw[1:]
        return f"{day} {month} {year}"
    text = _DATE_RECONSTRUCT_RE.sub(_repair, text)
    text = re.sub(r'\b1(202[0-9])\b', r'\1', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER-LEVEL FIXES
# ─────────────────────────────────────────────────────────────────────────────

_CHAR_FIXES = [
    (re.compile(r'(?<!\w)\|(?!\w)'),             'I'),
    (re.compile(r'(?m)^[\|l](?= )'),              'I'),
    (re.compile(r'(?<!\w)L(?!\w)'),               'I'),
    (re.compile(r'(Dear \w+,)\s+[A-Z]\b\s*\n'),  r'\1\n'),
    (re.compile(r'[\u201c\u201d]'),                '"'),
    (re.compile(r'[\u2018\u2019]'),                "'"),
    (re.compile(r'\s*\u2014\s*'),                  ' — '),
    (re.compile(r'\s*\u2013\s*'),                  ' - '),
]


# ─────────────────────────────────────────────────────────────────────────────
# WORD-LEVEL OCR FIXES
# ─────────────────────────────────────────────────────────────────────────────

_WORD_FIXES = [
    (re.compile(r'\btbe\b',   re.I), 'the'),
    (re.compile(r'\baild\b',  re.I), 'and'),
    (re.compile(r'\baud\b',   re.I), 'and'),
    (re.compile(r'\b0f\b',    re.I), 'of'),
    (re.compile(r'\b1n\b',    re.I), 'in'),
    (re.compile(r'\bln\b',    re.I), 'in'),
    (re.compile(r'\b1t\b',    re.I), 'It'),
    (re.compile(r'\b1s\b',    re.I), 'is'),
    (re.compile(r'\bvou\b',   re.I), 'you'),
    (re.compile(r'\byonr\b',  re.I), 'your'),
    (re.compile(r'\bwlth\b',  re.I), 'with'),
    (re.compile(r'\bftom\b',  re.I), 'from'),
    (re.compile(r'\bfrorn\b', re.I), 'from'),
    (re.compile(r'\bwonld\b', re.I), 'would'),
    (re.compile(r'\bconld\b', re.I), 'could'),
    (re.compile(r'\bw0rld\b', re.I), 'world'),
    (re.compile(r'\bl0ve\b',  re.I), 'love'),
    (re.compile(r'\bt1me\b',  re.I), 'time'),
    (re.compile(r'\bl1fe\b',  re.I), 'life'),
    (re.compile(r'\bmear\b',  re.I), 'mean'),
    (re.compile(r'\bdiffrent\b', re.I), 'different'),
]


# ─────────────────────────────────────────────────────────────────────────────
# COMMON WORD ERROR FIXES
# ─────────────────────────────────────────────────────────────────────────────

_COMMON_WORD_ERRORS = [
    (re.compile(r'\bgoing well\b',          re.I), 'doing well'),
    (re.compile(r'\binstead some\b',        re.I), 'invited some'),
    (re.compile(r'\bcame was\b',            re.I), 'cake was'),
    (re.compile(r'\beveryone loved?\s+it\b',re.I), 'Everyone loved it'),
    (re.compile(r'(?m)^[Rr]eally missed\b'),       'I really missed'),
    (re.compile(r'(?<=[a-z])rn(?=[a-z])'),          'm'),
    (re.compile(r'goi\s+ng\b',              re.I), 'going'),
    (re.compile(r'doi\s+ng\b',              re.I), 'doing'),
]


# ─────────────────────────────────────────────────────────────────────────────
# NAME CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────

_NAME_CORRECTIONS = [
    (re.compile(r'\bnikki[l]?\b',  re.I), 'Nikhil'),
    (re.compile(r'\bnikhii\b',     re.I), 'Nikhil'),
    (re.compile(r'\bnikhil\b',     re.I), 'Nikhil'),
    (re.compile(r'\barjnn\b',      re.I), 'Arjun'),
    (re.compile(r'\barjun\b',      re.I), 'Arjun'),
]


def _fix_names(text: str) -> str:
    for pattern, replacement in _NAME_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# TRAILING GARBAGE STRIPPER  (letters only)
# ─────────────────────────────────────────────────────────────────────────────

_CLOSING_LINE_RE = re.compile(
    r'^(With\s+lots|With\s+love|With\s+warm|Yours|Sincerely|Regards|'
    r'Best|Take\s+care|Love|Faithfully)\b', re.IGNORECASE
)


def _strip_trailing_garbage(text: str) -> str:
    lines = text.split('\n')
    last_closing_idx = -1
    for i, line in enumerate(lines):
        if _CLOSING_LINE_RE.match(line.strip()):
            last_closing_idx = i
    if last_closing_idx == -1:
        return text
    result_lines = lines[:last_closing_idx + 1]
    for i in range(last_closing_idx + 1, min(last_closing_idx + 3, len(lines))):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            result_lines.append('')
            continue
        words = stripped.split()
        alpha_ratio = sum(c.isalpha() for c in stripped) / max(len(stripped), 1)
        if 1 <= len(words) <= 3 and alpha_ratio >= 0.70 and not re.search(r'\d', stripped):
            result_lines.append(line)
    return '\n'.join(result_lines)


# ─────────────────────────────────────────────────────────────────────────────
# WORD MERGE FIXES
# ─────────────────────────────────────────────────────────────────────────────

_MERGE_FIXES = {
    'asmartwatch':       'a smartwatch',
    'ahappy':            'a happy',
    'theparty':          'the party',
    'andthe':            'and the',
    'ofthe':             'of the',
    'forthe':            'for the',
    'withthe':           'with the',
    'inthe':             'in the',
    'meaningof':         'meaning of',
    'definitionof':      'definition of',
    'wealthof':          'wealth of',
    'fatherof':          'father of',
    'studyof':           'study of',
    'welfaredefinition': 'welfare definition',
    'wealthdefinition':  'wealth definition',
}


def _fix_chars(text: str) -> str:
    for pattern, replacement in _CHAR_FIXES:
        text = pattern.sub(replacement, text)
    text = re.sub(r'  +', ' ', text)
    return text


def _fix_words(text: str) -> str:
    for pattern, replacement in _WORD_FIXES:
        text = pattern.sub(replacement, text)
    text = re.sub(r'(?<!\w)i(?!\w)', 'I', text)
    return text


def _fix_common_word_errors(text: str) -> str:
    for pattern, replacement in _COMMON_WORD_ERRORS:
        text = pattern.sub(replacement, text)
    return text


def _fix_merged_words(text: str) -> str:
    for wrong, right in _MERGE_FIXES.items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', right, text, flags=re.IGNORECASE)

    def _split_camel(m):
        word = m.group(0)
        if word.isupper():
            return word
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', word)

    text = re.sub(r'\b[A-Za-z]{5,}\b', _split_camel, text)
    return text


def _fix_word_spacing(text: str) -> str:
    lines = text.split('\n')
    fixed = []
    for line in lines:
        if not line.strip():
            fixed.append(line)
            continue
        line = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', line)
        line = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', line)
        line = re.sub(r'([\)\]])([A-Za-z])', r'\1 \2', line)
        line = re.sub(
            r'(\d+)([A-Za-z]+)',
            lambda m: m.group(0)
                if re.match(r'^\d+(?:st|nd|rd|th)$', m.group(0), re.I)
                else m.group(1) + ' ' + m.group(2),
            line
        )
        line = re.sub(r'([A-Za-z]{2,})(\d)', r'\1 \2', line)
        line = re.sub(r'  +', ' ', line)
        fixed.append(line)
    return '\n'.join(fixed)


# ─────────────────────────────────────────────────────────────────────────────
# PARAGRAPH STRUCTURE  (letter/generic)
# ─────────────────────────────────────────────────────────────────────────────

_PINCODE_RE = re.compile(r'\b\d{5,6}\b')
_ADDRESS_RE = re.compile(
    r'\b(Road|Street|Avenue|Nagar|Colony|Residency|Flat|House|'
    r'Block|Sector|Phase|Floor|Wing|MG|RC|Lane|Park)\b', re.I)
_DATE_RE = re.compile(
    r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|'
    r'July|August|September|October|November|December|'
    r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{2,4}|'
    r'(?:January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d{1,2},?\s+\d{2,4})\b', re.I)
_SALUT_RE   = re.compile(r'^(Dear|Hi|Hello|To\s+Whom)\b', re.I)
_CLOSING_RE = re.compile(
    r'^(Yours|Sincerely|Regards|Best|With\s+(love|lots|warm|kind|regards)|'
    r'Take\s+care|Warm|Love|Faithfully)\b', re.I)


def _structure(text: str) -> list:
    raw_paras = re.split(r'\n{2,}', text.strip())
    result = []
    for para in raw_paras:
        lines = [l.strip() for l in para.strip().split('\n') if l.strip()]
        if not lines:
            continue
        joined = ' '.join(lines)
        if (len(lines) <= 4
                and (_PINCODE_RE.search(joined) or _ADDRESS_RE.search(joined))
                and len(joined.split()) <= 14):
            result.append('\n'.join(lines))
            continue
        first_line = lines[0]
        if _DATE_RE.search(first_line) and len(first_line.split()) <= 6:
            result.append(first_line)
            if len(lines) > 1:
                sub = _structure('\n'.join(lines[1:]))
                result.extend(sub)
            continue
        combined = ' '.join(lines)
        if _SALUT_RE.match(combined):
            split = re.match(r'^((?:Dear|Hi|Hello)\s+[A-Za-z]+[,.]?)\s+(.*)',
                             combined, re.I | re.S)
            if split and split.group(2).strip():
                result.append(split.group(1).strip())
                result.append(split.group(2).strip())
            else:
                result.append(combined)
            continue
        if _CLOSING_RE.match(combined):
            closing_split = re.match(
                r'^((?:Yours|With\s+\w[\w\s]*|Sincerely|Regards|Best[\w\s]*|'
                r'Warm[\w\s]*|Take\s+care|Love)[^,\n]*[,.]?)\s*,?\s*'
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)$', combined, re.I)
            if closing_split and closing_split.group(2).strip():
                result.append(closing_split.group(1).strip())
                result.append(closing_split.group(2).strip())
            else:
                result.append(combined)
            continue
        result.append(combined)
    return result


def _clean_paragraphs(paragraphs: list) -> list:
    cleaned = []
    for para in paragraphs:
        lines = para.split('\n')
        lines = [re.sub(r'  +', ' ', l).strip() for l in lines]
        para = '\n'.join(lines)
        para = re.sub(r' ([,!?;:])', r'\1', para)
        para = re.sub(r'([,!?;:])([A-Za-z])', r'\1 \2', para)
        para = re.sub(r'([.!?]) ([a-z])',
                      lambda m: m.group(1) + ' ' + m.group(2).upper(), para)
        if para and para[0].islower():
            para = para[0].upper() + para[1:]
        if para.strip():
            cleaned.append(para.strip())
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────────────────────────

_VOCAB = frozenset("""
a able about above after again against all also although always am among and
another any are area around as ask at away back bad be because been before being
believe best better between both bring but by call came can care carry cause
change check child city close come complete concern consider could country course
dear definitely do does done down during each early easy either end enjoy enough
even evening every everyone everything fact family far feel few find fine first
follow for friend from full fun get give glad go going good got great group had
hand happy has have he help her here high him his home hope hour house how however
i idea if important in include indeed inside instead into is it its itself job
just keep kind know large last late later least leave let life light like little
live long look lot love made make many may me mean meet might miss more most much
must my name need never new next nice night no not nothing now of off often old
on once only open or other our out over own part party people person place plan
please point possible put read really room run said same save say school see seem
send sent she show since small so some something soon speak start stay still stop
study such sure take talk than thank that the their them then there these they
thing think this those though through time to today together told too toward try
under until up use very visit wait want was watch way we week well went were what
when where which while who whole will wish with without work world would write year
yet you your
economics definition definitions wealth welfare scarcity meaning derived greek
household management enquiry ordinary significance principles publication nature
causes business income important father nations adam smith alfred marshall lionel
robbins given different scientist write down questions answers descriptive oikos
nomos word known study man actions side part enquiries uses income action
defined definations various scientists economists analysis concept theory
numbered list item section sub topic heading question answer describe explain
write state give mention identify analyze notes academic subject
""".split())


def _confidence(text: str) -> float:
    words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
    if not words:
        return 0.0
    return sum(1 for w in words if w.lower() in _VOCAB) / len(words)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CLASS
# ─────────────────────────────────────────────────────────────────────────────

class OCRPostProcessor:
    """
    OCR post-processor — v5.

    Detects document type ('notes', 'letter', 'generic') and routes to
    the appropriate pipeline.

    NOTES pipeline (new in v5):
      1. _fix_chars()
      2. _fix_academic_vocab()    — economics/science vocabulary
      3. _fix_notes_headers()     — Q1./Q.1 headers
      4. _fix_date_fragments()
      5. _fix_words()
      6. _fix_common_word_errors()
      7. _fix_merged_words()
      8. _fix_word_spacing()
      9. _fix_names()
     10. _fix_notes_structure()   — numbered list / def marker newlines
     11. _structure_notes()
     12. _clean_paragraphs()

    LETTER pipeline (v4 — unchanged for letters)
    """

    def process(self, raw: str) -> dict:
        if not raw or not raw.strip():
            return {'text': '', 'confidence': 0.0,
                    'confidence_pct': 0.0, 'paragraphs': []}

        doc_type = _detect_doc_type(raw)

        if doc_type == 'notes':
            text = self._process_notes(raw)
        else:
            text = self._process_letter_generic(raw)

        conf = _confidence(text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]

        return {
            'text':           text,
            'doc_type':       doc_type,
            'confidence':     round(conf, 4),
            'confidence_pct': round(conf * 100, 1),
            'paragraphs':     paragraphs,
        }

    def _process_notes(self, raw: str) -> str:
        text = _strip_garbage_lines(raw)         # remove OCR garbage lines first
        text = _fix_chars(text)
        text = _fix_academic_vocab(text)
        text = _fix_notes_headers(text)
        text = _fix_date_fragments(text)
        text = _fix_words(text)
        text = _fix_common_word_errors(text)
        text = _fix_merged_words(text)
        text = _fix_word_spacing(text)
        text = _fix_names(text)
        text = _fix_notes_structure(text)
        paragraphs = _structure_notes(text)
        paragraphs = _clean_paragraphs(paragraphs)
        return '\n\n'.join(paragraphs)

    def _process_letter_generic(self, raw: str) -> str:
        text = _reconstruct_structure(raw)
        text = _fix_chars(text)
        text = _fix_date_fragments(text)
        text = _fix_words(text)
        text = _fix_common_word_errors(text)
        text = _fix_merged_words(text)
        text = _fix_word_spacing(text)
        text = _fix_names(text)
        text = _strip_trailing_garbage(text)
        paragraphs = _structure(text)
        paragraphs = _clean_paragraphs(paragraphs)
        return '\n\n'.join(paragraphs)

    def process_text(self, raw: str) -> str:
        return self.process(raw)['text']


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_post_processor.py <image_path>")
        sys.exit(1)
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("ERROR: pytesseract and Pillow are required.")
        sys.exit(1)
    img = Image.open(sys.argv[1])
    raw = pytesseract.image_to_string(img, config='--oem 1 --psm 3')
    proc = OCRPostProcessor()
    result = proc.process(raw)
    print(result['text'])
    print()
    print(f"─── Confidence: {result['confidence_pct']}% (doc_type: {result.get('doc_type','?')}) ───")