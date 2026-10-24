"""Task families: verifiable programming tasks with a reference solution and known-buggy variants.

A family is a parametrised generator. ``ref_src``/``bug_srcs`` are real Python source (``FN`` is replaced
by the function name) that the sandbox really executes. A bug only shows on some inputs, so a flawed
solution can pass a single instance by luck. That is the failure mode the skill gate has to catch.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Callable

# --------------------------------------------------------------------------- reference behaviours


def _to_roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
            (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)


def _from_roman(s: str) -> int:
    m = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = m[ch]
        total += -v if i + 1 < len(s) and m[s[i + 1]] > v else v
    return total


def _duration(s: str) -> int:
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(int(n) * mult[u] for n, u in re.findall(r"(\d+)([dhms])", s))


def _balanced(s: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    st = []
    for c in s:
        if c in "([{":
            st.append(c)
        elif not st or st.pop() != pairs[c]:
            return False
    return not st


def _anagram_groups(words: list[str]) -> int:
    g: dict[str, int] = Counter("".join(sorted(w.lower())) for w in words)
    return sum(1 for v in g.values() if v >= 2)


def _days_between(a: str, b: str) -> int:
    return abs((dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days)


def _to_base(n: int, base: int) -> str:
    digits = "0123456789ABCDEF"
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return "".join(reversed(out))


def _csv_sum(csv: str, column: str) -> int:
    lines = [ln for ln in csv.split("\n") if ln.strip()]
    head = lines[0].split(",")
    idx = head.index(column)
    return sum(int(ln.split(",")[idx]) for ln in lines[1:])


def _top_word(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    c = Counter(words)
    best = max(c.values())
    return sorted(w for w, v in c.items() if v == best)[0]


def _rle(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(f"{s[i]}{j - i}")
        i = j
    return "".join(out)


def _flatten_sum(data) -> int:
    return sum(_flatten_sum(x) for x in data) if isinstance(data, list) else data


def _date_add(start: str, days: int) -> str:
    return (dt.date.fromisoformat(start) + dt.timedelta(days=days)).isoformat()


# --------------------------------------------------------------------------- generators


def _g_roman(r: Random):
    return {"n": r.randint(1, 3999)}


def _g_duration(r: Random):
    parts = []
    if r.random() < 0.4:
        parts.append(f"{r.randint(1, 3)}d")
    for u, hi in (("h", 23), ("m", 59), ("s", 59)):
        if r.random() < 0.7:
            parts.append(f"{r.randint(1, hi)}{u}")
    return {"s": " ".join(parts) or "42s"}


def _g_balanced(r: Random):
    def make(depth: int) -> str:
        if depth == 0 or r.random() < 0.3:
            return ""
        o, c = r.choice(["()", "[]", "{}"])
        return o + make(depth - 1) + c + make(depth - 1)

    s = make(4) or "()"
    roll = r.random()
    if roll < 0.25 and len(s) > 3:           # same multiset of brackets, wrong order
        i = r.randrange(len(s) - 1)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
    elif roll < 0.5:                         # wrong type of closer
        i = r.randrange(len(s))
        s = s[:i] + r.choice(")]}") + s[i + 1:]
    return {"s": s}


_VOCAB = ["listen", "silent", "enlist", "google", "gogole", "cat", "act", "tac", "dog", "god", "bird", "stone",
          "notes", "onset", "tones", "angel", "glean", "angle", "rat", "tar", "art", "moon", "mono"]


def _g_anagram(r: Random):
    words = r.sample(_VOCAB, r.randint(6, 9))
    words = [w.capitalize() if r.random() < 0.12 else w for w in words]
    return {"words": words}


def _g_date_diff(r: Random):
    base = dt.date(2020, 1, 1)
    a = base + dt.timedelta(days=r.randint(0, 2000))
    b = base + dt.timedelta(days=r.randint(0, 2000))
    return {"a": a.isoformat(), "b": b.isoformat()}


def _g_base(r: Random):
    return {"n": r.randint(10, 5000), "base": r.choice([2, 8, 16, 16, 12, 14])}


def _g_csv(r: Random):
    cols = r.sample(["qty", "price", "tax", "stock", "score"], 3)
    rows = [",".join(str(r.randint(1, 99)) for _ in cols) for _ in range(r.randint(3, 7))]
    text = ",".join(cols) + "\n" + "\n".join(rows)
    if r.random() < 0.5:
        text += "\n"
    return {"csv": text, "column": r.choice(cols)}


_SENT = ["The cat saw the dog.", "A dog, a cat and the bird!", "Is the Cat there?", "Birds sing; the sun is up.",
         "Dog days are here.", "The end of the road, the end."]


def _g_topword(r: Random):
    return {"text": " ".join(r.choices(_SENT, k=r.randint(2, 4)))}


def _g_rle(r: Random):
    out = []
    for ch in r.choices(string.ascii_lowercase[:6], k=r.randint(3, 6)):
        out.append(ch * (r.randint(10, 13) if r.random() < 0.18 else r.randint(1, 4)))
    return {"s": "".join(out)}


def _g_flatten(r: Random):
    def make(depth: int):
        if depth == 0:
            return r.randint(-5, 20)
        return [make(depth - 1) if r.random() < 0.5 else r.randint(-5, 20) for _ in range(r.randint(2, 3))]

    return {"data": make(r.choice([1, 2, 2, 3, 4]))}


def _g_roman_to_int(r: Random):
    return {"s": _to_roman(r.randint(1, 3999))}


def _g_date_add(r: Random):
    return {"start": (dt.date(2021, 1, 1) + dt.timedelta(days=r.randint(0, 1500))).isoformat(),
            "days": r.randint(1, 400)}


def _g_dur_total(r: Random):
    return {"items": [_g_duration(r)["s"] for _ in range(r.randint(2, 4))]}


def _g_roman_add(r: Random):
    a, b = r.randint(1, 1900), r.randint(1, 1900)
    return {"a": _to_roman(a), "b": _to_roman(b)}


def _g_balanced_count(r: Random):
    return {"items": [_g_balanced(r)["s"] for _ in range(r.randint(4, 7))]}


def _g_span_total(r: Random):
    return {"pairs": [list(_g_date_diff(r).values()) for _ in range(r.randint(2, 4))]}


# --------------------------------------------------------------------------- family table


@dataclass(frozen=True)
class Family:
    name: str
    fn_name: str
    keys: tuple[str, ...]
    stems: tuple[str, ...]
    gen: Callable[[Random], dict]
    truth: Callable[..., object]
    ref_src: str
    bug_srcs: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    glue_src: str = ""                       # for composites: body defining ``solve``
    split: str = "train"                     # "train" | "transfer"
    tags: tuple[str, ...] = field(default_factory=tuple)


ROMAN_REF = '''
def FN(n):
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)
'''
ROMAN_BUG = '''
def FN(n):
    vals = [(1000, "M"), (500, "D"), (100, "C"), (50, "L"), (10, "X"), (5, "V"), (1, "I")]
    out = []
    for v, s in vals:
        while n >= v:
            out.append(s)
            n -= v
    return "".join(out)
'''
FROM_ROMAN_REF = '''
def FN(s):
    m = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(s):
        v = m[ch]
        if i + 1 < len(s) and m[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total
'''
FROM_ROMAN_BUG = '''
def FN(s):
    m = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    return sum(m[ch] for ch in s)
'''
DURATION_REF = '''
import re
def FN(s):
    mult = {"d": 86400, "h": 3600, "m": 60, "s": 1}
    return sum(int(n) * mult[u] for n, u in re.findall(r"(\\d+)([dhms])", s))
'''
DURATION_BUG = '''
import re
def FN(s):
    mult = {"h": 3600, "m": 60, "s": 1}
    return sum(int(n) * mult[u] for n, u in re.findall(r"(\\d+)([hms])", s))
'''
BALANCED_REF = '''
def FN(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    st = []
    for c in s:
        if c in "([{":
            st.append(c)
        elif not st or st.pop() != pairs[c]:
            return False
    return not st
'''
BALANCED_BUG = '''
def FN(s):
    return s.count("(") == s.count(")") and s.count("[") == s.count("]") and s.count("{") == s.count("}")
'''
ANAGRAM_REF = '''
def FN(words):
    groups = {}
    for w in words:
        k = "".join(sorted(w.lower()))
        groups[k] = groups.get(k, 0) + 1
    return sum(1 for v in groups.values() if v >= 2)
'''
ANAGRAM_BUG = '''
def FN(words):
    groups = {}
    for w in words:
        k = "".join(sorted(w))
        groups[k] = groups.get(k, 0) + 1
    return sum(1 for v in groups.values() if v >= 2)
'''
DATEDIFF_REF = '''
from datetime import date
def FN(a, b):
    return abs((date.fromisoformat(b) - date.fromisoformat(a)).days)
'''
DATEDIFF_BUG = '''
from datetime import date
def FN(a, b):
    return (date.fromisoformat(b) - date.fromisoformat(a)).days
'''
BASE_REF = '''
def FN(n, base):
    digits = "0123456789ABCDEF"
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return "".join(reversed(out))
'''
BASE_BUG = '''
def FN(n, base):
    digits = "0123456789abcdef"
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, base)
        out.append(digits[r])
    return "".join(reversed(out))
'''
CSV_REF = '''
def FN(csv, column):
    lines = [ln for ln in csv.split("\\n") if ln.strip()]
    idx = lines[0].split(",").index(column)
    return sum(int(ln.split(",")[idx]) for ln in lines[1:])
'''
CSV_BUG = '''
def FN(csv, column):
    lines = csv.split("\\n")
    idx = lines[0].split(",").index(column)
    return sum(int(ln.split(",")[idx]) for ln in lines[1:-1])
'''
TOPWORD_REF = '''
import re
from collections import Counter
def FN(text):
    c = Counter(re.findall(r"[a-z']+", text.lower()))
    best = max(c.values())
    return sorted(w for w, v in c.items() if v == best)[0]
'''
TOPWORD_BUG = '''
from collections import Counter
def FN(text):
    c = Counter(text.split())
    best = max(c.values())
    return sorted(w for w, v in c.items() if v == best)[0]
'''
RLE_REF = '''
def FN(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str(j - i))
        i = j
    return "".join(out)
'''
RLE_BUG = '''
def FN(s):
    out, i = [], 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i]:
            j += 1
        out.append(s[i] + str((j - i) % 10))
        i = j
    return "".join(out)
'''
FLATTEN_REF = '''
def FN(data):
    if isinstance(data, list):
        return sum(FN(x) for x in data)
    return data
'''
FLATTEN_BUG = '''
def FN(data):
    return sum(sum(x) if isinstance(x, list) else x for x in data)
'''
DATEADD_REF = '''
from datetime import date, timedelta
def FN(start, days):
    return (date.fromisoformat(start) + timedelta(days=days)).isoformat()
'''

FAMILIES: dict[str, Family] = {}


def _reg(f: Family) -> Family:
    FAMILIES[f.name] = f
    return f


_reg(Family("roman", "int_to_roman", ("n",),
            ("Convert the integer in INPUT to a Roman numeral.",
             "Write the Roman numeral representation of the positive integer n given below.",
             "Return the Roman numeral for the number n."),
            _g_roman, _to_roman, ROMAN_REF, (ROMAN_BUG,)))
_reg(Family("roman_to_int", "roman_to_int", ("s",),
            ("Convert the Roman numeral string s in INPUT to an integer.",
             "Parse the Roman numeral s given below and return its integer value.",
             "What integer does the Roman numeral s represent?"),
            _g_roman_to_int, _from_roman, FROM_ROMAN_REF, (FROM_ROMAN_BUG,)))
_reg(Family("duration", "parse_duration", ("s",),
            ("Parse the duration string s in INPUT (units d, h, m, s) and return the total number of seconds.",
             "Convert a duration such as '1h 30m 15s' (string s below) into seconds.",
             "How many seconds are in the duration s given below? Units are days, hours, minutes, seconds."),
            _g_duration, _duration, DURATION_REF, (DURATION_BUG,)))
_reg(Family("balanced", "is_balanced", ("s",),
            ("Check whether the brackets in string s from INPUT are balanced and properly nested. Print True or False.",
             "Is the bracket string s below valid (every opener closed by the matching type, in order)? Answer True or False.",
             "Decide if the parentheses, square and curly brackets in s are balanced. Answer True or False."),
            _g_balanced, _balanced, BALANCED_REF, (BALANCED_BUG,)))
_reg(Family("anagram", "count_anagram_groups", ("words",),
            ("Count the groups of anagrams (groups with at least two words, ignoring case) in the word list in INPUT.",
             "Given the list of words below, how many distinct anagram groups of size two or more exist? Ignore letter case.",
             "Return the number of anagram groups with two or more members among the words."),
            _g_anagram, _anagram_groups, ANAGRAM_REF, (ANAGRAM_BUG,)))
_reg(Family("date_diff", "days_between", ("a", "b"),
            ("Return the absolute number of days between the ISO dates a and b in INPUT.",
             "How many days apart are dates a and b below (ISO format, always non-negative)?",
             "Compute the number of days separating the two ISO dates a and b."),
            _g_date_diff, _days_between, DATEDIFF_REF, (DATEDIFF_BUG,)))
_reg(Family("date_add", "add_days", ("start", "days"),
            ("Add the given number of days to the ISO date in INPUT and return the new ISO date.",
             "What ISO date is `days` days after `start` below?",
             "Shift the start date by days days and print the result as an ISO date."),
            _g_date_add, _date_add, DATEADD_REF))
_reg(Family("base_conv", "to_base", ("n", "base"),
            ("Convert the integer n in INPUT to the given base (2 to 16). Use uppercase letters for digits above 9.",
             "Write n below in base `base`, using digits 0-9 and uppercase A-F.",
             "Return the representation of n in the numeral system with the given base, uppercase."),
            _g_base, _to_base, BASE_REF, (BASE_BUG,)))
_reg(Family("csv_sum", "sum_column", ("csv", "column"),
            ("Sum the integer values in the named column of the CSV text in INPUT (first line is the header).",
             "The CSV below has a header row. Return the sum of the column called `column`.",
             "Add up all the numbers in column `column` of the CSV data."),
            _g_csv, _csv_sum, CSV_REF, (CSV_BUG,)))
_reg(Family("top_word", "most_common_word", ("text",),
            ("Find the most frequent word in the text in INPUT, ignoring case and punctuation. Break ties alphabetically.",
             "Which word occurs most often in the text below? Lowercase, no punctuation, ties alphabetical.",
             "Return the most common word of the text (case-insensitive, punctuation stripped, alphabetical tie-break)."),
            _g_topword, _top_word, TOPWORD_REF, (TOPWORD_BUG,)))
_reg(Family("rle", "run_length_encode", ("s",),
            ("Run-length encode the string s in INPUT, e.g. 'aaab' becomes 'a3b1'.",
             "Compress the string s below as character followed by run count ('aaab' -> 'a3b1').",
             "Return the run-length encoding of s (each character followed by the length of its run)."),
            _g_rle, _rle, RLE_REF, (RLE_BUG,)))
_reg(Family("flatten_sum", "nested_sum", ("data",),
            ("Return the sum of all integers in the arbitrarily nested list data in INPUT.",
             "Add every number in the nested list below, however deeply nested.",
             "Sum all the integers found in the nested list structure."),
            _g_flatten, _flatten_sum, FLATTEN_REF, (FLATTEN_BUG,)))

# transfer (never in the learning stream): composites of the above
_reg(Family("duration_total", "total_duration", ("items",),
            ("Sum the durations in the list items from INPUT (each like '1h 30m 15s', units d/h/m/s) and return total seconds.",
             "Given a list of duration strings, what is the total number of seconds?"),
            _g_dur_total, lambda items: sum(_duration(x) for x in items), "", (), ("duration",),
            "def solve(items):\n    return sum(parse_duration(x) for x in items)\n", "transfer"))
_reg(Family("roman_add", "roman_add", ("a", "b"),
            ("Add the two Roman numerals a and b in INPUT and return the sum as a Roman numeral.",
             "Take the Roman numerals a and b below, add their values, and print the result in Roman numerals."),
            _g_roman_add, lambda a, b: _to_roman(_from_roman(a) + _from_roman(b)), "", (),
            ("roman", "roman_to_int"),
            "def solve(a, b):\n    return int_to_roman(roman_to_int(a) + roman_to_int(b))\n", "transfer"))
_reg(Family("balanced_count", "count_balanced", ("items",),
            ("How many of the bracket strings in the list items from INPUT are balanced and properly nested?",
             "Count the valid (balanced, properly nested) bracket strings in the list below."),
            _g_balanced_count, lambda items: sum(_balanced(x) for x in items), "", (), ("balanced",),
            "def solve(items):\n    return sum(1 for x in items if is_balanced(x))\n", "transfer"))
_reg(Family("date_span_total", "total_span_days", ("pairs",),
            ("Each element of pairs in INPUT is two ISO dates; return the sum of the absolute day differences over all pairs.",
             "Given the date pairs below, add up the number of days between the two dates of each pair (non-negative)."),
            _g_span_total, lambda pairs: sum(_days_between(a, b) for a, b in pairs), "", (),
            ("date_diff",),
            "def solve(pairs):\n    return sum(days_between(a, b) for a, b in pairs)\n", "transfer"))

TRAIN_FAMILIES = [f.name for f in FAMILIES.values() if f.split == "train"]
TRANSFER_FAMILIES = [f.name for f in FAMILIES.values() if f.split == "transfer"]


def make_instance(fam: Family, rng: Random) -> tuple[dict, str, str]:
    """Return ``(inputs, prompt, expected)``."""
    inputs = fam.gen(rng)
    expected = str(fam.truth(**inputs))
    stem = rng.choice(fam.stems)
    prompt = (f"{stem}\nYour code must define `solve({', '.join(fam.keys)})` and print the answer. "
              f"The variable INPUT (a dict) is predefined.\nINPUT = {json.dumps(inputs)}")
    return inputs, prompt, expected
