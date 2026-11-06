"""
numcite.py: turn the author-year citations written in the templates into Frontiers numbered citations.

The templates keep author-year citations because they are easy to read and check while writing. At build time every
citation is replaced by the number of its reference, references are numbered in order of first citation, and the
reference list is printed in that order. The manuscript and the supplement each number their own list.

Forms recognized (the only forms the templates use):
  parenthetical  (Costa et al., 2005; Wu et al., 2014)   -> (7, 8)
                 (Książek et al., 2023, 2024)            -> (2, 3)
                 (Marwan et al., 2002, 2007)             -> (17, 18)
                 (Task Force, 1996)                      -> (12)
  narrative      Bigger et al. (1996)                    -> Bigger et al. (9)
                 the Task Force (1996)                   -> the Task Force (12)

Numbers inside one pair of parentheses are sorted, and runs of three or more consecutive numbers are collapsed with
an en dash, as in the Frontiers medical citation style: (1, 2), (3-5), (1, 4, 6-8).

A parenthesis that mixes citations with other text, such as "(tie-corrected; Dunn, 1964)", is refused rather than
guessed at: the template must be written so that each citation stands in its own parentheses. A citation-shaped
"Name, Year" or "Name et al., Year" with no matching reference is also refused.
"""
import re

YEAR = r"(?:19|20)\d{2}"
NAME = r"[A-Za-zÀ-žąćęłńóśźżĄĆĘŁŃÓŚŹŻ][\w\-'ÀžąćęłńóśźżĄĆĘŁŃÓŚŹŻ]*"
# one citation inside parentheses: surname (one or two words, e.g. 'de Leon', 'Task Force'), optional
# 'et al.' or 'and Second', then one or more years
PART = re.compile(rf"^(?P<sur>{NAME}(?: {NAME})?)(?P<rest> et al\.| and {NAME}(?: {NAME})?)?, (?P<years>{YEAR}(?:, {YEAR})*)$")
LOOKS_LIKE = re.compile(rf"^[A-Za-zÀ-ž][^;]*?, {YEAR}(?:, {YEAR})*$")


def collapse(nums):
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:
            out.append(f"{nums[i]}–{nums[j]}")
            out += []
        else:
            out += [str(n) for n in nums[i:j + 1]]
        i = j + 1
    return ", ".join(out)


class Citer:
    def __init__(self, keys):
        self.keys = list(keys)
        self.lookup_tbl = {}
        for k in self.keys:
            sur, y = k.rsplit(" ", 1)
            self.lookup_tbl[(sur, y)] = k
        self.surnames = sorted({k.rsplit(" ", 1)[0] for k in self.keys}, key=len, reverse=True)
        self.order = []          # keys in order of first citation
        self.n_citations = 0     # citation instances converted
        self.n_groups = 0        # parenthesized number groups written (the checkers count them in the built file)

    def number(self, key):
        if key not in self.order:
            self.order.append(key)
        return self.order.index(key) + 1

    def _key(self, sur, year, where):
        k = self.lookup_tbl.get((sur, year))
        if k is None:
            raise SystemExit(f"citation '{sur} {year}' has no reference ({where[:120]})")
        return k

    def _paren(self, m):
        inner = m.group("inner")
        parts = [p.strip() for p in inner.split(";")]
        hits = [PART.match(p) for p in parts]
        cite_like = [bool(h) or bool(LOOKS_LIKE.match(p)) for h, p in zip(hits, parts)]
        if not any(cite_like):
            return m.group(0)
        if not all(cite_like):
            raise SystemExit(f"parenthesis mixes citations with other text; rewrite it: {m.group(0)[:200]}")
        nums = []
        for h, p in zip(hits, parts):
            if not h:
                raise SystemExit(f"citation form not recognized: '{p}' in {m.group(0)[:200]}")
            sur = h.group("sur")
            for y in re.findall(YEAR, h.group("years")):
                nums.append(self.number(self._key(sur, y, m.group(0))))
                self.n_citations += 1
        self.n_groups += 1
        return "(" + collapse(nums) + ")"

    def _narr(self, m):
        sur = m.group("sur")
        nums = []
        for y in re.findall(YEAR, m.group("years")):
            nums.append(self.number(self._key(sur, y, m.group(0))))
            self.n_citations += 1
        self.n_groups += 1
        return f"{sur}{m.group('rest') or ''} ({collapse(nums)})"

    def convert(self, text):
        """One left-to-right pass, so numbers are assigned in reading order whichever form a citation takes."""
        if not text:
            return text
        sur_alt = "|".join(re.escape(s) for s in self.surnames)
        both = re.compile(rf"(?<![\w\-])(?P<sur>{sur_alt})(?P<rest> et al\.| and {NAME})? \((?P<years>{YEAR}(?:, {YEAR})*)\)"
                          r"|\((?P<inner>[^()]*)\)")
        return both.sub(lambda m: self._narr(m) if m.group("sur") else self._paren(m), text)

    def convert_any(self, obj):
        """Convert strings inside nested lists and dicts (table rows), in reading order."""
        if isinstance(obj, str):
            return self.convert(obj)
        if isinstance(obj, list):
            return [self.convert_any(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self.convert_any(v) for k, v in obj.items()}
        return obj


def leftover_author_year(text):
    """Author-year citations that survived conversion (should be none)."""
    pat = re.compile(rf"\b[A-Z][\w\-ąćęłńóśźż]+(?: et al\.| and [A-Z][\w\-]+)?(?:, | \(){YEAR}\b")
    return [m.group(0) for m in pat.finditer(text)
            if not m.group(0).split()[0] in ("Table", "Figure", "Version", "Python", "October", "August", "July",
                                             "Section", "Methods", "Results", "Supplementary")]


if __name__ == "__main__":   # self-test
    c = Citer(["Costa 2005", "Wu 2014", "Książek 2023", "Książek 2024", "Task Force 1996", "de Leon 2005",
               "Bigger 1996", "Marwan 2002", "Marwan 2007", "Sacha 2014", "Monfredi 2014"])
    s = ("A (Costa et al., 2005; Wu et al., 2014) B (Książek et al., 2023, 2024) C (Task Force, 1996) "
         "D Bigger et al. (1996) E the Task Force (1996) F (de Leon and Diaz, 2005) G (Marwan et al., 2002, 2007) "
         "H (Monfredi et al., 2014; Sacha, 2014) I (Costa et al., 2005; Wu et al., 2014; Książek et al., 2023) "
         "J (not a citation, 2,000 resamples) K (n = 30)")
    out = c.convert(s)
    print(out)
    assert out == ("A (1, 2) B (3, 4) C (5) D Bigger et al. (6) E the Task Force (5) F (7) G (8, 9) H (10, 11) "
                   "I (1–3) J (not a citation, 2,000 resamples) K (n = 30)"), out
    assert collapse([1, 2, 3, 5, 7, 8, 9, 10]) == "1–3, 5, 7–10"
    for bad in ("(tie-corrected; Dunn, 1964)", "(Nobody et al., 2001)"):
        try:
            Citer(["Dunn 1964"]).convert(bad)
        except SystemExit as e:
            print("refused as expected:", e)
        else:
            raise AssertionError(bad)
    print("self-test passed")
