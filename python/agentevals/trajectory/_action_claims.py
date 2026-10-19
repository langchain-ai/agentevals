"""Rule-based extraction of completed-action claims from an assistant reply.

Adapted from toolreceipt (https://github.com/yohannesjoya/toolreceipt), by the same
author. Standard library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence


class ToolSpec(NamedTuple):
    name: str
    description: str = ""


@dataclass
class Claim:
    text: str
    verb: str
    object: Optional[str] = None
    candidates: list[str] = field(default_factory=list)
    # True when the candidates matched on an object word of their tool name, False when
    # on the verb family alone, None when unknown (a claim built by hand).
    object_matched: Optional[bool] = None


def normalize_tool_specs(tools: Iterable[Any]) -> list[ToolSpec]:
    """Normalize tool names and tool dicts (OpenAI function format or with "name" and
    "description" keys) to ``ToolSpec``. Raises ``ValueError`` on anything else."""
    specs: list[ToolSpec] = []
    for tool in tools:
        if isinstance(tool, str):
            specs.append(ToolSpec(tool, ""))
            continue
        src = tool.get("function") if isinstance(tool, Mapping) else None
        if not isinstance(src, Mapping):
            src = tool
        if not isinstance(src, Mapping) or not isinstance(src.get("name"), str):
            raise ValueError(
                f"Invalid tool: {tool!r}. Expected a tool name, or a dict with a 'name' "
                "key or a 'function' dict with a 'name' key."
            )
        specs.append(ToolSpec(src["name"], str(src.get("description") or "")))
    return specs


# Lexicon: verb families as (completed forms, tool-name tokens).
# fmt: off
_FAMILIES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "send": (
        ("sent", "emailed", "e-mailed", "mailed", "notified", "messaged", "texted", "forwarded", "dispatched"),
        ("send", "email", "mail", "notify", "message", "text", "forward", "dispatch"),
    ),
    "create": (
        ("created", "opened", "filed", "raised", "logged", "added", "generated", "set up"),
        ("create", "open", "file", "raise", "log", "make", "add", "new", "generate"),
    ),
    "update": (
        ("updated", "modified", "changed", "edited", "amended", "revised", "patched", "assigned"),
        ("update", "modify", "change", "edit", "amend", "revise", "patch", "set"),
    ),
    "delete": (
        ("deleted", "removed", "cancelled", "canceled", "erased", "dropped", "archived"),
        ("delete", "remove", "cancel", "close", "erase", "drop", "archive", "destroy"),
    ),
    "refund": (
        ("refunded", "reimbursed", "credited", "issued", "processed"),
        ("refund", "reimburse", "credit", "issue", "process", "payment"),
    ),
    "charge": (
        ("charged", "paid", "billed"),
        ("charge", "pay", "bill", "invoice"),
    ),
    "schedule": (
        ("scheduled", "booked", "rescheduled", "arranged", "reserved"),
        ("schedule", "book", "reschedule", "arrange", "reserve", "calendar", "appointment"),
    ),
    "save": (
        ("saved", "stored", "recorded", "wrote", "written", "persisted", "uploaded", "exported"),
        ("save", "store", "record", "write", "persist", "upload", "export"),
    ),
    "submit": (
        ("submitted", "posted", "published"),
        ("submit", "post", "publish", "apply", "register", "enroll", "form"),
    ),
    "trigger": (
        ("triggered", "launched", "started", "ran", "executed", "initiated", "kicked off", "deployed", "invoked"),
        ("trigger", "launch", "start", "run", "execute", "initiate", "deploy", "invoke"),
    ),
}
# Non-effectful verbs: a clause whose first verb is one of these is not a claim.
_EXCLUSIONS = (
    "found", "checked", "looked", "saw", "noticed", "verified", "reviewed", "read", "confirmed",
    "located", "retrieved", "fetched", "searched", "analysed", "analyzed", "understood",
    "received", "logged in", "logged out", "ran into",
)
_NEGATION = (
    "not", "n't", "never", "unable", "failed to", "couldn't", "could not", "cannot", "can't",
    "wasn't able", "was not able", "didn't", "did not", "nothing", "none",
)
_FUTURE = (
    "will", "would", "can ", "could ", "shall", "going to", "once ", "if ", "when you",
    "should i", "let me know", "would you like", "'ll",
)
_ADVICE = ("you can", "please", "you may", "you should", "you'll need")
_OTHER_SUBJECT = ("you sent", "you created", "they sent", "the customer sent")
_CONDITIONAL_LEAD = ("if", "once", "unless", "as soon as", "in case", "should you")
_NON_AGENT_NOUNS = frozenset({
    "customer", "customers", "client", "clients", "user", "users", "buyer", "buyers",
    "member", "members", "subscriber", "subscribers",
})
_FILTER_EXEMPT = ("can confirm", "please note", "please be advised", "happy to confirm")
_TIME_MARKERS = (
    r"\byesterday\b",
    r"\blast (?:night|week|month|year|time|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bago\b", r"\bearlier\b", r"\bthis (?:morning|afternoon|evening)\b",
    r"\bon (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\bon the \d{1,2}(?:st|nd|rd|th)\b",
    r"\bon (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2}\b",
    r"\bat \d{1,2}(?::\d{2})? ?(?:am|pm)\b", r"\bat \d{1,2}:\d{2}\b",
)
_BY_EXEMPT = frozenset({
    "me", "us", "our", "myself", "ourselves", "email", "e-mail", "sms", "text", "phone",
    "mail", "post", "fax", "today", "tomorrow", "tonight", "end", "noon",
})
_SYNONYMS = {
    "email": ("mail", "notify", "message", "reach out", "confirmation"),
    "refund": ("reimburse", "reimbursement", "money back"),
    "order": ("purchase",),
    "row": ("record", "entry"),
    "sms": ("text", "text message"),
}
_NOUN_TOKENS = frozenset({
    "email", "mail", "message", "text", "file", "log", "new", "issue", "payment", "invoice",
    "appointment", "calendar", "record", "form", "post",
})
_READ_VERBS = (
    "get", "read", "fetch", "lookup", "look", "list", "search", "find", "check", "view",
    "query", "show", "retrieve",
)

_PERFECT = frozenset({"have", "has", "had"})
_BE_DONE = frozenset({"been", "was", "were", "got"})
_BE_PRESENT = frozenset({"is", "are", "am", "'s", "be", "being", "get", "gets", "getting"})
_HEADLINE_ADVERBS = frozenset({"already", "just", "now", "successfully", "also"})
_ADVERBS = _HEADLINE_ADVERBS | {"all", "both", "since"}
_AUX = _PERFECT | _BE_DONE | _BE_PRESENT
# A determiner or preposition right before a participle makes it an adjective
# ("your scheduled call", "with saved card"), not a completed action.
_ADJECTIVAL_BEFORE = frozenset({
    "the", "a", "an", "your", "my", "our", "their", "his", "her", "its", "this", "that",
    "these", "those", "any", "no", "some", "every", "each", "with", "of", "for", "from",
})
_FIRST_PERSON = frozenset({"i", "we"})
_OTHER_PRONOUNS = frozenset({"you", "they", "he", "she", "it", "someone", "somebody", "everyone"})
_SUBJECT_DETERMINERS = frozenset({
    "the", "a", "an", "your", "their", "his", "her", "its", "this", "that", "these", "those",
    "my", "some", "someone", "somebody", "everyone",
})
_OBJECT_STRIP = frozenset({"you", "me", "the", "a", "an", "your", "our"})
# Words that never identify a tool (object filler, pronouns, prepositions).
_MATCH_STOP = frozenset({
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "by", "with", "from", "into",
    "and", "or", "it", "its", "this", "that", "these", "those", "you", "your", "me", "my",
    "i", "we", "us", "our", "they", "them", "their", "he", "she", "him", "her", "is", "are",
    "was", "were", "be", "been", "has", "have", "had", "all", "any", "some", "up", "out",
    "now", "again", "just", "already", "also", "as", "so", "here", "there",
})
# fmt: on
_SUBJECT_BOUNDARY = (
    _SUBJECT_DETERMINERS | _OBJECT_STRIP | _FIRST_PERSON | {"that", "which", "who"}
)
_SUBJECT_PREPOSITIONS = frozenset({"to", "for", "of", "from", "with"})

_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_CONJUNCTIONS = r"(?:and|but|so|because|however|then)"
# Sentences are whitespace-collapsed before splitting. Also splits on " - ", en/em
# dashes and ": " (not "e-mail" or "3:00").
_CLAUSE_SPLIT = re.compile(
    rf" ?; ?|,(?= |$) ?| {_CONJUNCTIONS}\b ?| - | ?[–—] ?|:(?= |$) ?",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_LEADING_CONJ = re.compile(rf"^(?:{_CONJUNCTIONS}\b[\s,]*)+", re.IGNORECASE)
_TRAILING_PUNCT = ".!?,;:"
_TOKEN = re.compile(r"'s|[a-z0-9#$@_-]+")
_CONTRACTIONS = (
    ("'ve", " have"),
    ("'re", " are"),
    ("'m", " am"),
    ("'d", " had"),
    ("'ll", " will"),
)
_NAME_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z0-9]+|[A-Z]+[0-9]*")
_NAME_SPLIT = re.compile(r"[^A-Za-z0-9]+")
_WORD = re.compile(r"[a-z0-9]+")

_Term = tuple[str, ...]


def _phrase(entry: str) -> str:
    """Regex for a lexicon phrase, word-bounded on whichever ends are alphanumeric.

    Contraction suffixes such as ``n't`` match inside a word ("haven't").
    """
    e = entry.strip().lower()
    suffix = e.startswith("n'") or not e[:1].isalnum()
    left = "" if suffix else r"(?<![a-z0-9_'])"
    right = r"(?![a-z0-9_])" if e[-1:].isalnum() else ""
    return left + r"\s+".join(re.escape(w) for w in e.split()) + right


def _compile(entries: Iterable[str]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(_phrase(e)) for e in entries)


def _normalize(text: str) -> str:
    """Straighten curly quotes; keeps every character position."""
    return text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')


def _stem(word: str) -> str:
    """Lowercase and strip one plural ``s`` ("tickets" -> "ticket"; keeps "sms")."""
    w = word.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(_normalize(text).lower())]


def _tokenize_tool_name(name: str) -> list[str]:
    return [
        _stem(part)
        for chunk in _NAME_SPLIT.split(name)
        for part in _NAME_PART.findall(chunk)
    ]


def _occurs(term: _Term, words: Sequence[str]) -> set[int]:
    """Indices of ``words`` covered by contiguous occurrences of ``term``."""
    n = len(term)
    hits: set[int] = set()
    for i in range(len(words) - n + 1):
        if tuple(words[i : i + n]) == term:
            hits.update(range(i, i + n))
    return hits


# Precomputed matchers and token sets, built once at import.
_SCAN_ENTRIES: list[tuple[re.Pattern[str], Optional[str]]] = [
    (re.compile(_phrase(form)), family)
    for form, family in sorted(
        [(form, name) for name, (forms, _) in _FAMILIES.items() for form in forms]
        + [(form, None) for form in _EXCLUSIONS],
        key=lambda e: -len(e[0]),
    )
]
_NEGATED = _compile(_NEGATION + _OTHER_SUBJECT)
_EXEMPT = _compile(_FILTER_EXEMPT)
_FUTURE_OR_ADVICE = _compile(_FUTURE + _ADVICE)
_LEADS = _compile(_CONDITIONAL_LEAD)
_TIMED = tuple(re.compile(p) for p in _TIME_MARKERS)
_SYNONYM_GROUPS: list[set[_Term]] = [
    {tuple(_words(w)) for w in (key, *values)} for key, values in _SYNONYMS.items()
]
# Noun-like family tokens ("email", "issue", "record") mapped to their family.
_NOUN_FAMILIES = {
    _stem(tok): name
    for name, (_, tokens) in _FAMILIES.items()
    for tok in tokens
    if tok in _NOUN_TOKENS
}
_READ_STEMS = frozenset(_stem(v) for v in _READ_VERBS)
_FAMILY_TOKEN_STEMS = {
    name: frozenset(_stem(t) for t in tokens) for name, (_, tokens) in _FAMILIES.items()
}
# A family's verb tokens: its name and every token that is not a noun token.
_VERB_TOKENS = {
    name: frozenset({_stem(name)} | {_stem(t) for t in tokens if t not in _NOUN_TOKENS})
    for name, (_, tokens) in _FAMILIES.items()
}


def _family_terms(family: str) -> frozenset[_Term]:
    """Family tokens; verb tokens are also expanded through their synonym groups."""
    verbs = _VERB_TOKENS[family]
    terms: set[_Term] = {(v,) for v in verbs} | {
        tuple(_words(t)) for t in _FAMILIES[family][1]
    }
    for group in _SYNONYM_GROUPS:
        if any(len(term) == 1 and term[0] in verbs for term in group):
            terms |= group
    return frozenset(t for t in terms if t)


_FAMILY_TERMS = {name: _family_terms(name) for name in _FAMILIES}


def _tokens(text: str) -> list[str]:
    low = _normalize(text).lower()
    for short, full in _CONTRACTIONS:
        low = low.replace(short, full)
    return _TOKEN.findall(low.replace("'s", " 's"))


def _split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = _BULLET.sub("", line).strip()
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def _split_clauses(sentence: str) -> list[str]:
    out: list[str] = []
    for piece in _CLAUSE_SPLIT.split(_WHITESPACE.sub(" ", sentence).strip()):
        piece = _LEADING_CONJ.sub("", piece.strip()).strip()
        if piece.strip(_TRAILING_PUNCT + " "):
            out.append(piece)
    return out


class _Action(NamedTuple):
    family: str
    start: int
    end: int
    subject: list[str]  # tokens before the verb, auxiliaries removed
    aux: list[str]  # auxiliary/adverb chain directly before the verb


def _context(clause: str, start: int) -> tuple[list[str], list[str]]:
    before = _tokens(clause[:start])
    aux: list[str] = []
    while before and len(aux) < 4 and (before[-1] in _AUX or before[-1] in _ADVERBS):
        aux.insert(0, before.pop())
    return before, aux


def _completed(subject: list[str], aux: list[str]) -> bool:
    if subject and subject[-1] in _ADJECTIVAL_BEFORE and not set(aux) & _AUX:
        return False  # adjectival: "your scheduled call", "the already sent email"
    present = set(aux) & _BE_PRESENT
    return not present or bool(set(aux) & {"been", "now"})


def _scan(clause: str) -> Optional[_Action]:
    """The first completed action verb in ``clause``, if it comes before any
    non-effectful verb and is not inside a quotation."""
    low = _normalize(clause).lower()
    hits: list[tuple[int, int, Optional[str]]] = []
    for pattern, family in _SCAN_ENTRIES:
        for m in pattern.finditer(low):
            if not any(m.start() < e and s < m.end() for s, e, _ in hits):
                hits.append((m.start(), m.end(), family))
    for start, end, family in sorted(hits):
        if low.count('"', 0, start) % 2:
            continue  # inside a quotation
        if family is None:
            return None  # a non-effectful verb comes first
        subject, aux = _context(clause, start)
        if _completed(subject, aux):
            return _Action(family, start, end, subject, aux)
    return None


def _by_other_agent(clause: str) -> bool:
    tokens = _tokens(clause)
    for i, tok in enumerate(tokens[:-1]):
        if tok != "by":
            continue
        rest = [t for t in tokens[i + 1 :] if t not in {"the", "a", "an"}]
        if rest and rest[0] not in _BY_EXEMPT:
            return True
    return False


def _is_agent_subject(subject: list[str]) -> bool:
    """``I``/``we`` right before the verb chain, or ``our <noun>`` ("our team") unless
    the noun is customer-like ("our customer")."""
    if subject[-1] in _FIRST_PERSON:
        return True
    return "our" in subject[-3:-1] and subject[-1] not in _NON_AGENT_NOUNS


def _subject_kind(action: _Action, clause: str) -> str:
    """Classify who did the action: agent, implicit, other, state or passive."""
    subject, aux = action.subject, set(action.aux)
    passive = bool(aux & (_BE_DONE | _BE_PRESENT))
    if not subject:
        return "implicit"
    if _is_agent_subject(subject):
        return "agent"
    last = subject[-1]
    if not passive:
        # Active voice with any other subject is someone else's action ("John
        # refunded"), except a headline whose participle ends the clause ("Refund
        # issued").
        headline = (
            aux <= _HEADLINE_ADVERBS
            and last not in _OTHER_PRONOUNS
            and not _tokens(clause[action.end :])
        )
        if not headline:
            return "other"
    elif last == "you" and "been" not in aux:
        return "state"  # "you were charged"
    low = _normalize(clause).lower()
    if any(p.search(low) for p in _TIMED) or _by_other_agent(clause):
        return "state"
    return "passive"


def _verbless_other_subject(clause: str) -> Optional[bool]:
    """For a clause with no lexicon verb: does it open with a non-agent subject?"""
    tokens = _tokens(clause)
    if not tokens:
        return None
    if tokens[0] == "our":
        return tokens[1:2] != [] and tokens[1] in _NON_AGENT_NOUNS
    if tokens[0] in _FIRST_PERSON:
        return False
    if tokens[0] in _SUBJECT_DETERMINERS or tokens[0] in _OTHER_PRONOUNS:
        return True
    if any(t in _PERFECT or t in _BE_DONE for t in tokens[1:3]):
        return True  # "Stripe has shipped"
    return None


def _lexically_filtered(clause: str) -> bool:
    """Questions, negations, other subjects, future/conditional and advice."""
    if clause.strip().endswith("?"):
        return True
    low = _normalize(clause).lower()
    if any(p.search(low) for p in _NEGATED):
        return True
    # "I can confirm ..." / "Please note ..." are not future or advice.
    for pattern in _EXEMPT:
        low = pattern.sub(" ", low)
    return any(p.search(low) for p in _FUTURE_OR_ADVICE)


def _extract_object(clause: str, verb_end: int) -> Optional[str]:
    words = clause[verb_end:].strip(_TRAILING_PUNCT + " -").split()
    while words and words[0].lower() in _OBJECT_STRIP:
        words.pop(0)
    return " ".join(words) or None


def family_ran(family: str, tools: Iterable[str]) -> bool:
    """True when any of ``tools`` carries a verb token of ``family`` in its name."""
    verbs = _VERB_TOKENS.get(family)
    if verbs is None:
        return False
    return any(verbs & set(_tokenize_tool_name(t)) for t in tools)


def _conflicts(family: str, name: Sequence[str]) -> bool:
    """True when the tool name says it does something other than the claim: a read
    verb ("lookup_order"), or another family's verb without the claim's own."""
    tokens = set(name)
    if tokens & _READ_STEMS:
        return True
    if tokens & _VERB_TOKENS[family]:
        return False
    return any(
        tokens & verbs for other, verbs in _VERB_TOKENS.items() if other != family
    )


def _object_terms(family: str, words: Sequence[str]) -> set[_Term]:
    """Content words of the object plus synonyms of any group they hit, never another
    family's noun token."""
    foreign = {n for n, fam in _NOUN_FAMILIES.items() if fam != family}
    content = [w for w in words if w not in _MATCH_STOP]
    terms: set[_Term] = {(w,) for w in content}
    for group in _SYNONYM_GROUPS:
        if any(_occurs(member, content) for member in group if member):
            terms |= {m for m in group if m and " ".join(m) not in foreign}
    return terms


def _names_object(verb_stem: Optional[str], name: Sequence[str], family: str) -> bool:
    """True when the claim's verb stem is the object of the tool name: its last token,
    after another token of the family ("refunded" -> the ``refund`` of ``issue_refund``).
    """
    return (
        verb_stem is not None
        and len(name) > 1
        and name[-1] == verb_stem
        and any(t in _FAMILY_TOKEN_STEMS[family] for t in name[:-1])
    )


def _score(
    family: str, object_words: list[str], spec: ToolSpec, verb_stem: Optional[str]
) -> tuple[int, bool]:
    """``(score, object evidence)`` of a claim against one tool.

    +2 for a family token in the tool name, +2 for an object word in the tool name (not
    the same name token), +1 when the description holds an object word the name lacks.
    """
    name = _tokenize_tool_name(spec.name)
    fam_terms = _FAMILY_TERMS[family]
    obj_terms = _object_terms(family, [_stem(w) for w in object_words])
    fam_hits: set[int] = set().union(*(_occurs(t, name) for t in fam_terms))
    obj_hits: set[int] = set().union(*(_occurs(t, name) for t in obj_terms))
    object_matched = bool(obj_hits) and (not fam_hits or len(fam_hits | obj_hits) > 1)
    score = (2 if fam_hits else 0) + (2 if object_matched else 0)
    desc = _words(spec.description)
    if any(_occurs(t, desc) and not _occurs(t, name) for t in obj_terms):
        score += 1
    # Evidence only, never score: a sibling ``<verb>_<stem>`` tool must not outrank the
    # tool that ran ("refunded you" vs ``issue_refund`` and ``refund_payment``).
    return score, object_matched or _names_object(verb_stem, name, family)


def _match_tools(
    family: str,
    obj: Optional[str],
    known_tools: Sequence[ToolSpec],
    verb_stem: Optional[str],
) -> tuple[list[str], bool]:
    """All eligible tools at the maximum score (ties kept) when it is at least 2, and
    whether any of them earned object evidence."""
    words = _words(obj) if obj else []
    scored: list[tuple[str, int, bool]] = []
    for spec in known_tools:
        name = _tokenize_tool_name(spec.name)
        if _VERB_TOKENS[family] & set(name) and not _conflicts(family, name):
            scored.append((spec.name, *_score(family, words, spec, verb_stem)))
    best = max((score for _, score, _ in scored), default=0)
    if best < 2:
        return [], False
    top = [(name, matched) for name, score, matched in scored if score == best]
    return list(dict.fromkeys(name for name, _ in top)), any(m for _, m in top)


def _subject_phrase(subject: Sequence[str]) -> Optional[str]:
    """The noun phrase ending the subject tokens, cut at the first preposition."""
    head = list(subject)
    for i, tok in enumerate(head):
        if tok in _SUBJECT_PREPOSITIONS:
            head = head[:i]
            break
    phrase: list[str] = []
    for tok in reversed(head):
        if tok in _SUBJECT_BOUNDARY or len(phrase) == 4:
            break
        phrase.insert(0, tok)
    return " ".join(phrase) or None


def _verb_stem(form: str, family: str) -> Optional[str]:
    """The family verb token a matched form is built on ("refunded" -> "refund")."""
    compact = re.sub(r"[^a-z]", "", form.lower())
    found = [v for v in _VERB_TOKENS[family] if compact.startswith(v)]
    return max(found, key=len) if found else None


def _surface_stem(form: str, family: str) -> Optional[str]:
    """The noun token a matched form is built on ("emailed" -> "email")."""
    compact = re.sub(r"[^a-z]", "", form.lower())
    found = [
        n
        for n, fam in _NOUN_FAMILIES.items()
        if fam == family and compact.startswith(n) and n != _stem(family)
    ]
    return max(found, key=len) if found else None


def _dedupe(claims: list[Claim]) -> list[Claim]:
    """One claim per candidate set; unmatched claims collapse on ``(verb, object)``.

    The first claim's span is kept, with object evidence when any duplicate has it.
    """
    seen: dict[tuple[Optional[str], ...], Claim] = {}
    out: list[Claim] = []
    for claim in claims:
        key: tuple[Optional[str], ...] = (
            ("tools", *sorted(claim.candidates))
            if claim.candidates
            else ("verb", claim.verb, claim.object)
        )
        first = seen.get(key)
        if first is None:
            seen[key] = claim
            out.append(claim)
        elif claim.object_matched:
            first.object_matched = True
    return out


def _clean(clause: str) -> str:
    return clause.strip().rstrip(_TRAILING_PUNCT).strip()


def _sentence_filtered(sentence: str) -> bool:
    low = _normalize(sentence).lower().lstrip(" \"'(*_")
    return any(p.match(low) for p in _LEADS)


def extract_claims(text: str, known_tools: Sequence[ToolSpec]) -> list[Claim]:
    """Extract completed-action claims from ``text`` and match them to ``known_tools``.

    A claim's ``candidates`` are the tools that could have performed it; it is empty
    when no known tool matches.
    """
    claims: list[Claim] = []
    for sentence in _split_sentences(_normalize(text)):
        if _sentence_filtered(sentence):
            continue
        prev_other = False
        for clause in _split_clauses(sentence):
            action = _scan(clause)
            if action is None:
                other = _verbless_other_subject(clause)
                if other is not None:
                    prev_other = other
                continue
            kind = _subject_kind(action, clause)
            # A subject-less clause carries the previous clause's non-agent subject
            # ("The customer sent the form and has since cancelled the order").
            inherited = kind == "implicit" and prev_other
            if kind != "implicit":
                prev_other = kind == "other"
            if inherited or kind in {"other", "state"} or _lexically_filtered(clause):
                continue
            obj = _extract_object(clause, action.end)
            match_text = obj
            if kind == "passive":
                # "Your refund has been issued": the subject names the object.
                subject = _subject_phrase(action.subject)
                obj = obj or subject
                match_text = " ".join(p for p in (subject, match_text) if p) or None
            # "I emailed you": the verb itself names the channel.
            form = clause[action.start : action.end]
            stem = _surface_stem(form, action.family)
            match_text = " ".join(p for p in (match_text, stem) if p) or None
            candidates, matched = _match_tools(
                action.family,
                match_text,
                known_tools,
                _verb_stem(form, action.family),
            )
            claims.append(
                Claim(
                    text=_clean(clause),
                    verb=action.family,
                    object=obj,
                    candidates=candidates,
                    object_matched=matched,
                )
            )
    return _dedupe(claims)
