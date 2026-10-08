"""Pure-Python, deterministic metamorphic transformation operators.

Every operator returns a :class:`TransformResult` ``(text, applied)``: the
transformed text and a flag stating whether the transformation actually changed
the input. When ``applied`` is False, ``text`` equals the (unchanged) input.

Operators are deterministic: identical ``(text, variant)`` inputs always give
identical outputs. ``variant`` deterministically selects among alternatives.
"""

from __future__ import annotations

import re
from typing import NamedTuple


class TransformResult(NamedTuple):
    """Outcome of a metamorphic operator."""

    text: str
    applied: bool


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _match_case(original: str, replacement: str) -> str:
    """Transfer the capitalisation pattern of ``original`` onto ``replacement``."""
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _result(original: str, transformed: str) -> TransformResult:
    return TransformResult(transformed, transformed != original)


# --------------------------------------------------------------------------- #
# MR-1: lexical equivalence
# --------------------------------------------------------------------------- #

LEXICAL_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    # positive adjectives
    "good": ("fine", "decent"),
    "great": ("excellent", "superb"),
    "wonderful": ("marvelous", "delightful"),
    "fantastic": ("terrific", "fabulous"),
    "amazing": ("astonishing", "remarkable"),
    "beautiful": ("gorgeous", "stunning"),
    "enjoyable": ("pleasurable", "entertaining"),
    "impressive": ("striking", "admirable"),
    "helpful": ("useful", "supportive"),
    "friendly": ("amiable", "welcoming"),
    "nice": ("pleasant", "lovely"),
    "happy": ("glad", "cheerful"),
    "clean": ("spotless", "tidy"),
    "pleasant": ("agreeable", "enjoyable"),
    "lovely": ("charming", "delightful"),
    "exciting": ("thrilling", "gripping"),
    "interesting": ("engaging", "fascinating"),
    "satisfying": ("gratifying", "rewarding"),
    "polite": ("courteous", "civil"),
    # negative adjectives
    "bad": ("poor", "lousy"),
    "terrible": ("awful", "dreadful"),
    "awful": ("terrible", "dreadful"),
    "horrible": ("dreadful", "atrocious"),
    "boring": ("tedious", "dull"),
    "disappointing": ("underwhelming", "unsatisfying"),
    "poor": ("inferior", "substandard"),
    "ugly": ("unsightly", "hideous"),
    "slow": ("sluggish", "plodding"),
    "dull": ("tedious", "lifeless"),
    "annoying": ("irritating", "vexing"),
    "rude": ("impolite", "discourteous"),
    "mediocre": ("lackluster", "unremarkable"),
    "unpleasant": ("disagreeable", "objectionable"),
    "dirty": ("filthy", "grubby"),
    # intensifiers (polarity-neutral amplifiers)
    "very": ("really", "truly"),
    "really": ("truly", "genuinely"),
    "extremely": ("incredibly", "exceptionally"),
    "quite": ("rather", "fairly"),
}

_LEXICAL_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, LEXICAL_EQUIVALENTS), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def apply_synonym_substitution(text: str, variant: int = 0) -> TransformResult:
    """MR-1: replace polarity-bearing adjectives and intensifiers by equivalents.

    Args:
        text: Input sentence.
        variant: Selects which equivalent is used (cycled modulo the list length).

    Returns:
        ``TransformResult`` with every dictionary word replaced (case preserved);
        ``applied`` is False when no dictionary word occurs.
    """

    def _substitute(match: re.Match[str]) -> str:
        word = match.group(0)
        options = LEXICAL_EQUIVALENTS[word.lower()]
        return _match_case(word, options[variant % len(options)])

    return _result(text, _LEXICAL_PATTERN.sub(_substitute, text))


# --------------------------------------------------------------------------- #
# MR-2: syntactic invariance
# --------------------------------------------------------------------------- #

_TERMINAL_PUNCTUATION = ".!?"
PADDING_STYLES: int = 3


def apply_punctuation_padding(text: str, variant: int = 0) -> TransformResult:
    """MR-2: inject benign whitespace / punctuation variations.

    Styles (selected by ``variant % 3``):
        0. Double every inter-word space and add trailing whitespace.
        1. Detach terminal punctuation with a space (``"good ."``); add one if absent.
        2. Wrap the sentence in straight double quotes.

    Returns:
        ``TransformResult``; ``applied`` is False only for empty/blank input.
    """
    stripped = text.strip()
    if not stripped:
        return TransformResult(text, False)
    style = variant % PADDING_STYLES
    if style == 0:
        padded = re.sub(r"\s+", "  ", stripped) + "  "
    elif style == 1:
        if stripped[-1] in _TERMINAL_PUNCTUATION:
            padded = stripped[:-1] + " " + stripped[-1]
        else:
            padded = stripped + " ."
    else:
        padded = f'"{stripped}"'
    return _result(text, padded)


PARTICIPLE_TO_PAST: dict[str, str] = {
    "directed": "directed", "written": "wrote", "composed": "composed",
    "prepared": "prepared", "built": "built", "praised": "praised",
    "criticized": "criticized", "designed": "designed", "painted": "painted",
    "cooked": "cooked", "performed": "performed", "reviewed": "reviewed",
    "made": "made", "chosen": "chose", "approved": "approved",
}

_PASSIVE_PATTERN = re.compile(
    r"^\s*(?P<obj>The\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(?:was|were)\s+(?P<part>[a-z]+)\s+by\s+"
    r"(?P<agent>[A-Za-z']+(?:\s+[A-Za-z']+){0,3})\s*(?P<punct>[.!?]?)\s*$"
)
_AGENT_STOPWORDS = frozenset({"in", "on", "at", "for", "with", "during", "after", "before", "near", "from"})


def apply_voice_conversion(text: str) -> TransformResult:
    """MR-2: convert simple passives ("The film was praised by critics.") to active voice.

    Only the pattern ``The <noun phrase> was/were <participle> by <agent>[.!?]`` with a
    known participle is converted; everything else is returned unchanged.
    """
    match = _PASSIVE_PATTERN.match(text)
    if match is None:
        return TransformResult(text, False)
    past = PARTICIPLE_TO_PAST.get(match.group("part"))
    agent = match.group("agent")
    if past is None or _AGENT_STOPWORDS & {tok.lower() for tok in agent.split()}:
        return TransformResult(text, False)
    obj = "the" + match.group("obj")[3:]
    active = f"{agent[:1].upper() + agent[1:]} {past} {obj}{match.group('punct')}"
    return _result(text, active)


PAST_TO_PARTICIPLE: dict[str, str] = {past: part for part, past in PARTICIPLE_TO_PAST.items()}
_ACTIVE_PATTERN = re.compile(
    r"^\s*(?P<agent>[A-Z][A-Za-z']*(?:\s+[A-Za-z']+){0,2})\s+(?P<verb>[a-z]+)\s+"
    r"(?P<obj>the\s+[a-z]+)\s*(?P<punct>[.!?]?)\s*$"
)


def apply_active_to_passive(text: str) -> TransformResult:
    """MR-2: convert simple actives ("Critics praised the film.") to passive voice.

    Only ``<Agent> <known past-tense verb> the <noun>[.!?]`` is converted; person names
    keep their capitalisation, other agents are lower-cased ("... was praised by critics.").
    """
    match = _ACTIVE_PATTERN.match(text)
    if match is None:
        return TransformResult(text, False)
    participle = PAST_TO_PARTICIPLE.get(match.group("verb"))
    if participle is None:
        return TransformResult(text, False)
    agent = match.group("agent")
    if agent.split()[0] not in PERSON_NAMES:
        agent = agent[:1].lower() + agent[1:]
    obj = match.group("obj")
    aux = "were" if obj.endswith("s") else "was"
    return _result(text, f"{obj[:1].upper() + obj[1:]} {aux} {participle} by {agent}{match.group('punct')}")


_ADVERBIALS: tuple[tuple[str, str], ...] = (
    ("Last week", "last week"), ("Last month", "last month"), ("Last year", "last year"),
    ("Yesterday", "yesterday"), ("This morning", "this morning"),
    ("On Monday", "on Monday"), ("On Tuesday", "on Tuesday"),
)
_FRONTED = re.compile(
    r"^(?P<adv>" + "|".join(f for f, _ in _ADVERBIALS) + r"),\s+(?P<rest>.+?)(?P<punct>[.!?])\s*$")
_POSTPOSED = re.compile(
    r"^(?P<rest>.+?)\s+(?P<adv>" + "|".join(p for _, p in _ADVERBIALS) + r")(?P<punct>[.!?])\s*$")
_LOWERABLE_FIRST = frozenset({"The", "A", "An", "Our", "We", "They", "There", "It", "This"})


def apply_adverbial_reposition(text: str) -> TransformResult:
    """MR-2: move a closed set of time adverbials between sentence-initial and final position.

    "Last week, Alice traveled to London." <-> "Alice traveled to London last week."
    """
    match = _FRONTED.match(text)
    if match is not None:
        rest = match.group("rest")
        return _result(text, f"{rest[:1].upper() + rest[1:]} {dict(_ADVERBIALS)[match.group('adv')]}{match.group('punct')}")
    match = _POSTPOSED.match(text)
    if match is not None:
        rest = match.group("rest")
        if rest.split()[0] in _LOWERABLE_FIRST:
            rest = rest[:1].lower() + rest[1:]
        fronted = {p: f for f, p in _ADVERBIALS}[match.group("adv")]
        return _result(text, f"{fronted}, {rest}{match.group('punct')}")
    return TransformResult(text, False)


_COMPLEMENTIZER = re.compile(
    r"\b(?P<head>(?:I|We|They|[A-Z][a-z]+)\s+(?:think|thought|believe|believed|said|felt|agreed|noticed))"
    r"(?P<tail>\s+(?:the|it|this|our)\b)"
)


def apply_complementizer_insertion(text: str) -> TransformResult:
    """MR-2: insert the optional complementizer "that" ("I thought the film ..." -> "I thought that the film ...")."""
    return _result(text, _COMPLEMENTIZER.sub(lambda m: f"{m.group('head')} that{m.group('tail')}", text, count=1))


# --------------------------------------------------------------------------- #
# MR-3: entity swapping
# --------------------------------------------------------------------------- #

PERSON_NAMES: tuple[str, ...] = (
    "Alice", "Bob", "Charlie", "Diana", "Ethan", "Fatima",
    "Giovanni", "Hiroshi", "Imani", "Jamal", "Priya", "Mei",
)
LOCATIONS: tuple[str, ...] = (
    "London", "Tokyo", "Paris", "Berlin", "Mumbai",
    "Lagos", "Toronto", "Sydney", "Cairo", "Seoul",
)
_ENTITY_POOLS: tuple[tuple[str, ...], ...] = (PERSON_NAMES, LOCATIONS)
_ENTITY_INDEX: dict[str, tuple[tuple[str, ...], int]] = {
    entity: (pool, idx) for pool in _ENTITY_POOLS for idx, entity in enumerate(pool)
}
_ENTITY_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, _ENTITY_INDEX), key=len, reverse=True)) + r")\b"
)


def apply_neutral_entity_swap(text: str, variant: int = 0) -> TransformResult:
    """MR-3: replace known person names and locations by other entities of the same type.

    The replacement is a deterministic cyclic shift inside the entity's pool
    (shift = ``1 + variant % (len(pool) - 1)``), so the mapping is a bijection: two
    distinct entities never collapse onto the same replacement, and no entity maps to
    itself. Substitution is performed in a single regex pass (no chained rewrites).
    """

    def _swap(match: re.Match[str]) -> str:
        pool, idx = _ENTITY_INDEX[match.group(0)]
        shift = 1 + (variant % (len(pool) - 1))
        return pool[(idx + shift) % len(pool)]

    return _result(text, _ENTITY_PATTERN.sub(_swap, text))


# --------------------------------------------------------------------------- #
# MR-4: negation inversion (litotes / double negation)
# --------------------------------------------------------------------------- #

NEGATION_MAP: dict[str, tuple[str, ...]] = {
    # positive adjective -> negated antonym (litotes)
    "good": ("not bad", "not poor"),
    "great": ("not bad at all", "not poor at all"),
    "wonderful": ("not unpleasant", "not disagreeable"),
    "fantastic": ("not unimpressive", "not unremarkable"),
    "amazing": ("not unremarkable", "not unimpressive"),
    "beautiful": ("not ugly", "not unattractive"),
    "enjoyable": ("not unenjoyable", "not unpleasant"),
    "impressive": ("not unimpressive", "not unremarkable"),
    "nice": ("not unpleasant",),
    "happy": ("not unhappy",),
    "helpful": ("not unhelpful",),
    "friendly": ("not unfriendly",),
    "clean": ("not dirty",),
    "pleasant": ("not unpleasant", "not disagreeable"),
    "lovely": ("not unattractive", "not unpleasant"),
    "exciting": ("not boring", "not dull"),
    "interesting": ("not boring", "not uninteresting"),
    "satisfying": ("not disappointing", "not unsatisfying"),
    "polite": ("not rude", "not impolite"),
    # negative adjective -> negated antonym
    "bad": ("not good", "not great"),
    "terrible": ("not good", "not acceptable"),
    "awful": ("not good", "not acceptable"),
    "horrible": ("not good", "not acceptable"),
    "boring": ("not interesting", "not exciting"),
    "disappointing": ("not satisfying", "not impressive"),
    "poor": ("not good", "not adequate"),
    "ugly": ("not beautiful", "not attractive"),
    "slow": ("not fast", "not quick"),
    "dull": ("not exciting", "not lively"),
    "annoying": ("not pleasant", "not agreeable"),
    "rude": ("not polite", "not courteous"),
    "mediocre": ("not good", "not impressive"),
    "unpleasant": ("not pleasant", "not nice"),
    "dirty": ("not clean", "not spotless"),
}

_PREDICATE_PATTERN = re.compile(
    r"\b(?P<verb>is|was|are|were|be|been|seems|seemed|felt|looked|looks|sounded|sounds)\s+"
    r"(?:(?:very|really|extremely|quite|truly|so|incredibly)\s+)?"
    r"(?P<adj>" + "|".join(sorted(map(re.escape, NEGATION_MAP), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
_EXISTING_NEGATION = re.compile(r"\b(?:not|never|no|neither|nor|nothing|nobody|none)\b|n't", re.IGNORECASE)


def apply_double_negation(text: str, variant: int = 0) -> TransformResult:
    """MR-4: rewrite predicate adjectives as litotes (``"was good"`` -> ``"was not bad"``).

    Only predicate-position adjectives (after a copular verb, optionally preceded by an
    intensifier) are rewritten, which keeps the output grammatical. Sentences that
    already contain a negation are left untouched to avoid triple negatives.

    The relation asserts *label-level* invariance: a compositional model should keep the
    polarity label, even though litotes is typically a weaker statement (that residual
    intensity loss is captured by the confidence-degradation metric).
    """
    if _EXISTING_NEGATION.search(text):
        return TransformResult(text, False)

    def _negate(match: re.Match[str]) -> str:
        options = NEGATION_MAP[match.group("adj").lower()]
        return f"{match.group('verb')} {options[variant % len(options)]}"

    return _result(text, _PREDICATE_PATTERN.sub(_negate, text))
