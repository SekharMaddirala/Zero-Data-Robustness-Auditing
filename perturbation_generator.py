"""Dataset-free seed synthesis and metamorphic test-case generation."""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from metamorphic_nlp_auditor.config import PerturbationConfig, RelationType, to_jsonable
from metamorphic_nlp_auditor.core.metamorphic_operators import (
    LOCATIONS,
    PERSON_NAMES,
    TransformResult,
    apply_active_to_passive,
    apply_adverbial_reposition,
    apply_complementizer_insertion,
    apply_double_negation,
    apply_neutral_entity_swap,
    apply_punctuation_padding,
    apply_synonym_substitution,
    apply_voice_conversion,
)

CATEGORIES: tuple[str, ...] = ("affirmation", "critique", "description", "neutral")


@dataclass(frozen=True)
class SeedSentence:
    """A synthetic baseline sentence."""

    seed_id: str
    text: str
    category: str
    template_id: str


@dataclass(frozen=True)
class MetamorphicTestCase:
    """A metamorphic pair ``(x, x')`` plus provenance.

    Attributes:
        test_id: Unique identifier, e.g. ``MR1-S0007-00``.
        seed_text: The original sentence ``x``.
        transformed_text: The metamorphic follow-up ``x'``.
        relation_type: Which MR generated the pair.
        expected_invariance: True if ``f(x)`` must equal ``f(x')``.
        seed_id: Identifier of the seed sentence.
        category: Linguistic category of the seed.
        operator: Name of the operator that produced ``x'``.
        variant: Variant index passed to the operator.
    """

    test_id: str
    seed_text: str
    transformed_text: str
    relation_type: RelationType
    expected_invariance: bool
    seed_id: str
    category: str
    operator: str
    variant: int

    def as_tuple(self) -> tuple[str, str, str, RelationType, bool]:
        """The canonical ``(test_id, seed_text, transformed_text, relation_type, expected_invariance)``."""
        return (self.test_id, self.seed_text, self.transformed_text,
                self.relation_type, self.expected_invariance)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return to_jsonable(self)


# --------------------------------------------------------------------------- #
# Lexical resources and templates for seed synthesis (no external datasets)
# --------------------------------------------------------------------------- #

_NOUNS = ("film", "meal", "concert", "book", "performance", "game", "trip", "show",
          "documentary", "album", "novel", "lecture", "exhibition", "tour", "play", "podcast")
_POS_ADJ = ("good", "great", "wonderful", "fantastic", "amazing", "beautiful", "enjoyable", "impressive",
            "nice", "pleasant", "exciting", "interesting", "satisfying", "lovely")
_NEG_ADJ = ("bad", "terrible", "awful", "boring", "disappointing", "horrible", "poor", "ugly",
            "slow", "dull", "annoying", "mediocre", "unpleasant")
_PEOPLE_POS = ("friendly", "helpful", "polite")
_PEOPLE_NEG = ("rude", "annoying")
_GROUPS = ("staff", "waiters", "guides", "nurses", "drivers", "hosts")
_PLACES = ("hotel", "restaurant", "clinic", "bakery", "museum", "library", "cinema", "airline")
_NEUTRAL_ADJ = ("clean", "slow", "small", "quiet", "modern", "crowded", "dirty")
_AGENTS = ("critics", "the audience", "local reviewers", "many visitors", "the jury", "Alice", "Priya", "Giovanni")
_FACTS = (  # (noun, past tense, participle)
    ("film", "directed", "directed"), ("novel", "wrote", "written"), ("song", "composed", "composed"),
    ("soup", "prepared", "prepared"), ("bridge", "built", "built"), ("house", "designed", "designed"),
    ("painting", "painted", "painted"), ("report", "wrote", "written"), ("meal", "cooked", "cooked"),
    ("play", "performed", "performed"),
)
_OBJECTS = ("books", "cups", "tickets", "maps", "chairs", "lamps")
_NUMBERS = ("two", "three", "four", "five", "six")
_HOURS = ("six o'clock", "seven thirty", "nine", "noon", "five fifteen")
_YEARS = ("2015", "2018", "2020", "2022")
_TIMES = ("last week", "last month", "last year", "yesterday", "this morning", "on Monday", "on Tuesday")
_INTENSIFIERS = ("very", "really", "extremely", "quite", "truly", "incredibly", "")

_SIMPLE_POOLS: dict[str, Sequence[str]] = {
    "n": _NOUNS, "city": LOCATIONS, "name": PERSON_NAMES, "place": _PLACES, "group": _GROUPS,
    "nadj": _NEUTRAL_ADJ, "agent": _AGENTS, "obj": _OBJECTS, "num": _NUMBERS, "hour": _HOURS,
    "year": _YEARS, "time": _TIMES,
}
_ADJ_POOLS: dict[str, Sequence[str]] = {
    "pos": _POS_ADJ, "neg": _NEG_ADJ, "mix": _POS_ADJ + _NEG_ADJ, "pp": _PEOPLE_POS, "pn": _PEOPLE_NEG,
}
_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _fill(pattern: str, rng: random.Random) -> str:
    """Fill the ``{slot}`` placeholders of ``pattern`` (repeated slots get the same value)."""
    chosen: dict[str, str] = {}

    def pick(key: str) -> str:
        if key in chosen:
            return chosen[key]
        if key in _SIMPLE_POOLS:
            value = rng.choice(_SIMPLE_POOLS[key])
        elif key == "n2":
            value = rng.choice([x for x in _NOUNS if x != pick("n")])
        elif key == "name2":
            value = rng.choice([x for x in PERSON_NAMES if x != pick("name")])
        elif key == "Time":
            value = pick("time")[:1].upper() + pick("time")[1:]
        elif key == "Agent":
            value = pick("agent")[:1].upper() + pick("agent")[1:]
        elif key in ("fn", "fpast", "fpart"):
            fact = rng.choice(_FACTS)
            chosen["fn"], chosen["fpast"], chosen["fpart"] = fact
            value = chosen[key]
        elif key.startswith("a_") and key[2:] in _ADJ_POOLS:
            value = rng.choice(_ADJ_POOLS[key[2:]])
        elif key.startswith("ia_") and key[3:] in _ADJ_POOLS:
            inten, adj = rng.choice(_INTENSIFIERS), rng.choice(_ADJ_POOLS[key[3:]])
            value = f"{inten} {adj}" if inten else adj
        elif key in ("Apos", "Aneg"):
            adj = rng.choice(_ADJ_POOLS[key[1:4]])
            value = f"{_article(adj)} {adj}"
        else:
            raise KeyError(f"Unknown template slot {{{key}}}")
        chosen[key] = value
        return value

    return _PLACEHOLDER.sub(lambda m: pick(m.group(1)), pattern)


_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "affirmation": (
        ("aff_pred", "The {n} was {ia_pos}."),
        ("aff_opinion", "I thought the {n} was {ia_pos}."),
        ("aff_reported", "I heard from {name} that the {n} in {city} was {ia_pos}."),
        ("aff_exclaim", "What {Apos} {n}!"),
        ("aff_staff", "The {group} at the {place} were {ia_pp}."),
        ("aff_trip", "Our trip to {city} was {ia_pos}, and {name} loved every minute."),
        ("aff_seemed", "The {n} seemed {ia_pos} to everyone."),
        ("aff_said", "{name} said the {n} was {ia_pos}."),
        ("aff_felt", "{name} felt the {n} was {ia_pos}."),
        ("aff_review", "In my opinion, the {n} was {ia_pos}."),
        ("aff_after", "After the {n}, we agreed that it was {ia_pos}."),
        ("aff_city", "The {n} in {city} was {ia_pos}."),
        ("aff_recommend", "I would recommend the {n} because it was {ia_pos}."),
        ("aff_double", "The {n} and the {n2} were both {ia_pos}."),
        ("aff_passive", "The {n} was praised by {agent}."),
        ("aff_active", "{Agent} praised the {n}."),
    ),
    "critique": (
        ("cri_pred", "The {n} was {ia_neg}."),
        ("cri_opinion", "I found the {n} to be {ia_neg}."),
        ("cri_reported", "According to {name}, the {n} in {city} was {ia_neg}."),
        ("cri_exclaim", "What {Aneg} {n}!"),
        ("cri_staff", "The {group} at the {place} were {ia_pn}."),
        ("cri_leave", "The service at the {place} was {ia_neg}, and we left early."),
        ("cri_seemed", "The {n} seemed {ia_neg} to everyone."),
        ("cri_said", "{name} said the {n} was {ia_neg}."),
        ("cri_felt", "{name} felt the {n} was {ia_neg}."),
        ("cri_review", "In my opinion, the {n} was {ia_neg}."),
        ("cri_city", "The {n} in {city} was {ia_neg}."),
        ("cri_avoid", "I would avoid the {n} because it was {ia_neg}."),
        ("cri_double", "The {n} and the {n2} were both {ia_neg}."),
        ("cri_passive", "The {n} was criticized by {agent}."),
        ("cri_active", "{Agent} criticized the {n}."),
        ("cri_complained", "{name} complained that the {n} in {city} was {ia_neg}."),
    ),
    "description": (
        ("des_passive", "The {fn} was {fpart} by {agent}."),
        ("des_passive_located", "In {city}, the {fn} was {fpart} by {agent}."),
        ("des_active", "{name} {fpast} the {fn}."),
        ("des_active_neutral", "{Agent} reviewed the {n}."),
        ("des_described", "{name} described the {n} as {ia_mix}."),
        ("des_state", "The {place} in {city} is {nadj}."),
        ("des_location", "The {place} is located near the old market in {city}."),
        ("des_time_front", "{Time}, {name} visited the {n} in {city}."),
        ("des_time_back", "{name} visited the {n} in {city} {time}."),
        ("des_clause", "The {n} that {name} saw was {ia_mix}."),
        ("des_looked", "The {place} in {city} looked {nadj} from outside."),
        ("des_two", "{name} and {name2} visited the {place} in {city}."),
        ("des_think", "I think the {n} was {ia_mix}."),
        ("des_but", "The {n} was {ia_mix}, but {name} stayed until the end."),
        ("des_found", "We visited the {n} in {city} and found it {ia_mix}."),
        ("des_report", "The report said that the {n} was {ia_mix}."),
    ),
    "neutral": (
        ("neu_travel", "{Time}, {name} traveled to {city}."),
        ("neu_open", "The {place} opens at nine in the morning."),
        ("neu_meeting", "The meeting about the {n} was held in {city}."),
        ("neu_walk", "On Monday, {name} walked from the station to the {place}."),
        ("neu_cost", "The {n} costs twelve dollars."),
        ("neu_table", "There are three {obj} on the table."),
        ("neu_time_back", "{name} traveled to {city} {time}."),
        ("neu_train", "The train to {city} leaves at {hour}."),
        ("neu_two", "{name} and {name2} met in {city} on Tuesday."),
        ("neu_floor", "The {place} is on the second floor."),
        ("neu_count", "There are {num} {obj} in the {place}."),
        ("neu_said", "{name} said that the {place} closes at {hour}."),
        ("neu_think", "I think the {place} opens at {hour}."),
        ("neu_arrive", "{name} arrived in {city} on Monday morning."),
        ("neu_release", "The {n} was released in {year}."),
        ("neu_send", "{name} sent a message to {name2} about the {n}."),
    ),
}
N_TEMPLATES: int = sum(len(v) for v in _TEMPLATES.values())


class PerturbationGenerator:
    """Builds seed sentences and pairs them with metamorphic variants (MR-1..MR-4)."""

    def __init__(self, config: PerturbationConfig, seed: int = 1234) -> None:
        """
        Args:
            config: Generation parameters.
            seed: Master seed; the generator is fully deterministic given this value.
        """
        self._config = config
        self._seed = seed

    # -- seeds -------------------------------------------------------------- #
    def generate_seeds(self) -> list[SeedSentence]:
        """Synthesise ``config.num_seeds`` unique seed sentences, balanced across categories.

        Returns:
            Seeds ordered by category, with ids ``S0000``, ``S0001``, ...

        Raises:
            RuntimeError: If the template space cannot supply enough unique sentences.
        """
        rng = random.Random(self._seed)
        base, extra = divmod(self._config.num_seeds, len(CATEGORIES))
        seeds: list[SeedSentence] = []
        seen: set[str] = set()
        for cat_idx, category in enumerate(CATEGORIES):
            quota = base + (1 if cat_idx < extra else 0)
            templates = _TEMPLATES[category]
            produced = attempts = 0
            while produced < quota:
                if attempts >= quota * 500:
                    raise RuntimeError(f"Could not synthesise {quota} unique '{category}' seeds.")
                template_id, pattern = templates[attempts % len(templates)]
                attempts += 1
                text = _fill(pattern, rng)
                if text in seen:
                    continue
                seen.add(text)
                seeds.append(SeedSentence(f"S{len(seeds):04d}", text, category, template_id))
                produced += 1
        return seeds

    # -- test cases --------------------------------------------------------- #
    def generate_test_cases(self, seeds: Sequence[SeedSentence] | None = None) -> list[MetamorphicTestCase]:
        """Pair each seed with every applicable variant of every enabled relation.

        Pairs whose transformation was not applied, and duplicate ``(seed, relation, x')``
        triples, are discarded.

        Args:
            seeds: Optional pre-built seeds; generated from the config if omitted.

        Returns:
            Deterministically ordered list of :class:`MetamorphicTestCase`.
        """
        seed_list = list(seeds) if seeds is not None else self.generate_seeds()
        cfg = self._config
        cases: list[MetamorphicTestCase] = []
        seen: set[tuple[str, RelationType, str]] = set()

        for index, seed in enumerate(seed_list):
            counters: Counter[RelationType] = Counter()
            for relation, operator, variant, result in self._variants_for(seed.text, index):
                if relation not in cfg.enabled_relations or not result.applied:
                    continue
                key = (seed.seed_id, relation, result.text)
                if key in seen:
                    continue
                seen.add(key)
                test_id = f"{relation.short_id}-{seed.seed_id}-{counters[relation]:02d}"
                counters[relation] += 1
                cases.append(MetamorphicTestCase(
                    test_id=test_id, seed_text=seed.text, transformed_text=result.text,
                    relation_type=relation, expected_invariance=True, seed_id=seed.seed_id,
                    category=seed.category, operator=operator, variant=variant,
                ))
        return cases

    def _variants_for(self, text: str, seed_index: int = 0) -> list[tuple[RelationType, str, int, TransformResult]]:
        cfg = self._config
        out: list[tuple[RelationType, str, int, TransformResult]] = []
        mr1, mr2 = RelationType.LEXICAL_EQUIVALENCE, RelationType.SYNTACTIC_INVARIANCE
        for v in range(cfg.lexical_variants):
            out.append((mr1, "synonym_substitution", v, apply_synonym_substitution(text, v)))
        if cfg.include_voice_conversion:
            out.append((mr2, "voice_conversion", 0, apply_voice_conversion(text)))
            out.append((mr2, "voice_conversion", 1, apply_active_to_passive(text)))
        if cfg.include_structural:
            out.append((mr2, "adverbial_reposition", 0, apply_adverbial_reposition(text)))
            out.append((mr2, "complementizer_insertion", 0, apply_complementizer_insertion(text)))
        if cfg.padding_variants > 0 and cfg.padding_fraction > 0:
            period = max(1, round(1.0 / cfg.padding_fraction))
            if seed_index % period == 0:
                for v in range(cfg.padding_variants):
                    out.append((mr2, "punctuation_padding", v,
                                apply_punctuation_padding(text, seed_index // period + v)))
        for v in range(cfg.entity_variants):
            out.append((RelationType.ENTITY_SWAP, "entity_swap", v, apply_neutral_entity_swap(text, v)))
        for v in range(cfg.negation_variants):
            out.append((RelationType.NEGATION_INVERSION, "double_negation", v, apply_double_negation(text, v)))
        return out

    @staticmethod
    def summarize(seeds: Sequence[SeedSentence], cases: Sequence[MetamorphicTestCase]) -> dict[str, Any]:
        """Counts of seeds/test cases by category, relation and operator."""
        return {
            "n_seeds": len(seeds),
            "n_test_cases": len(cases),
            "seeds_by_category": dict(Counter(s.category for s in seeds)),
            "cases_by_relation": dict(Counter(c.relation_type.value for c in cases)),
            "cases_by_operator": dict(Counter(c.operator for c in cases)),
        }
