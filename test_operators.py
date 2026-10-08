"""Unit tests for the metamorphic operators."""

from metamorphic_nlp_auditor.core.metamorphic_operators import (
    apply_double_negation,
    apply_neutral_entity_swap,
    apply_punctuation_padding,
    apply_synonym_substitution,
    apply_voice_conversion,
)


def test_synonym_substitution_replaces_and_preserves_case():
    text, applied = apply_synonym_substitution("The film was very good.")
    assert applied and text == "The film was really fine."
    text, applied = apply_synonym_substitution("Great film.")
    assert applied and text == "Excellent film."


def test_synonym_substitution_not_applied_without_dictionary_word():
    text, applied = apply_synonym_substitution("The meeting opens at nine.")
    assert not applied and text == "The meeting opens at nine."


def test_synonym_variants_differ():
    assert apply_synonym_substitution("A good film.", 0).text != apply_synonym_substitution("A good film.", 1).text


def test_punctuation_padding_styles():
    base = "The film was good."
    assert apply_punctuation_padding(base, 0).text == "The  film  was  good.  "
    assert apply_punctuation_padding(base, 1).text == "The film was good ."
    assert apply_punctuation_padding(base, 2).text == '"The film was good."'
    assert apply_punctuation_padding("   ", 0) == ("   ", False)


def test_voice_conversion():
    text, applied = apply_voice_conversion("The film was praised by critics.")
    assert applied and text == "Critics praised the film."
    text, applied = apply_voice_conversion("The novel was written by Alice.")
    assert applied and text == "Alice wrote the novel."


def test_voice_conversion_rejects_non_passive_and_prepositional_agents():
    assert not apply_voice_conversion("The film was very good.").applied
    assert not apply_voice_conversion("In Paris, the film was praised by critics.").applied
    assert not apply_voice_conversion("The film was praised by critics in Paris.").applied


def test_entity_swap_is_bijective_and_single_pass():
    text, applied = apply_neutral_entity_swap("Alice met Bob in London.")
    assert applied and text == "Bob met Charlie in Tokyo."
    assert apply_neutral_entity_swap("Alice met Bob.", 3).text != "Alice met Bob."


def test_entity_swap_not_applied_without_entities():
    assert not apply_neutral_entity_swap("The film opens at nine.").applied


def test_double_negation_predicate_only():
    assert apply_double_negation("The food was very good.") == ("The food was not bad.", True)
    assert apply_double_negation("The service was bad.") == ("The service was not good.", True)
    assert not apply_double_negation("What a good film!").applied


def test_double_negation_skips_existing_negation():
    assert not apply_double_negation("The food was not good.").applied
    assert not apply_double_negation("The food wasn't good.").applied


def test_operators_are_deterministic():
    s = "Alice said the film in London was very good."
    assert apply_synonym_substitution(s, 1) == apply_synonym_substitution(s, 1)
    assert apply_neutral_entity_swap(s, 1) == apply_neutral_entity_swap(s, 1)


def test_active_to_passive():
    from metamorphic_nlp_auditor.core.metamorphic_operators import apply_active_to_passive
    assert apply_active_to_passive("Critics praised the film.") == ("The film was praised by critics.", True)
    assert apply_active_to_passive("Alice wrote the novel.") == ("The novel was written by Alice.", True)
    assert not apply_active_to_passive("The film was very good.").applied


def test_adverbial_reposition_both_directions():
    from metamorphic_nlp_auditor.core.metamorphic_operators import apply_adverbial_reposition
    assert apply_adverbial_reposition("Last week, Alice traveled to London.").text == "Alice traveled to London last week."
    assert apply_adverbial_reposition("Alice traveled to London last week.").text == "Last week, Alice traveled to London."
    assert apply_adverbial_reposition("On Monday, the meeting was held.").text == "The meeting was held on Monday."
    assert not apply_adverbial_reposition("The film was good.").applied


def test_complementizer_insertion():
    from metamorphic_nlp_auditor.core.metamorphic_operators import apply_complementizer_insertion
    assert apply_complementizer_insertion("I thought the film was good.").text == "I thought that the film was good."
    assert apply_complementizer_insertion("Alice said the film was good.").applied
    assert not apply_complementizer_insertion("Alice said that the film was good.").applied
