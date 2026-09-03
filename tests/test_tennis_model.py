from collections import defaultdict

from src.tennis_model import PlayerState, _features, _name, _score_games


def test_score_games_handles_tiebreaks_and_retirement():
    assert _score_games("7-6(5) 3-6 6-3") == (16, 15)
    assert _score_games("6-4 2-1 RET") == (8, 5)
    assert _score_games("W/O") is None


def test_names_are_stable_across_accents_and_punctuation():
    assert _name("João Fonseca") == "joaofonseca"
    assert _name("Joao-Fonseca") == "joaofonseca"


def test_feature_vector_is_complete():
    states = defaultdict(PlayerState)
    assert len(_features(states["a"], states["b"], "Hard", 5)) == 12
