from src.market_math import american_implied_probability, no_vig_two_way_probabilities


def test_american_implied_probability():
    assert round(american_implied_probability(-110), 6) == round(110 / 210, 6)
    assert round(american_implied_probability(100), 6) == 0.5


def test_no_vig_two_way_probabilities_remove_hold():
    home, away = no_vig_two_way_probabilities(-109, -110)
    assert home is not None and away is not None
    assert round(home + away, 12) == 1.0
    assert round(home, 4) == 0.4989
    assert round(away, 4) == 0.5011


def test_no_vig_requires_both_sides():
    assert no_vig_two_way_probabilities(-110, None) == (None, None)
