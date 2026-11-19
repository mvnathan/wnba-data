from datetime import datetime, timedelta, timezone
from market_engine import MarketQuote, evaluate, implied_probability


def quote(**overrides):
    now = datetime.now(timezone.utc)
    values = dict(event_id="e1", market_id="m1", selection_id="s1",
                  market_name="2nd half total", period="H2", american_odds=-110,
                  quote_timestamp=now, game_timestamp=now)
    values.update(overrides)
    return MarketQuote(**values)


def test_american_odds():
    assert round(implied_probability(-110), 4) == 0.5238
    assert round(implied_probability(150), 4) == 0.4


def test_unmodeled_is_not_alert():
    assert evaluate(quote(), None).status == "UNMODELED"


def test_suspended_is_not_alert():
    assert evaluate(quote(suspended=True), 0.7).status == "SUSPENDED"


def test_stale_is_not_alert():
    old = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert evaluate(quote(quote_timestamp=old), 0.7).status == "STALE_QUOTE"


def test_research_alert():
    assert evaluate(quote(), 0.6).status == "RESEARCH_ALERT"
