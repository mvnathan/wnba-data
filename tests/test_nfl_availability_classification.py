import unittest
from src.availability_intelligence import extract_status
from scripts.generate_nfl_shadow_model import availability_adjustments


class AvailabilityClassificationTests(unittest.TestCase):
    def test_clearances_and_negation(self):
        self.assertEqual(extract_status("Eagles RB Saquon Barkley cleared to play after being ruled out earlier"), "available")
        self.assertEqual(extract_status("Commanders QB Jayden Daniels will start despite earlier injury concern"), "available")
        self.assertEqual(extract_status("Giants QB Smith is not ruled out"), "available")
        self.assertEqual(extract_status("Bears QB Smith is out for the game"), "out")
        self.assertEqual(extract_status("Bears QB Smith is questionable"), "questionable")
        self.assertIsNone(extract_status("Chargers WR Tre Harris questionable to return today"))

    def test_later_clearance_cancels_earlier_out(self):
        game = {"game_id": "g1", "game_date_utc": "2026-10-11T17:00:00Z",
                "home_abbr": "PHI", "away_abbr": "WAS"}
        signal = lambda text, when: {
            "sport": "NFL", "source": "@reporter", "text": text,
            "status_signal": "out", "confidence": 0.8, "published_at_utc": when
        }
        reports = {"signals": [
            signal("Eagles RB Saquon Barkley ruled out", "2026-10-10T17:00:00Z"),
            signal("Eagles RB Saquon Barkley cleared to play", "2026-10-11T16:00:00Z"),
            signal("Commanders QB Jayden Daniels ruled out", "2026-10-11T18:00:00Z")
        ]}
        self.assertEqual(availability_adjustments({"games": [game]}, reports)["g1"], [])


if __name__ == "__main__":
    unittest.main()
