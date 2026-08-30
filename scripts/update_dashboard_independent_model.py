#!/usr/bin/env python3
from pathlib import Path

path = Path("docs/index.html")
text = path.read_text(encoding="utf-8")
replacements = {
    "Actionable edges first · market-adjusted forecast · pure model shown separately": "Independent model · market is benchmark only · actionable disagreement highlighted",
    "Decision Forecast · Market Adjusted": "Independent Model Forecast",
    "Derived from adjusted margin + total": "Derived from independent model margin + total",
    "Actionable blended forecast": "Independent model forecast",
    "Pure Model · Before Market Anchor": "Model vs Market · Edge",
    "Market weight ${E(g.market_total_weight)?P(g.market_total_weight):'—'}": "Market is not used in prediction",
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8")
print("Updated dashboard labels for independent-model architecture")
