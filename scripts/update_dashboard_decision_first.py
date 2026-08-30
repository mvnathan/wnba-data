#!/usr/bin/env python3
from pathlib import Path

path = Path("docs/index.html")
text = path.read_text(encoding="utf-8")

text = text.replace(
    '<section id="today" class="panel active"><div id="status" class="statusbar">Loading…</div><div class="section"><h2>Actionable Opportunities</h2><div id="opportunities"></div></div><div id="games"></div></section>',
    '<section id="today" class="panel active"><div id="status" class="statusbar">Loading…</div><div class="section"><h2>Best Model-vs-Market Signals</h2><div class="muted" style="font-size:.7rem;margin-bottom:8px">Totals are currently the strongest validated research signal. Spread disagreements remain exploratory until sample size improves.</div><div id="opportunities"></div></div><div id="games"></div></section>'
)

text = text.replace(
    "function opportunityCard(o){const market=o.market==='spread'?'Spread':'Total';",
    "function opportunityCard(o){const market=o.market==='spread'?'Spread':'Total';const research=o.market==='total'&&Number(o.edge)>=6?'Research-supported':'Exploratory';"
)
text = text.replace(
    "${o.side} · ${N(o.edge)} pt edge</div><div class=\"oppmeta\">Market:",
    "${o.side} · ${N(o.edge)} pt edge</div><div class=\"oppmeta\">${research} · Market:"
)

text = text.replace(
    '<div class="sectiontitle">Independent Model Forecast</div>',
    '<div class="sectiontitle">Independent Forecast · Market Excluded</div>'
)
text = text.replace(
    '<div class="sectiontitle">Model vs Market · Edge</div>',
    '<div class="sectiontitle">Decision Signal · Model vs Market</div>'
)
text = text.replace(
    '<div class="metric"><div class="k">Model Spread</div><div class="v">${spread(g,\'model_predicted_margin\')}</div></div>',
    '<div class="metric"><div class="k">Model Spread</div><div class="v">${spread(g,\'model_predicted_margin\')}</div><div class="note">Edge ${E(g.model_market_margin_edge)?N(Math.abs(g.model_market_margin_edge)):"—"} pts · exploratory</div></div>'
)
text = text.replace(
    '<div class="metric"><div class="k">Model Win Prob</div><div class="v">${E(g.model_home_win_probability)?(Number(g.model_home_win_probability)>=.5?H:A)+\' \'+P(Math.max(Number(g.model_home_win_probability),1-Number(g.model_home_win_probability))):\'—\'}</div>',
    '<div class="metric"><div class="k">Model Win Prob</div><div class="v">${E(g.model_home_win_probability)?(Number(g.model_home_win_probability)>=.5?H:A)+\' \'+P(Math.max(Number(g.model_home_win_probability),1-Number(g.model_home_win_probability))):\'—\'}</div>'
)

# Add a compact signal banner before the model forecast.
needle = '${lp&&E(g.live_projected_home_score)?`<div class="livehero"'
if needle in text and 'signalBanner(g)' not in text:
    insert_after = "function marketSpread(g){if(!E(g.market_home_spread))return'—';const s=Number(g.market_home_spread);if(Math.abs(s)<.05)return'PK';return `${ab(g,'home')} ${s>0?'+':''}${s.toFixed(1)}`}\n"
    helper = "function signalBanner(g){const te=E(g.model_market_total_edge)?Number(g.model_market_total_edge):null;const se=E(g.model_market_margin_edge)?Number(g.model_market_margin_edge):null;if(te!==null&&Math.abs(te)>=6){const side=te>0?'OVER':'UNDER';return `<div class=\"statusbar\" style=\"border-color:rgba(104,217,157,.55);background:rgba(104,217,157,.08)\"><strong>${side} ${N(g.market_total)}</strong> · model ${N(g.model_predicted_total||g.predicted_total)} · ${N(Math.abs(te))} pt total edge · strongest current research signal</div>`}if(se!==null&&Math.abs(se)>=4){const side=se>0?ab(g,'home'):ab(g,'away');return `<div class=\"statusbar\" style=\"border-color:rgba(243,204,108,.45)\"><strong>${side} spread lean</strong> · ${N(Math.abs(se))} pt model edge · exploratory spread signal</div>`}return ''}\n"
    text = text.replace(insert_after, insert_after + helper)
    text = text.replace('<div class="body">${lp||f?', '<div class="body">${signalBanner(g)}${lp||f?')

path.write_text(text, encoding="utf-8")
print("Updated dashboard to decision-first independent-model layout")
