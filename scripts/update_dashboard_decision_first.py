#!/usr/bin/env python3
from pathlib import Path

path = Path("docs/index.html")
text = path.read_text(encoding="utf-8")

text = text.replace('<h2>Actionable Opportunities</h2>', '<h2>Best Model-vs-Market Signals</h2>')
text = text.replace('<div class="sectiontitle">Independent Model Forecast</div>', '<div class="sectiontitle">Independent Forecast · Market Excluded</div>')
text = text.replace('<div class="sectiontitle">Model vs Market · Edge</div>', '<div class="sectiontitle">Decision Signal · Model vs Market</div>')

if '.compareviz{' not in text:
    text = text.replace('.empty{padding:14px;color:var(--muted)}', '.empty{padding:14px;color:var(--muted)}.compareviz{padding:11px;border:1px solid var(--border);border-radius:12px;background:var(--panel2);margin:8px 0}.vizhead{display:flex;justify-content:space-between;gap:8px;font-size:.68rem;font-weight:900;margin-bottom:7px}.track{position:relative;height:9px;border-radius:999px;background:#26303d;margin:17px 4px 10px}.range{position:absolute;top:2px;height:5px;border-radius:999px;background:rgba(132,169,255,.5)}.mark{position:absolute;top:-6px;width:3px;height:21px;border-radius:2px;background:#f5f7fb}.mark.dk{background:#f3cc6c}.mark.model{background:#68d99d;width:4px}.vizlabels{display:flex;justify-content:space-between;gap:8px;font-size:.6rem;color:var(--muted)}.edgepill{display:inline-block;padding:3px 7px;border-radius:999px;border:1px solid var(--border);font-size:.6rem;font-weight:900}.edgepill.good{border-color:rgba(104,217,157,.55);color:var(--good)}')

helper_anchor = "function marketSpread(g){if(!E(g.market_home_spread))return'—';const s=Number(g.market_home_spread);if(Math.abs(s)<.05)return'PK';return `${ab(g,'home')} ${s>0?'+':''}${s.toFixed(1)}`}\n"
if helper_anchor in text and 'function comparisonViz(g)' not in text:
    helper = r'''function pos(v,min,max){if(!E(v)||max<=min)return 50;return Math.max(2,Math.min(98,(Number(v)-min)/(max-min)*100))}
function comparisonViz(g){const mt=E(g.model_predicted_total)?Number(g.model_predicted_total):E(g.predicted_total)?Number(g.predicted_total):null,dk=E(g.market_total)?Number(g.market_total):null,ct=E(g.consensus_total)?Number(g.consensus_total):null,tlo=E(g.market_total_min)?Number(g.market_total_min):ct,thi=E(g.market_total_max)?Number(g.market_total_max):ct;const mm=E(g.model_predicted_margin)?Number(g.model_predicted_margin):null,dks=E(g.market_home_spread)?-Number(g.market_home_spread):null,cs=E(g.consensus_home_spread)?-Number(g.consensus_home_spread):null,slo=E(g.market_home_spread_max)?-Number(g.market_home_spread_max):cs,shi=E(g.market_home_spread_min)?-Number(g.market_home_spread_min):cs;function one(title,model,dkv,cons,lo,hi,edge,total){if(model===null||dkv===null)return'';const vals=[model,dkv,cons,lo,hi].filter(v=>v!==null&&Number.isFinite(v));let mn=Math.min(...vals),mx=Math.max(...vals);const pad=Math.max(2,(mx-mn)*.25);mn-=pad;mx+=pad;const ep=E(edge)?Number(edge):model-(cons??dkv);const cls=total&&Math.abs(ep)>=6?' good':'';return `<div class="compareviz"><div class="vizhead"><span>${title}</span><span class="edgepill${cls}">${ep>=0?'+':''}${N(ep)} vs consensus</span></div><div class="track"><span class="range" style="left:${pos(lo??cons,mn,mx)}%;width:${Math.max(1,pos(hi??cons,mn,mx)-pos(lo??cons,mn,mx))}%"></span><span class="mark dk" style="left:${pos(dkv,mn,mx)}%"></span><span class="mark" style="left:${pos(cons??dkv,mn,mx)}%"></span><span class="mark model" style="left:${pos(model,mn,mx)}%"></span></div><div class="vizlabels"><span>DK ${N(dkv)}</span><span>Consensus ${N(cons)} · ${g.market_book_count||1} books</span><span>Model ${N(model)}</span></div></div>`}return one('TOTAL · market range vs model',mt,dk,ct,tlo,thi,g.model_consensus_total_edge,true)+one('MARGIN · home-team perspective',mm,dks,cs,slo,shi,g.model_consensus_margin_edge,false)}
'''
    text = text.replace(helper_anchor, helper_anchor + helper)

# Put the comparison visualization before detailed forecast metrics.
text = text.replace('<div class="sectiontitle">Independent Forecast · Market Excluded</div>', '${comparisonViz(g)}<div class="sectiontitle">Independent Forecast · Market Excluded</div>')

# Upgrade market metric to show DK and consensus explicitly.
old = '<div class="metric decision"><div class="k">Market</div><div class="v">${marketSpread(g)} · O/U ${N(g.market_total)}</div><div class="note ${marketFresh?\'fresh\':\'stale\'}">${marketFresh?\'Fresh\':\'⚠ Stale\'} · ${ago(g.market_updated_at)}</div></div>'
new = '<div class="metric decision"><div class="k">DK / Consensus</div><div class="v">DK ${marketSpread(g)} · ${N(g.market_total)}</div><div class="note">Consensus spread ${E(g.consensus_home_spread)?(Number(g.consensus_home_spread)>0?"+":"")+N(g.consensus_home_spread):"—"} · total ${N(g.consensus_total)} · ${g.market_book_count||1} books</div><div class="note ${marketFresh?\'fresh\':\'stale\'}">${marketFresh?\'Fresh\':\'⚠ Stale\'} · ${ago(g.market_updated_at)}</div></div>'
text = text.replace(old,new)

path.write_text(text, encoding="utf-8")
print("Updated dashboard with DK, consensus and visual model-market ranges")
