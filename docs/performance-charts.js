(function(){
const STYLE=`
.px{--px-accent:var(--accent,#84a9ff);--px-line:var(--line,var(--border,#27303d));--px-panel:var(--panel2,#0d131b);--px-muted:var(--muted,#98a3b3)}
.px-controls{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0}.px-control{display:flex;flex-direction:column;gap:5px}.px-control label{font-size:.72rem;color:var(--px-muted);font-weight:800}.px select,.px input[type=range]{width:100%}.px select{background:var(--px-panel);color:inherit;border:1px solid var(--px-line);border-radius:9px;padding:9px;font:inherit}.px-range{grid-column:1/-1}.px-rangehead{display:flex;justify-content:space-between;font-size:.72rem;color:var(--px-muted)}.px-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:12px 0}.px-stat{padding:10px;border:1px solid var(--px-line);border-radius:11px;background:var(--px-panel)}.px-stat b{display:block;font-size:1.1rem}.px-stat span{font-size:.65rem;color:var(--px-muted)}.px-charts{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.px-chart{border:1px solid var(--px-line);border-radius:12px;background:var(--px-panel);padding:10px;min-width:0}.px-chart h3{font-size:.72rem;margin:0 0 2px}.px-chart p{font-size:.62rem;color:var(--px-muted);margin:0 0 7px}.px-chart svg{display:block;width:100%;height:150px;overflow:visible}.px-axis{stroke:var(--px-line);stroke-width:1}.px-line{fill:none;stroke:var(--px-accent);stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}.px-dot{fill:var(--px-accent)}.px-avg{stroke:#f3cc6c;stroke-width:1.5;stroke-dasharray:4 4}.px-label{fill:var(--px-muted);font-size:9px}.px-table{margin-top:12px;overflow-x:auto}.px-row{display:grid;grid-template-columns:minmax(170px,1.6fr) repeat(4,minmax(82px,1fr));gap:7px;padding:8px 3px;border-bottom:1px solid var(--px-line);font-size:.7rem;min-width:560px}.px-row.px-head{color:var(--px-muted);font-size:.62rem;font-weight:900;text-transform:uppercase}.px-good{color:#68d99d}.px-bad{color:#ff8b91}.px-empty{padding:18px;color:var(--px-muted)}
@media(max-width:760px){.px-summary{grid-template-columns:1fr 1fr}.px-charts{grid-template-columns:1fr}.px-chart svg{height:170px}}
`;
if(!document.getElementById('performance-explorer-style')){const s=document.createElement('style');s.id='performance-explorer-style';s.textContent=STYLE;document.head.appendChild(s)}
const finite=v=>v!==null&&v!==undefined&&v!==''&&Number.isFinite(Number(v));
const pct=v=>finite(v)?(Number(v)*100).toFixed(1)+'%':'—';
const num=(v,d=1)=>finite(v)?Number(v).toFixed(d):'—';
const mean=(rows,key)=>{const a=rows.map(r=>Number(r[key])).filter(Number.isFinite);return a.length?a.reduce((x,y)=>x+y,0)/a.length:null};
function normalize(payload,sport){
 if(sport==='tennis')return (payload.matches||[]).map(r=>({
  date:r.start_time_utc||r.target_date,label:`${r.player_1} vs ${r.player_2}`,group:r.tour,
  winner_correct:Boolean(r.winner_correct),brier:Number(r.brier_score),margin:Number(r.absolute_margin_error),total:Number(r.absolute_total_error),
  predicted:`${r.predicted_winner} · ${pct(Math.max(Number(r.player_1_win_probability),1-Number(r.player_1_win_probability)))}`,
  actual:`${r.actual_winner_player_1?r.actual_player_1:r.actual_player_2} · ${num(r.actual_player_1_games,0)}–${num(r.actual_player_2_games,0)}`
 }));
 return (payload.games||[]).map(r=>({
  date:r.game_date_utc,label:`${r.away_abbr||r.away_team} @ ${r.home_abbr||r.home_team}`,group:'WNBA',
  winner_correct:Boolean(r.winner_correct),brier:Number(r.brier_score),margin:Number(r.absolute_margin_error),total:Number(r.absolute_total_error),
  predicted:`${r.predicted_winner_abbr||'—'} · ${num(r.predicted_away_score)}–${num(r.predicted_home_score)}`,
  actual:`${r.actual_winner_abbr||'—'} · ${num(r.actual_away_score,0)}–${num(r.actual_home_score,0)}`
 }));
}
function chart(title,subtitle,values,format,domain){
 const W=320,H=150,L=28,R=8,T=10,B=22,n=Math.max(1,values.length),valid=values.filter(Number.isFinite);
 if(!valid.length)return `<div class="px-chart"><h3>${title}</h3><div class="px-empty">No data</div></div>`;
 let lo=domain?domain[0]:Math.min(0,...valid),hi=domain?domain[1]:Math.max(...valid);if(hi<=lo)hi=lo+1;
 const x=i=>L+(W-L-R)*(n===1?.5:i/(n-1)),y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
 const points=values.map((v,i)=>Number.isFinite(v)?`${x(i)},${y(v)}`:null).filter(Boolean).join(' '),avg=valid.reduce((a,b)=>a+b,0)/valid.length;
 const dots=values.map((v,i)=>Number.isFinite(v)?`<circle class="px-dot" cx="${x(i)}" cy="${y(v)}" r="${n<18?3:1.8}"/>`:'').join('');
 return `<div class="px-chart"><h3>${title}</h3><p>${subtitle}</p><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}"><line class="px-axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="px-avg" x1="${L}" y1="${y(avg)}" x2="${W-R}" y2="${y(avg)}"/><polyline class="px-line" points="${points}"/>${dots}<text class="px-label" x="2" y="${T+5}">${format(hi)}</text><text class="px-label" x="2" y="${H-B}">${format(lo)}</text><text class="px-label" x="${W-R-55}" y="${Math.max(10,y(avg)-5)}">avg ${format(avg)}</text></svg></div>`;
}
window.mountPerformanceExplorer=async function({rootId,url,sport}){
 const root=document.getElementById(rootId);if(!root)return;root.classList.add('px');root.innerHTML='<div class="px-empty">Loading settled performance…</div>';
 try{
  const response=await fetch(url+'?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('Performance file unavailable');
  const rows=normalize(await response.json(),sport).sort((a,b)=>String(a.date).localeCompare(String(b.date)));
  if(!rows.length){root.innerHTML='<div class="px-empty">No settled predictions yet.</div>';return}
  const groups=[...new Set(rows.map(r=>r.group))];
  root.innerHTML=`<div class="px-controls"><div class="px-control"><label>Tour / league</label><select class="px-group"><option value="ALL">All</option>${groups.map(g=>`<option>${g}</option>`).join('')}</select></div><div class="px-control"><label>Detail level</label><select class="px-window"><option value="1">One ${sport==='tennis'?'match':'game'}</option><option value="5">Last 5</option><option value="10" selected>Last 10</option><option value="20">Last 20</option><option value="ALL">Full season</option></select></div><div class="px-control px-range"><div class="px-rangehead"><label>Move through settled predictions</label><span class="px-period"></span></div><input class="px-end" type="range" min="1" step="1"/></div></div><div class="px-summary"></div><div class="px-charts"></div><div class="px-table"></div>`;
  const group=root.querySelector('.px-group'),windowSelect=root.querySelector('.px-window'),end=root.querySelector('.px-end');
  function draw(){
   const filtered=rows.filter(r=>group.value==='ALL'||r.group===group.value);end.max=String(Math.max(1,filtered.length));if(!end.value||Number(end.value)>filtered.length)end.value=String(filtered.length);
   const finish=Number(end.value),size=windowSelect.value==='ALL'?filtered.length:Number(windowSelect.value),shown=filtered.slice(Math.max(0,finish-size),finish);
   const acc=mean(shown,'winner_correct'),brier=mean(shown,'brier'),margin=mean(shown,'margin'),total=mean(shown,'total');
   root.querySelector('.px-period').textContent=shown.length?`${new Date(shown[0].date).toLocaleDateString()} – ${new Date(shown[shown.length-1].date).toLocaleDateString()} · ${shown.length}`:'—';
   root.querySelector('.px-summary').innerHTML=`<div class="px-stat"><b>${shown.length}</b><span>Settled ${sport==='tennis'?'matches':'games'}</span></div><div class="px-stat"><b>${pct(acc)}</b><span>Winner accuracy</span></div><div class="px-stat"><b>${num(margin)}</b><span>${sport==='tennis'?'Spread':'Margin'} MAE</span></div><div class="px-stat"><b>${num(total)}</b><span>Total MAE</span></div>`;
   root.querySelector('.px-charts').innerHTML=chart('Winner result',`Accuracy ${pct(acc)} · Brier ${num(brier,3)}`,shown.map(r=>r.winner_correct?1:0),v=>Math.round(v*100)+'%',[0,1])+chart(`${sport==='tennis'?'Spread':'Margin'} error`,`Absolute error · lower is better`,shown.map(r=>r.margin),v=>num(v),null)+chart('Total error','Absolute error · lower is better',shown.map(r=>r.total),v=>num(v),null);
   root.querySelector('.px-table').innerHTML=`<div class="px-row px-head"><span>${sport==='tennis'?'Match':'Game'}</span><span>Result</span><span>Prediction</span><span>${sport==='tennis'?'Spread':'Margin'} error</span><span>Total error</span></div>${shown.slice().reverse().map(r=>`<div class="px-row"><b>${r.label}</b><span class="${r.winner_correct?'px-good':'px-bad'}">${r.winner_correct?'Correct':'Miss'}</span><span title="Actual: ${r.actual}">${r.predicted}</span><span>${num(r.margin)}</span><span>${num(r.total)}</span></div>`).join('')}`;
  }
  group.onchange=()=>{end.value='';draw()};windowSelect.onchange=draw;end.oninput=draw;draw();
 }catch(error){root.innerHTML=`<div class="px-empty">${error.message}</div>`}
};
})();
