(function(){
function E(v){return v!==null&&v!==undefined&&!Number.isNaN(Number(v))}
function pct(v){return E(v)?(Number(v)*100).toFixed(1)+'%':'—'}
function esc(s){return String(s??'').replace(/[&<>"]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[x]))}
function rec(x){if(!x)return '—';return x.graded?x.wins+'–'+x.losses+(x.pushes?'–'+x.pushes:''):'—'}
function tile(label,x,note){
  const available=x&&x.graded;
  return '<div class="mp-tile '+(available?'':'mp-na')+'"><div class="mp-label">'+label+'</div><div class="mp-rate">'+(available?pct(x.rate):'N/A')+'</div><div class="mp-record">'+(available?rec(x)+' · '+x.graded+' graded':'No retained market history')+'</div><div class="mp-note">'+esc(note||'')+'</div></div>'
}
function mini(rows,key){
 const data=(rows||[]).filter(r=>r[key]&&r[key].graded&&E(r[key].rate));
 if(!data.length)return '<div class="mp-empty">Prospective tracking will populate this trend.</div>';
 const W=680,H=150,L=34,R=10,T=10,B=28;
 const x=i=>L+(W-L-R)*(data.length===1?.5:i/(data.length-1));
 const y=v=>T+(H-T-B)*(1-(v-.35)/.40);
 const pts=data.map((r,i)=>x(i)+','+y(Math.max(.35,Math.min(.75,r[key].rate)))).join(' ');
 return '<svg class="mp-svg" viewBox="0 0 '+W+' '+H+'"><line class="mp-base" x1="'+L+'" x2="'+(W-R)+'" y1="'+y(.5)+'" y2="'+y(.5)+'"/><polyline class="mp-line" points="'+pts+'"/>'+data.map((r,i)=>'<circle class="mp-dot" cx="'+x(i)+'" cy="'+y(r[key].rate)+'" r="4"><title>'+esc(r.week_label)+' · '+pct(r[key].rate)+' · '+rec(r[key])+'</title></circle>').join('')+data.map((r,i)=>'<text class="mp-axis" x="'+x(i)+'" y="'+(H-8)+'" text-anchor="middle">'+esc(r.week_label.split('–')[0])+'</text>').join('')+'</svg>'
}
function markup(m,d){
 const o=m.overall||{},notes=m.market_notes||{};
 return '<div class="mp-wrap"><div class="mp-tiles">'+tile('Moneyline',o.ml,notes.ml)+tile('Against spread',o.ats,notes.ats)+tile('Over / Under',o.ou,notes.ou)+'</div><div class="mp-trends"><div class="mp-trend"><div class="mp-title">ML weekly hit rate</div>'+mini(m.weekly,'ml')+'</div><div class="mp-trend"><div class="mp-title">ATS weekly hit rate</div>'+mini(m.weekly,'ats')+'</div><div class="mp-trend"><div class="mp-title">O/U weekly hit rate</div>'+mini(m.weekly,'ou')+'</div></div><div class="mp-foot">'+esc(d.metric_definition)+' '+esc(d.method_note)+'</div></div>'
}
window.mountMarketPerformance=async function(opts){
 const root=document.getElementById(opts.rootId);if(!root)return;
 try{const r=await fetch((opts.url||'model-performance-weekly.json')+'?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);const d=await r.json();const m=(d.models||[]).find(x=>x.sport===opts.sport);root.innerHTML=m?markup(m,d):'<div class="mp-empty">No performance history yet.</div>'}catch(e){root.innerHTML='<div class="mp-empty">Performance data unavailable.</div>'}
}
window.marketPerformanceMarkup=markup;
window.marketPerformanceHelpers={pct,rec,esc,mini};
})();