(function(){
const E=v=>v!==null&&v!==undefined&&v!==''&&!Number.isNaN(Number(v));
const N=(v,d=1)=>E(v)?Number(v).toFixed(d):'—';
const P=v=>E(v)?(Number(v)*100).toFixed(0)+'%':'—';
const esc=s=>String(s??'').replace(/[&<>"]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[x]));

function marketValue(g){
  const vals=[];
  if(E(g.model_market_spread_edge)) vals.push({kind:'spread',v:Number(g.model_market_spread_edge),txt:(Number(g.model_market_spread_edge)>=0?(g.home_abbr||'HOME'):(g.away_abbr||'AWAY'))+' spread · '+N(Math.abs(g.model_market_spread_edge))+' pt gap'});
  if(E(g.model_market_margin_edge)) vals.push({kind:'spread',v:Number(g.model_market_margin_edge),txt:(Number(g.model_market_margin_edge)>=0?(g.home_abbr||'HOME'):(g.away_abbr||'AWAY'))+' spread · '+N(Math.abs(g.model_market_margin_edge))+' pt gap'});
  if(E(g.model_market_total_edge)) vals.push({kind:'total',v:Number(g.model_market_total_edge),txt:(Number(g.model_market_total_edge)>=0?'OVER ':'UNDER ')+(E(g.market_total)?N(g.market_total):'market total')+' · '+N(Math.abs(g.model_market_total_edge))+' pt gap'});
  if(E(g.model_market_home_win_edge)) vals.push({kind:'ml',v:Number(g.model_market_home_win_edge)*100,txt:(Number(g.model_market_home_win_edge)>=0?(g.home_abbr||'HOME'):(g.away_abbr||'AWAY'))+' ML · '+N(Math.abs(Number(g.model_market_home_win_edge)*100),0)+' pp gap'});
  if(E(g.model_consensus_total_edge)) vals.push({kind:'total',v:Number(g.model_consensus_total_edge),txt:(Number(g.model_consensus_total_edge)>=0?'OVER ':'UNDER ')+'consensus · '+N(Math.abs(g.model_consensus_total_edge))+' pt gap'});
  if(E(g.model_consensus_margin_edge)) vals.push({kind:'spread',v:Number(g.model_consensus_margin_edge),txt:'Spread disagreement · '+N(Math.abs(g.model_consensus_margin_edge))+' pt gap'});
  return vals.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v))[0]||null;
}

function baseRead(sport,g){
  const hp=E(g.home_win_probability)?Number(g.home_win_probability):E(g.model_home_win_probability)?Number(g.model_home_win_probability):null;
  const ap=E(g.away_win_probability)?Number(g.away_win_probability):(hp!==null?1-hp:null);
  const winner=g.predicted_winner||(hp!==null?(hp>=.5?(g.home_team||g.home_abbr):(g.away_team||g.away_abbr)):'Model');
  const conf=hp!==null&&ap!==null?Math.max(hp,ap):null;
  const margin=E(g.predicted_margin)?Number(g.predicted_margin):E(g.model_predicted_margin)?Number(g.model_predicted_margin):null;
  const total=E(g.predicted_total)?Number(g.predicted_total):E(g.model_predicted_total)?Number(g.model_predicted_total):null;
  return {read:winner+(conf!==null?' '+P(conf):'')+(margin!==null?' · margin '+(margin>=0?'+':'')+N(margin):'')+(total!==null?' · total '+N(total):''),value:marketValue(g)};
}

function nba(g){
  const b=baseRead('NBA',g),rc=g.roster_context||{},h=rc.home||{},a=rc.away||{},x=g.x_availability_adjustments||[];
  const why=['Recent possession-adjusted form and four-factor profile'];
  if(E(h.continuity)&&E(a.continuity)) why.push((h.continuity>a.continuity?(g.home_abbr||'Home'):(g.away_abbr||'Away'))+' has better roster continuity ('+P(Math.max(h.continuity,a.continuity))+')');
  if(x.length) why.push(x.length+' availability adjustment'+(x.length===1?'':'s')+' from trusted X reporting');
  const risk=[];
  if(E(h.incoming_share)&&Number(h.incoming_share)>.35) risk.push((g.home_abbr||'Home')+' roster turnover is high');
  if(E(a.incoming_share)&&Number(a.incoming_share)>.35) risk.push((g.away_abbr||'Away')+' roster turnover is high');
  if(!g.market_bookmaker) risk.push('No current market benchmark');
  if(!risk.length) risk.push('Normal early-season / availability uncertainty');
  return {read:b.read,value:b.value?.txt||'No actionable market disagreement yet',why:why.join('. ')+'.',risk:risk.join('. ')+'.'};
}

function nfl(g){
  const b=baseRead('NFL',g),c=g.competitive_context||{},h=c.home||{},a=c.away||{};
  const why=['Model combines team form, scoring efficiency, rest and QB/roster continuity'];
  if(E(h.rest_rotation_risk)||E(a.rest_rotation_risk)){const side=Number(h.rest_rotation_risk||0)>Number(a.rest_rotation_risk||0)?(g.home_abbr||'Home'):(g.away_abbr||'Away');why.push(side+' carries the higher effort/rest risk');}
  const risk=[];
  const rr=Math.max(Number(h.rest_rotation_risk||0),Number(a.rest_rotation_risk||0)); if(rr>=.25) risk.push('Meaningful effort/rest uncertainty ('+P(rr)+')');
  if(!g.market_bookmaker) risk.push('No current market benchmark');
  if(!risk.length) risk.push('Primary uncertainty is normal NFL game-to-game variance');
  return {read:b.read,value:b.value?.txt||'No material model-vs-market gap',why:why.join('. ')+'.',risk:risk.join('. ')+'.'};
}

function mlb(g){
  const hp=E(g.home_win_probability)?Number(g.home_win_probability):null,ap=hp!==null?1-hp:null;
  const conf=hp!==null?Math.max(hp,ap):null;
  const read=(g.predicted_winner||'Model winner')+(conf!==null?' '+P(conf):'')+(E(g.predicted_total_runs)?' · total '+N(g.predicted_total_runs):'');
  const val=marketValue(g);
  const h=g.home_competitive_context||{},a=g.away_competitive_context||{};
  const why=['Starting-pitcher, bullpen, park and recent team context drive the projection'];
  if(g.doubleheader) why.push('Doubleheader context is present');
  const rr=Math.max(Number(h.rest_rotation_risk||0),Number(a.rest_rotation_risk||0));
  const risk=[];if(rr>=.25)risk.push('Rotation/rest uncertainty '+P(rr));if(!g.market_bookmaker)risk.push('No current market benchmark');if(!risk.length)risk.push('Pitcher performance and bullpen usage remain the largest uncertainty');
  return {read,value:val?.txt||'No material model-vs-market gap',why:why.join('. ')+'.',risk:risk.join('. ')+'.'};
}

function wnba(g){
  const hp=E(g.model_home_win_probability)?Number(g.model_home_win_probability):E(g.home_win_probability)?Number(g.home_win_probability):null;
  const winner=hp===null?'Model':(hp>=.5?(g.home_abbr||'HOME'):(g.away_abbr||'AWAY'));
  const read=winner+(hp!==null?' '+P(Math.max(hp,1-hp)):'')+(E(g.model_predicted_margin)?' · margin '+(Number(g.model_predicted_margin)>=0?'+':'')+N(g.model_predicted_margin):'')+(E(g.model_predicted_total)?' · total '+N(g.model_predicted_total):'');
  const val=marketValue(g);
  const why=[];if(E(g.model_market_total_edge)&&Math.abs(Number(g.model_market_total_edge))>=6)why.push('Total is the strongest research-supported disagreement');else why.push('Independent model is compared with market only after prediction');
  const rr=Math.max(Number(g.home_rotation_rest_risk||0),Number(g.away_rotation_rest_risk||0));if(rr>=.2)why.push('Rotation/rest context is influencing confidence');
  const risk=[];if(rr>=.25)risk.push('Elevated rotation/rest uncertainty '+P(rr));if(!E(g.market_total)&&!E(g.market_home_spread))risk.push('Market benchmark unavailable');if(!risk.length)risk.push('Spread signals remain more exploratory than totals');
  return {read,value:val?.txt||'No material model-vs-market gap',why:why.join('. ')+'.',risk:risk.join('. ')+'.'};
}

function tennis(m){
  const p1=E(m.player_1_win_probability)?Number(m.player_1_win_probability):null,conf=E(m.winner_confidence)?Number(m.winner_confidence):(p1!==null?Math.max(p1,1-p1):null);
  const read=(m.predicted_winner||'Model winner')+(conf!==null?' '+P(conf):'')+(E(m.predicted_game_margin_player_1)?' · game margin '+N(m.predicted_game_margin_player_1):'')+(E(m.predicted_total_games)?' · total '+N(m.predicted_total_games):'');
  const vals=[];
  if(E(m.market_margin_player_1)&&E(m.predicted_game_margin_player_1)){const v=Number(m.predicted_game_margin_player_1)-Number(m.market_margin_player_1);vals.push({v,txt:(v>=0?'Player 1':'Player 2')+' spread · '+N(Math.abs(v))+' game gap'});}
  if(E(m.market_total_games)&&E(m.predicted_total_games)){const v=Number(m.predicted_total_games)-Number(m.market_total_games);vals.push({v,txt:(v>=0?'OVER ':'UNDER ')+N(m.market_total_games)+' · '+N(Math.abs(v))+' game gap'});}
  if(E(m.market_player_1_probability)&&p1!==null){const v=(p1-Number(m.market_player_1_probability))*100;vals.push({v,txt:(v>=0?m.player_1:m.player_2)+' ML · '+N(Math.abs(v),0)+' pp gap'});}
  vals.sort((a,b)=>Math.abs(b.v)-Math.abs(a.v));
  const why=['Surface/form/ranking and matchup features drive the pre-match forecast'];
  if(E(m.player_1_rank)&&E(m.player_2_rank))why.push('Ranking gap: #'+m.player_1_rank+' vs #'+m.player_2_rank);
  return {read,value:vals[0]?.txt||'No current market disagreement',why:why.join('. ')+'.',risk:'Tennis variance is concentrated in serve performance, fitness, and match-specific form; live score is not used to rewrite the pregame read.'};
}

function item(k,v,cls=''){return '<div class="model-read-item '+cls+'"><div class="model-read-k">'+esc(k)+'</div><div class="model-read-v">'+esc(v)+'</div></div>'}
window.modelRead=function(sport,obj){
  let x;switch(String(sport).toUpperCase()){case'NBA':x=nba(obj);break;case'NFL':x=nfl(obj);break;case'MLB':x=mlb(obj);break;case'WNBA':x=wnba(obj);break;default:x=tennis(obj);}
  return '<details class="model-read"><summary><span>Model Read</span><span style="color:var(--muted,#9aa7b8);font-weight:800">context + value</span></summary><div class="model-read-body">'+item('Read',x.read)+item('Value',x.value,'model-read-value')+item('Why',x.why)+item('Risk',x.risk)+'</div></details>';
};
})();