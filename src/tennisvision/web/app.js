'use strict';
const $ = id => document.getElementById(id);
const video = $('video'), canvas = $('overlay'), ctx = canvas.getContext('2d');
let job = null, config = null, result = null, events = [], labels = [], mode = null, cornerIndex = 0, busy = false, pollTimer = null, health = {};
let court = [{x:.32,y:.22},{x:.68,y:.22},{x:.90,y:.90},{x:.10,y:.90}];
let activeStep = 'setup', scoreRows = [], scoreFinal = null, motion = [], previewTimer, drawingBall = null;
const cornerNames = ['Far left','Far right','Near right','Near left'];
const clock = t => `${String(Math.floor(t/60)).padStart(2,'0')}:${(t%60).toFixed(2).padStart(5,'0')}`;
const url = name => `/api/matches/${job.id}/files/${name}`;
function showError(error) { $('alert').textContent = error.message || String(error); $('alert').hidden = false; }
function clearError() { $('alert').hidden = true; }
function toast(message) { $('toast').textContent = message; $('toast').hidden = false; setTimeout(() => $('toast').hidden = true, 3500); }
async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response.'); }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(Array.isArray(detail) ? detail.map(e => `${e.loc.slice(1).join('.')}: ${e.msg}`).join('; ') : detail || 'Request failed.');
  }
  return data;
}
const jsonOptions = (method, body) => ({method, headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
function setStep(step) {
  activeStep = step;
  for (const name of ['setup','review','research']) $(name+'-panel').hidden = name !== step;
  document.querySelectorAll('[data-step]').forEach(button => { button.classList.toggle('active',button.dataset.step === step); button.setAttribute('aria-current',button.dataset.step === step ? 'step' : 'false'); });
  if (step !== 'research' && mode === 'label') setMode(null);
  if (step !== 'setup' && mode === 'court') setMode(null);
}
document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click',() => setStep(button.dataset.step)));
function setBusy(value) {
  busy = value;
  $('progress-panel').hidden = !value;
  document.querySelectorAll('#setup-panel input,#setup-panel select,#setup-panel button').forEach(el => el.disabled = value || !job);
  for (const name of ['replace-video','source-view','seek-frame','label-mode','previous-frame','next-frame','remove-label','ball-hidden','label-coordinates','save-labels']) $(name).disabled = value || !job;
  for (const name of ['add-a','add-b','save-review']) $(name).disabled = value || !result;
  $('analysis-view').disabled = value || !result;
  $('analyze').disabled = value || !job || !health.models?.pose;
  $('cancel-job').disabled = false;
  document.querySelectorAll('#events input,#events select,#events button').forEach(el => el.disabled = value);
}
function updateNames() {
  const names = [$('player-a').value.trim() || 'Player A',$('player-b').value.trim() || 'Player B'];
  $('score-name-a').textContent = names[0]; $('score-name-b').textContent = names[1];
  $('near-player').options[0].textContent = names[0]; $('near-player').options[1].textContent = names[1];
}
$('player-a').addEventListener('input',updateNames); $('player-b').addEventListener('input',updateNames);
function fieldCoordinates() {
  $('corner-fields').replaceChildren();
  court.forEach((point,index) => {
    const group = document.createElement('div'), caption = document.createElement('label'), pair = document.createElement('div');
    caption.textContent = `${index+1}. ${cornerNames[index]}`; pair.className = 'pair';
    ['x','y'].forEach(axis => {
      const input = document.createElement('input'); input.type='number'; input.min=0;input.max=1;input.step=.001;input.value=point[axis].toFixed(3);input.setAttribute('aria-label',`${cornerNames[index]} ${axis.toUpperCase()}`);
      input.addEventListener('change',() => { const v = Number(input.value); if(Number.isFinite(v) && v>=0 && v<=1){ court[index][axis]=v; drawOverlay(); } }); pair.append(input);
    });group.append(caption,pair);$('corner-fields').append(group);
  });
}
function imageRect() {
  const width = canvas.width, height = canvas.height;
  const ratio = job ? job.info.width/job.info.height : 16/9;
  const rw = Math.min(width,height*ratio), rh=rw/ratio;
  return {x:(width-rw)/2,y:(height-rh)/2,w:rw,h:rh};
}
function drawOverlay() {
  const bounds = $('dropzone').getBoundingClientRect();
  canvas.width = Math.round(bounds.width*devicePixelRatio); canvas.height = Math.round(bounds.height*devicePixelRatio);
  const r=imageRect(), p=point => [r.x+point.x*r.w,r.y+point.y*r.h];
  if (mode === 'court') {
    ctx.strokeStyle='#d6f58a';ctx.lineWidth=2*devicePixelRatio;ctx.fillStyle='#d6f58a16';ctx.beginPath();
    court.forEach((point,i) => { const [x,y]=p(point); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); });ctx.closePath();ctx.fill();ctx.stroke();
    court.forEach((point,i) => {const [x,y]=p(point);ctx.fillStyle='#d6f58a';ctx.beginPath();ctx.arc(x,y,11*devicePixelRatio,0,Math.PI*2);ctx.fill();ctx.fillStyle='#152012';ctx.font=`bold ${11*devicePixelRatio}px system-ui`;ctx.textAlign='center';ctx.fillText(i+1,x,y+4*devicePixelRatio);});
  }
  if(mode==='label' && drawingBall) {const [x,y]=p(drawingBall);ctx.strokeStyle='#d6f58a';ctx.lineWidth=2*devicePixelRatio;ctx.beginPath();ctx.arc(x,y,9*devicePixelRatio,0,Math.PI*2);ctx.moveTo(x-16*devicePixelRatio,y);ctx.lineTo(x+16*devicePixelRatio,y);ctx.moveTo(x,y-16*devicePixelRatio);ctx.lineTo(x,y+16*devicePixelRatio);ctx.stroke();}
}
new ResizeObserver(drawOverlay).observe($('dropzone'));
function setMode(value) {
  mode=value;canvas.hidden=!value;canvas.classList.toggle('interactive',!!value);$('canvas-hint').hidden=!value;
  $('calibrate').textContent = value==='court'?'Finish calibration':'Mark court corners';
  $('label-mode').textContent = value==='label'?'Finish labeling':'Label ball on video';
  if(value) {showSource();video.pause();video.controls=false;}
  else video.controls=true;
  if(value==='court') $('canvas-hint').textContent=`Click corner ${cornerIndex+1}: ${cornerNames[cornerIndex]}. Or edit the coordinate fields below.`;
  if(value==='label') $('canvas-hint').textContent='Pause on the ball, then click its center. Use frame buttons below to move precisely.';
  drawOverlay();
}
$('calibrate').addEventListener('click',() => {cornerIndex=0;setMode(mode==='court'?null:'court');});
$('label-mode').addEventListener('click',() => setMode(mode==='label'?null:'label'));
canvas.addEventListener('click',event => {
  const bounds=canvas.getBoundingClientRect(),r=imageRect();
  const point={x:((event.clientX-bounds.left)*canvas.width/bounds.width-r.x)/r.w,y:((event.clientY-bounds.top)*canvas.height/bounds.height-r.y)/r.h};
  if(point.x<0||point.x>1||point.y<0||point.y>1)return;
  if(mode==='court') {court[cornerIndex]=point;cornerIndex++;fieldCoordinates();if(cornerIndex===4){$('corner-status').textContent='Four corners marked. Refine below if needed.';setMode(null);}else $('canvas-hint').textContent=`Click corner ${cornerIndex+1}: ${cornerNames[cornerIndex]}`;drawOverlay();}
  else if(mode==='label') saveLabel(true,point);
});
function showSource(seekTo=null) {
  if(!job)return;
  if(video.dataset.view!=='source'){const t=typeof seekTo==='number'?seekTo:video.currentTime;video.src=url('source.mp4');video.dataset.view='source';video.addEventListener('loadedmetadata',()=>{if(video.dataset.view==='source')video.currentTime=Math.min(t,video.duration||t);},{once:true});}else if(typeof seekTo==='number'){video.currentTime=Math.min(seekTo,video.duration||seekTo);}
  $('source-view').classList.add('selected');$('analysis-view').classList.remove('selected');
}
function showAnalysis() {if(!result)return;setMode(null);const t=video.currentTime;video.src=url('annotated.mp4')+'?v='+Date.now();video.dataset.view='analysis';video.addEventListener('loadedmetadata',()=>video.currentTime=Math.min(t,video.duration||t),{once:true});$('source-view').classList.remove('selected');$('analysis-view').classList.add('selected');}
$('source-view').addEventListener('click',()=>{setMode(null);showSource();});$('analysis-view').addEventListener('click',showAnalysis);
async function upload(file) {
  if(!file||busy)return;
  clearError();if(file.size>512*1024*1024){showError(new Error('Choose a video smaller than 512 MB.'));return;}
  const data=new FormData();data.append('video',file);$('clip-name').textContent='Uploading video…';$('video-file').disabled=true;
  try {job=await api('/api/matches',{method:'POST',body:data});result=null;events=[];labels=[];motion=[];scoreRows=[];scoreFinal=null;localStorage.setItem('tennisvision-match',job.id);attachVideo();setStep('setup');$('corner-status').textContent='Mark the actual court before analyzing.';renderEvents();drawCourt();resetSignals();}
  catch(error){showError(error);}finally{$('video-file').disabled=false;setBusy(false);}
}
function attachVideo(){ $('empty-video').hidden=true;video.hidden=false;video.src=url('source.mp4');video.poster=url('poster.jpg');video.dataset.view='source';video.controls=true;$('clip-name').textContent=job.filename;$('video-meta').textContent=`${job.info.width} × ${job.info.height} · ${job.info.duration.toFixed(1)} SEC`;$('video-title').textContent='MATCH FOOTAGE';setBusy(false);fieldCoordinates();}
$('choose-video').addEventListener('click',()=>$('video-file').click());
$('video-file').addEventListener('change',event=>upload(event.target.files[0]));$('replace-video').addEventListener('click',()=>$('video-file').click());
for(const type of ['dragenter','dragover'])$('dropzone').addEventListener(type,event=>{event.preventDefault();if(!busy)$('dropzone').classList.add('dragging');});
for(const type of ['dragleave','drop'])$('dropzone').addEventListener(type,event=>{event.preventDefault();$('dropzone').classList.remove('dragging');if(type==='drop')upload(event.dataTransfer.files[0]);});
function getConfig(){const tb=$('tiebreak').checked;return {players:[$('player-a').value.trim(),$('player-b').value.trim()],near_player:Number($('near-player').value),court,detector:$('detector').value,inference_fps:Number($('inference-fps').value),pose_fps:6,lost_seconds:1.8,initial:{points:tb?[Number($('tb-a').value),Number($('tb-b').value)]:[Number($('points-a').value),Number($('points-b').value)],games:[Number($('games-a').value),Number($('games-b').value)],sets:[Number($('sets-a').value),Number($('sets-b').value)],best_of:Number($('best-of').value),tiebreak:tb,tiebreak_target:Number($('tiebreak-target').value)}};}
$('tiebreak').addEventListener('change',()=>{$('tb-inputs').hidden=!$('tiebreak').checked;});
$('analyze').addEventListener('click',async()=>{if(!job)return;clearError();setMode(null);config=getConfig();try{if(result){job=await api(`/api/matches/${job.id}/fork`,{method:'POST'});localStorage.setItem('tennisvision-match',job.id);}await api(`/api/matches/${job.id}/analyze`,jsonOptions('POST',config));setBusy(true);result=null;events=[];scoreRows=[];motion=[];resetSignals();updateScore();drawCourt();poll();}catch(error){showError(error);}});
$('cancel-job').addEventListener('click',async()=>{try{await api(`/api/matches/${job.id}/cancel`,{method:'POST'});$('cancel-job').disabled=true;$('progress-message').textContent='Cancellation requested…';}catch(error){showError(error);}});
async function poll(){clearTimeout(pollTimer);try{job=await api(`/api/matches/${job.id}`);$('progress').value=job.progress;$('progress-number').textContent=`${Math.round(job.progress*100)}%`;$('progress-message').textContent=job.message;if(job.state==='complete'){result=job.result;events=result.events;config=await api(url('config.json'));court=config.court;applyConfig();scoreRows=result.scores.timeline;scoreFinal=result.scores.estimated;setBusy(false);renderResult();return;}if(['failed','cancelled'].includes(job.state)){setBusy(false);if(job.error)showError(new Error(job.error));else toast(job.message);return;}}catch(error){showError(error);}pollTimer=setTimeout(poll,1500);}
function applyConfig(){ for(const [i,letter] of ['a','b'].entries()) { $('sets-'+letter).value=config.initial.sets[i];$('games-'+letter).value=config.initial.games[i];$('points-'+letter).value=config.initial.tiebreak?0:config.initial.points[i];$('tb-'+letter).value=config.initial.tiebreak?config.initial.points[i]:0; } $('tiebreak').checked=config.initial.tiebreak;$('tb-inputs').hidden=!config.initial.tiebreak;$('tiebreak-target').value=config.initial.tiebreak_target;$('best-of').value=config.initial.best_of;$('inference-fps').value=config.inference_fps; $('player-a').value=config.players[0];$('player-b').value=config.players[1];$('detector').value=config.detector;$('near-player').value=config.near_player;updateNames();fieldCoordinates(); }
function resetSignals(){for(const id of ['ball-observations','pose-observations','hit-count','rally-count'])$(id).textContent='—';for(const id of ['download-video','download-csv','download-report']){$(id).classList.add('disabled');$(id).removeAttribute('href');$(id).setAttribute('aria-disabled','true');}}
async function renderResult(){const a=result.analysis;$('ball-observations').textContent=`${a.ball_observations} / ${a.ball_samples}`;$('pose-observations').textContent=`${a.pose_observations.near} / ${a.pose_observations.far}`;$('hit-count').textContent=a.hits.length;$('rally-count').textContent=events.length;$('research-stats').textContent=`${a.detector} · ${a.processed_frames} decoded frames · ${a.camera_cuts} detected camera cuts · ${a.seconds.toFixed(1)} seconds to analyze.`;
  for(const [id,name] of [['download-video','annotated.mp4'],['download-csv','points.csv'],['download-report','report.zip']]){const link=$(id);link.href=url(name)+'?download=true';link.classList.remove('disabled');link.setAttribute('aria-disabled','false');}
  renderEvents();updateScore();showAnalysis();setStep('review');
  try {const text=await (await fetch(url('motion.csv'))).text();const rows=text.trim().split(/\r?\n/);const keys=rows.shift().split(',');motion=rows.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[keys[i],v])));drawCourt();}catch(error){console.warn('Ground traces unavailable',error);}
}
function renderEvents(){const container=$('events');container.replaceChildren();$('review-count').textContent=`${events.length} EVENTS`;
  if(!events.length){const note=document.createElement('p');note.className='empty-events';note.textContent=result?'No completed rally could be inferred. Tracking gaps or a clip ending during play are not points. Use the point buttons while replaying to add confirmed events.':'Analyze your footage to review rally hypotheses.';container.append(note);return;}
  [...events].sort((a,b)=>a.time-b.time).forEach(event=>{const row=document.createElement('article');row.className='event';const top=document.createElement('div');top.className='event-top';const time=document.createElement('button');time.className='event-time';time.textContent=clock(event.time)+' ↗';time.setAttribute('aria-label','Replay rally ending at '+clock(event.time));time.addEventListener('click',()=>{video.currentTime=Math.max(0,event.start-.7);video.play().catch(()=>{});});const tag=document.createElement('span');tag.className='event-status';tag.textContent=`${event.source} · ${event.status}`;top.append(time,tag);
    const reason=document.createElement('p');reason.className='event-reason';reason.textContent=event.reason;
    const fields=document.createElement('div');fields.className='event-fields';
    function select(label,values,current,callback){const wrap=document.createElement('label');wrap.textContent=label;const input=document.createElement('select');for(const [value,text] of values){const option=new Option(text,value);input.append(option);}input.value=current;input.addEventListener('change',()=>{callback(input.value);tag.textContent=`${event.source} · ${event.status}`;previewScores();});wrap.append(input);fields.append(wrap);}
    select('Point winner',[['','Unresolved'],['0',config?.players[0]||'Player A'],['1',config?.players[1]||'Player B']],event.winner===null?'':String(event.winner),value=>event.winner=value===''?null:Number(value));
    select('Review decision',[['pending','Needs review'],['confirmed','Confirmed'],['rejected','Reject event']],event.status,value=>event.status=value);
    const remove=document.createElement('button');remove.className='text-button';remove.textContent='Remove';remove.addEventListener('click',()=>{events=events.filter(e=>e.id!==event.id);renderEvents();previewScores();});fields.append(remove);
    const evidence=document.createElement('small');evidence.textContent=event.source==='vision'?`${event.hits} contact cues · evidence index ${event.evidence_score.toFixed(2)} (not a probability)`:'Added by reviewer';row.append(top,reason,fields,evidence);container.append(row);
  });
}
function previewScores(){clearTimeout(previewTimer);previewTimer=setTimeout(async()=>{try{const data=await api(`/api/matches/${job.id}/preview-score`,jsonOptions('POST',{events}));scoreRows=data.timeline;scoreFinal=data.estimated;result.scores={...result.scores,...data};updateScore();$('review-note').textContent='Unsaved changes. Save review to update video and exports.';}catch(error){showError(error);}},200);}
function updateScore(){const time=video.currentTime||0;let score={points:config?.initial.tiebreak?config.initial.points.map(String):(config?.initial.points||[0,0]).map(p=>['0','15','30','40','AD'][p]),games:config?.initial.games||[0,0],sets:config?.initial.sets||[0,0]};for(const row of scoreRows){if(row.time<=time)score=row.score;else break;}
  for(const [i,letter] of ['a','b'].entries()){for(const key of ['points','games','sets'])$('score-'+key+'-'+letter).textContent=score[key][i];}
  const confirmed=result?.scores.confirmed?.total_points||[0,0];$('confirmed-points').textContent=`A ${confirmed[0]} · B ${confirmed[1]}`;
  const pending=events.some(e=>e.status==='pending');$('score-badge').textContent=!events.length?'NO EVENTS':pending?'ESTIMATED':'REVIEWED';$('score-badge').classList.toggle('warning',pending);
  $('current-time').textContent=clock(time);
}
video.addEventListener('timeupdate',()=>{updateScore();if(mode==='label'){const frame=Math.min(job.info.frames-1,Math.round(video.currentTime*job.info.fps));drawingBall=labels.find(l=>l.frame===frame&&l.visible)||null;drawOverlay();}});
function addPoint(winner){if(!result)return;const time=Math.min(job.info.duration,video.currentTime);events.push({id:crypto.randomUUID(),time,winner,status:'confirmed',source:'manual',evidence_score:1,reason:'Winner entered by the reviewer.',start:Math.max(0,time-3),hits:0});renderEvents();previewScores();}
$('add-a').addEventListener('click',()=>addPoint(0));$('add-b').addEventListener('click',()=>addPoint(1));
$('save-review').addEventListener('click',async()=>{clearError();try{if(events.some(e=>e.status==='confirmed'&&e.winner===null))throw new Error('Choose a winner for every confirmed point, or reject the event.');await api(`/api/matches/${job.id}/review`,jsonOptions('PUT',{events}));setMode(null);setBusy(true);poll();}catch(error){showError(error);}});
function saveLabel(visible,point=null){if(!job)return;video.pause();const frame=Math.min(job.info.frames-1,Math.max(0,Math.round(video.currentTime*job.info.fps)));const label={frame,visible,x:visible?point.x:null,y:visible?point.y:null};labels=labels.filter(l=>l.frame!==frame);labels.push(label);drawingBall=visible?point:null;$('label-count').textContent=`${labels.length} labeled frames · current frame ${frame}`;if(visible){$('ball-x').value=point.x.toFixed(4);$('ball-y').value=point.y.toFixed(4);}drawOverlay();}
$('seek-frame').addEventListener('click',()=>{if(!job)return;const frame=Number($('frame-index').value);if(!Number.isInteger(frame)||frame<0||frame>=job.info.frames){showError(new Error('Choose a frame inside the source video.'));return;}showSource(frame/job.info.fps);video.pause();});
$('remove-label').addEventListener('click',()=>{if(!job)return;const frame=Math.min(job.info.frames-1,Math.round(video.currentTime*job.info.fps));labels=labels.filter(l=>l.frame!==frame);drawingBall=null;$('label-count').textContent=`${labels.length} labeled frames`;drawOverlay();});
$('ball-hidden').addEventListener('click',()=>saveLabel(false));$('label-coordinates').addEventListener('click',()=>{const x=Number($('ball-x').value),y=Number($('ball-y').value);if(!$('ball-x').value||!$('ball-y').value||x<0||x>1||y<0||y>1){showError(new Error('Enter ball coordinates between 0 and 1.'));return;}saveLabel(true,{x,y});});
for(const [name,delta] of [['previous-frame',-1],['next-frame',1]])$(name).addEventListener('click',()=>{if(!job)return;const target=Math.max(0,Math.min((job.info.frames-1)/job.info.fps,(Math.round(video.currentTime*job.info.fps)+delta)/job.info.fps));showSource(target);video.pause();});
$('save-labels').addEventListener('click',async()=>{try{await api(`/api/matches/${job.id}/labels`,jsonOptions('PUT',labels));const link=document.createElement('a');link.href=url('labels.json')+'?download=true';link.download='labels.json';link.click();toast(`${labels.length} labels exported.`);}catch(error){showError(error);}});
function drawCourt(){const c=$('court-map'),x=c.getContext('2d'),w=c.width,h=c.height;x.clearRect(0,0,w,h);const left=37,top=24,cw=w-74,ch=h-48;x.strokeStyle='#647d57';x.lineWidth=1.2;x.strokeRect(left,top,cw,ch);x.beginPath();x.moveTo(left,top+ch/2);x.lineTo(left+cw,top+ch/2);x.moveTo(left,top+ch*.27);x.lineTo(left+cw,top+ch*.27);x.moveTo(left,top+ch*.73);x.lineTo(left+cw,top+ch*.73);x.moveTo(w/2,top+ch*.27);x.lineTo(w/2,top+ch*.73);x.stroke();
  for(const slot of ['near','far']){const rows=motion.filter(r=>r.slot===slot);x.strokeStyle=slot==='near'?'#b0f6c2':'#79bafa';x.lineWidth=1.5;x.globalAlpha=.7;let previous=null;for(const row of rows){const px=left+Number(row.x_m)/8.23*cw,py=top+Number(row.y_m)/23.77*ch;if(px<4||px>w-4||py<4||py>h-4){previous=null;continue;}if(previous&&Number(row.time)-previous.time<.5){x.beginPath();x.moveTo(previous.x,previous.y);x.lineTo(px,py);x.stroke();}previous={x:px,y:py,time:Number(row.time)};}x.globalAlpha=1;}
}
async function initialize(){fieldCoordinates();drawCourt();setBusy(false);try{health=await api('/api/health');const ready=health.models.tracknet&&health.torch;$('detector').options[0].disabled=!ready;if(!ready)$('detector').value='classical';$('model-status').textContent=!health.models.pose?'Pose model missing. Follow model setup in the repository README.':ready?'TrackNet uses three consecutive frames. CPU processing can take several seconds per sample.':'Fast baseline available. Install TrackNet assets to compare the temporal model.';
  const id=localStorage.getItem('tennisvision-match');if(id&&/^[a-f0-9]{32}$/.test(id)){try{job=await api('/api/matches/'+id);attachVideo();if(job.state==='running'){setBusy(true);poll();}else if(job.state==='complete'){result=job.result;events=result.events;config=await api(url('config.json'));court=config.court;applyConfig();scoreRows=result.scores.timeline;scoreFinal=result.scores.estimated;setBusy(false);renderResult();}try{const existing=await api(url('labels.json'));labels=existing.labels;$('label-count').textContent=`${labels.length} labeled frames`;}catch{}}catch{localStorage.removeItem('tennisvision-match');job=null;}}
}catch(error){showError(error);}setBusy(busy);}
initialize();
