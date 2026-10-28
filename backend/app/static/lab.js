"use strict";
const $ = (id) => document.getElementById(id);
let protocols = {}, family = "constant_velocity", current = null, synthetic = false;
let videoUrl = null, videoHash = null, calibration = [], pausedTime = null, marked = [], videoGeneration = 0;
const fmt = (v) => Number(v).toLocaleString(undefined, {maximumSignificantDigits: 6});
function message(id, text, error = false) { $(id).textContent = text; $(id).classList.toggle("error", error); }
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail));
  return result;
}
function list(id, values) { $(id).replaceChildren(...values.map(v => {const li = document.createElement("li"); li.textContent = v; return li;})); }
function choose(key) {
  family = key; synthetic = false; current = null; marked = []; calibration = []; pausedTime = null; $("data").value = ""; $("confirmed").checked = false; $("result").hidden = true;
  $("title").value = "Trial 1"; $("condition").value = "";
  $("notes").value = ""; $("source-name").value = "";
  $("source").value = "imported_observations"; $("video-panel").hidden = true; $("sample-controls").hidden = true;
  const p = protocols[key];
  $("protocol-title").textContent = p.title; $("application").textContent = p.application; $("limits").textContent = p.limits;
  list("steps", p.steps); list("assumptions", p.assumptions);
  $("observations-label").textContent = `Time (s), ${p.input_label}`;
  $("angle-field").hidden = key !== "ramp_friction";
  document.querySelectorAll(".protocol").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.family === key)));
  message("status", "Choose a source and record your observations.");
}
function parseRows() {
  return $("data").value.trim().split(/\r?\n/).map((line, i) => {
    const parts = line.trim().split(/[,;\s]+/);
    if (parts.length !== 2 || parts.some(v => !v || !Number.isFinite(Number(v)))) throw new Error(`Row ${i+1}: enter exactly two numbers: time, value.`);
    return {time_s:Number(parts[0]), value:Number(parts[1])};
  });
}
async function calculate() {
  $("analyse").disabled = true; message("status", "Checking observations and computing both methods…");
  try {
    const request = {family, title:$("title").value, question:$("question").value, condition:$("condition").value,
      source:synthetic ? "synthetic_demo" : $("source").value, source_name:$("source-name").value,
      source_sha256:!synthetic && $("source").value === "manual_video" ? videoHash : null,
      notes:$("notes").value, assumptions_confirmed:$("confirmed").checked,
      angle_degrees:family === "ramp_friction" ? Number($("angle").value) : null, observations:parseRows()};
    current = await api("/api/lab/trials", request); show(current); await refresh(); message("status", "Trial saved with its observations and calculation receipt.");
  } catch(e) { message("status", e.message, true); } finally { $("analyse").disabled = false; }
}
function show(trial) {
  current = trial; $("result").hidden = false;
  $("badge").textContent = trial.input.source === "synthetic_demo" ? "SYNTHETIC EXAMPLE · NOT EXPERIMENTAL EVIDENCE" : trial.status.replaceAll("_", " ").toUpperCase();
  $("value").textContent = `${fmt(trial.analysis.final_value)} ${trial.analysis.unit}`;
  $("uncertainty").textContent = `Estimated fit variability: ± ${fmt(trial.analysis.final_uncertainty)} ${trial.analysis.unit}`;
  $("checks").replaceChildren(...Object.entries(trial.checks).map(([k,v]) => {const el = document.createElement("span"); el.textContent = `${k.replaceAll("_"," ")}: ${v}`; return el;}));
  $("reasons").textContent = trial.reason_codes.length ? `Review: ${trial.reason_codes.join(", ")}` : "Computational checks passed. Independent validation has not been performed.";
  $("methods").replaceChildren(...[trial.analysis.method_a, trial.analysis.method_b].map(m => {const el=document.createElement("p"); el.textContent=`${m.name}: ${fmt(m.value)} ± ${fmt(m.uncertainty)} ${m.unit}`;return el;}));
  list("warnings", trial.warnings); plot(trial.input.observations);
}
function plot(points) {
  const c=$("plot"), ctx=c.getContext("2d"), w=c.width,h=c.height,p=55;
  ctx.clearRect(0,0,w,h); const xs=points.map(v=>v.time_s),ys=points.map(v=>v.value);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const x=v=>p+(v-xmin)/(xmax-xmin||1)*(w-2*p),y=v=>h-p-(v-ymin)/(ymax-ymin||1)*(h-2*p);
  ctx.font="12px monospace";
  for(let i=0;i<=4;i++){const yy=p+i*(h-2*p)/4;ctx.strokeStyle="#d8ddd1";ctx.beginPath();ctx.moveTo(p,yy);ctx.lineTo(w-p,yy);ctx.stroke();ctx.fillStyle="#52615c";ctx.fillText(fmt(ymax-i*(ymax-ymin)/4),2,yy+4);}
  ctx.fillText(`${fmt(xmin)} s`,p,h-20);ctx.fillText(`${fmt(xmax)} s`,w-p-45,h-20);
  ctx.strokeStyle="#225d49";ctx.lineWidth=2;ctx.beginPath();points.forEach((pt,i)=>i?ctx.lineTo(x(pt.time_s),y(pt.value)):ctx.moveTo(x(pt.time_s),y(pt.value)));ctx.stroke();
  ctx.fillStyle="#bd572d";points.forEach(pt=>{ctx.beginPath();ctx.arc(x(pt.time_s),y(pt.value),3,0,Math.PI*2);ctx.fill();});
}
async function refresh() {
  const trials = await api("/api/lab/trials"); $("trials").replaceChildren(); $("baseline").replaceChildren(); $("candidate").replaceChildren();
  for(const trial of trials) {
    const tr=document.createElement("tr");
    for(const text of [`${trial.input.title} / ${trial.input.condition || "unchanged"}`, protocols[trial.input.family].title,`${fmt(trial.analysis.final_value)} ${trial.analysis.unit}`,trial.input.source === "synthetic_demo" ? "Synthetic example" : trial.status.replaceAll("_"," ")]){const td=document.createElement("td");td.textContent=text;tr.append(td);}
    const td=document.createElement("td"),b=document.createElement("button");b.textContent="Open";b.className="secondary";b.onclick=()=>api(`/api/lab/trials/${trial.id}`).then(show).catch(e=>message("status",e.message,true));td.append(b);tr.append(td);$("trials").append(tr);
    if(trial.status === "checks_passed") for(const id of ["baseline","candidate"]){const o=document.createElement("option");o.value=trial.id;o.textContent=`${trial.input.title} · ${trial.input.condition || trial.input.family}`;$(id).append(o);}
  }
  if(!trials.length){const tr=document.createElement("tr"),td=document.createElement("td");td.colSpan=5;td.textContent="Your first experiment starts here. Saved trials will appear in this notebook.";tr.append(td);$("trials").append(tr);}
}
function example() {
  synthetic=true;marked=[];$("source").value="imported_observations";$("video-panel").hidden=true;$("sample-controls").hidden=true;
  $("title").value=`Synthetic ${protocols[family].title}`;$("source-name").value="Generated mathematical example (not a filmed experiment)";
  $("notes").value="Synthetic ideal observations for exploring the workbench; no physical experiment or ground truth capture.";
  $("confirmed").checked=false;$("angle").value="30";
  const rows=Array.from({length:family === "oscillation" ? 121 : 41},(_,i)=>{
    const t=family === "oscillation" ? i/10 : family === "free_fall" ? i/100 : i/10;
    const v=family === "constant_velocity" ? .2*t : family === "free_fall" ? 1-.5*9.81*t*t : family === "oscillation" ? .1*Math.cos(Math.PI*t) : family === "exponential" ? 10*Math.exp(-.3*t) : .5*9.80665*(.5-.2*Math.cos(Math.PI/6))*t*t;
    return `${t.toFixed(3)}, ${v.toFixed(8)}`;
  });$("data").value=rows.join("\n");message("status","Synthetic example loaded. Results remain labelled synthetic, including if you edit these rows. Choose a bench again to start a real trial.");
}
function drawFrame() {
  const video=$("video"); if(!video.videoWidth || video.seeking) throw new Error("Wait for the selected video frame to finish loading.");
  video.pause();const c=$("frame");c.width=video.videoWidth;c.height=video.videoHeight;c.getContext("2d").drawImage(video,0,0);pausedTime=video.currentTime;
  const ctx=c.getContext("2d");ctx.fillStyle="#dceaa2";ctx.strokeStyle="#dceaa2";ctx.lineWidth=3;
  calibration.forEach(p=>{ctx.beginPath();ctx.arc(p.x,p.y,6,0,2*Math.PI);ctx.fill();});
  if(calibration.length===2){ctx.beginPath();ctx.moveTo(calibration[0].x,calibration[0].y);ctx.lineTo(calibration[1].x,calibration[1].y);ctx.stroke();}
  message("video-status",`Frame at playback ${video.currentTime.toFixed(4)} s. ${calibration.length<2 ? `Click reference end ${calibration.length+1}.` : "Click the target to add an observation."}`);
}
function resetMarks(){calibration=[];marked=[];pausedTime=null;$("data").value="";synthetic=false;message("video-status","Calibration reset. Capture a paused frame and select the two reference ends.");}
$("video-file").onchange=async()=>{
  const file=$("video-file").files[0], generation=++videoGeneration;videoHash=null;resetMarks();if(videoUrl)URL.revokeObjectURL(videoUrl);$("video").removeAttribute("src");if(!file)return;
  if(file.size>100*1024*1024){message("video-status","Use a clip smaller than 100 MB.",true);return;}
  videoUrl=URL.createObjectURL(file);$("video").src=videoUrl;$("source-name").value=file.name;
  try{const buffer=await file.arrayBuffer();const digest=await crypto.subtle.digest("SHA-256",buffer);if(generation!==videoGeneration)return;videoHash=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,"0")).join("");message("video-status","Video fingerprint recorded. Pause at a clear frame, then capture it to calibrate.");}catch(e){message("video-status","Cannot fingerprint this video. Use localhost or HTTPS and try again.",true);}
};
$("frame").onclick=e=>{
  try{
    if(pausedTime===null || $("video").seeking || Math.abs($("video").currentTime-pausedTime)>.0001 || !$("video").paused)throw new Error("Capture the paused frame before clicking.");
    const c=$("frame"),r=c.getBoundingClientRect(),point={x:(e.clientX-r.left)*c.width/r.width,y:(e.clientY-r.top)*c.height/r.height};
    if(calibration.length<2){if(calibration.length && Math.hypot(point.x-calibration[0].x,point.y-calibration[0].y)<10)throw new Error("Reference endpoints must be at least ten pixels apart.");calibration.push(point);drawFrame();return;}
    const length=Number($("length").value),factor=Number($("time-factor").value);if(!(length>0&&factor>0))throw new Error("Reference length and time factor must be positive.");
    const dx=calibration[1].x-calibration[0].x,dy=calibration[1].y-calibration[0].y;
    const time=pausedTime*factor;if(marked.some(p=>Math.abs(p.time_s-time)<1e-8))throw new Error("This timestamp already has an observation. Seek to another frame.");
    marked.push({time_s:time,value:((point.x-calibration[0].x)*dx+(point.y-calibration[0].y)*dy)/(dx*dx+dy*dy)*length});
    writeMarks();$("notes").value=`Manual video projection. Reference ${length} m, endpoints ${JSON.stringify(calibration)} in native pixels. Actual seconds/playback second: ${factor}. Fixed camera and coplanarity must be checked.`;
    const ctx=c.getContext("2d");ctx.fillStyle="#ff804e";ctx.beginPath();ctx.arc(point.x,point.y,6,0,Math.PI*2);ctx.fill();message("video-status",`${marked.length} observations marked. Move to the next time, capture, and click.`);
  }catch(e){message("video-status",e.message,true);}
};
function writeMarks(){$("data").value=[...marked].sort((a,b)=>a.time_s-b.time_s).map(p=>`${p.time_s.toFixed(8)}, ${p.value.toFixed(8)}`).join("\n");}
$("capture").onclick=()=>{try{drawFrame();}catch(e){message("video-status",e.message,true);}};
async function seekSample(next) {
  try {
    const video=$("video"), step=Number($("sample-step").value);
    const target=next ? Number((video.currentTime+step).toFixed(8)) : Number($("sample-time").value);
    if(!video.videoWidth || !Number.isFinite(target) || target<0 || target>=video.duration || (next && !(step>0))) throw new Error("Choose a valid sample time within the loaded video and a positive step.");
    video.pause();
    if(Math.abs(video.currentTime-target)>.000001) await new Promise((resolve,reject)=>{
      const timeout=setTimeout(()=>{video.removeEventListener("seeked",done);reject(new Error("Video seek timed out. Try a different time."));},5000);
      function done(){clearTimeout(timeout);resolve();} video.addEventListener("seeked",done,{once:true});video.currentTime=target;
    });
    $("sample-time").value=target.toFixed(6);drawFrame();
  } catch(e) {message("video-status",e.message,true);}
}
$("seek-sample").onclick=()=>seekSample(false);$("next-sample").onclick=()=>seekSample(true);
$("calibrate").onclick=resetMarks;$("length").onchange=resetMarks;$("time-factor").onchange=resetMarks;
$("undo").onclick=()=>{marked.pop();writeMarks();message("video-status",`${marked.length} observations remain.`);};
$("source").onchange=()=>{$("video-panel").hidden=$("source").value!=="manual_video";$("sample-controls").hidden=$("video-panel").hidden;if(synthetic){message("status","Choose a bench again to leave the synthetic example.");}};
$("analyse").onclick=calculate;$("example").onclick=example;
$("refresh").onclick=()=>refresh().catch(e=>message("status",e.message,true));
$("export").onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download=`phonelab-${current.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$("compare").onclick=async()=>{try{const r=await api("/api/lab/compare",{baseline_id:$("baseline").value,candidate_id:$("candidate").value});message("comparison",`Changed condition − baseline: ${fmt(r.difference)} ${r.unit}${r.relative_change_pct===null?"":` (${fmt(r.relative_change_pct)}%)`}. ${r.interpretation}`);}catch(e){message("comparison",e.message,true);}};
$("plan").onclick=async()=>{$("plan").disabled=true;message("plan-status","Muse Spark is checking the supported protocols…");try{const r=await api("/api/lab/plan",{question:$("question").value});if(r.protocol)choose(r.family);message("plan-status",`${r.message} Estimated model cost: $${r.estimated_cost_usd.toFixed(5)}.`);}catch(e){message("plan-status",e.message,true);}finally{$("plan").disabled=false;}};
$("auto-run").onclick=async()=>{const file=$("auto-video").files[0];if(!file){message("auto-status","Choose a clip first.",true);return;}$("auto-run").disabled=true;message("auto-status","Running capture checks, provider tracking and analysis. This may take several minutes…");$("auto-result").textContent="";try{const data=new FormData();data.append("video",file);data.append("object_prompt",$("object").value);data.append("real_world_length_m",$("auto-scale").value);const r=await fetch("/api/measure/free-fall",{method:"POST",body:data});const result=await r.json();if(!r.ok)throw new Error(JSON.stringify(result.detail));$("auto-result").textContent=JSON.stringify(result,null,2);message("auto-status","Provider card returned. Its legacy certificate reflects computational gates, not independent real-world validation.");}catch(e){message("auto-status",e.message,true);}finally{$("auto-run").disabled=false;}};
api("/api/lab/protocols").then(async r=>{protocols=r.protocols;let i=0;for(const[key,p]of Object.entries(protocols)){const b=document.createElement("button");b.className="protocol";b.dataset.family=key;const small=document.createElement("small");small.textContent=`BENCH 0${++i} / ${p.unit}`;const strong=document.createElement("strong");strong.textContent=p.title;b.append(small,strong);b.onclick=()=>choose(key);$("protocols").append(b);}choose(family);await refresh();}).catch(e=>message("status",`Could not load the bench: ${e.message}`,true));
