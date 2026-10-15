"""Interactive HTML: 3D trajectory + 2D timelines with time-range selection and move-segment marks."""
import base64, csv, json, math, os, subprocess
import numpy as np
from .flow import hwaccel_default

CAT = {'push': 'push', 'pull': 'pull', 'orbit': 'orbit', 'truck': 'truck', 'vertical': 'vertical',
       'static': 'static', 'pan': 'pan', 'tilt': 'pan', 'diagonal': 'truck'}

THUMB_W, THUMB_H, THUMB_COLS = 160, 90, 10

def thumb_sprite(video, duration, out_jpg, max_thumbs=200, hwaccel='auto'):
    """One JPEG sprite of key-frame thumbnails every `step` seconds (cached). Returns (step, count, rows)."""
    step = max(1, math.ceil(duration / max_thumbs)); count = int(duration // step) + 1; rows = math.ceil(count / THUMB_COLS)
    if hwaccel == 'auto': hwaccel = hwaccel_default()
    elif hwaccel == 'none': hwaccel = None
    if not os.path.exists(out_jpg):
        cmd = ['ffmpeg', '-v', 'error', '-y'] + (['-hwaccel', hwaccel] if hwaccel else []) + \
              ['-skip_frame', 'nokey', '-i', video,
               '-vf', f'fps=1/{step}:round=up,scale={THUMB_W}:{THUMB_H},tile={THUMB_COLS}x{rows}', '-frames:v', '1', '-q:v', '6', out_jpg]
        subprocess.run(cmd, check=True)
    return step, count, rows

def build(video, tel_csv, flow_csv, seg_json, out_html, hz=10.0, hwaccel='auto', analysis=None):
    from .segment import features, classify_frames, aligned_tel, annotate_playback_speeds, playback_speed_text
    with open(seg_json, encoding='utf-8') as fh:
        d = json.load(fh)
    fps = d['fps']; thr = d['thresholds']; segs = d['segments']
    with open(tel_csv, encoding='utf-8', newline='') as fh:
        tel = list(csv.DictReader(fh))
    with open(flow_csv, encoding='utf-8', newline='') as fh:
        flow = list(csv.DictReader(fh))
    from .segment import DEFAULT_THR
    thr = {**DEFAULT_THR, **thr}
    if analysis is None:
        f, n = features(tel, flow, fps, rev_amp=thr['motion_rev_amp'])
        keep, labels = classify_frames(f, thr, fps)
    else:
        f, keep, labels = analysis
        n = len(keep)
    # Legacy catalog JSON has no speed assessment. Compute it from these same
    # cached feature arrays rather than interpreting missing data as unlimited.
    annotate_playback_speeds(segs, f, fps, output_fps=None)
    A, _ = aligned_tel(tel, n, fps)
    lat, lon = A['lat'], A['lon']
    R = 6371000.0; lat0 = math.radians(lat.mean())
    north = np.radians(lat - lat.mean()) * R; east = np.radians(lon - lon.mean()) * R * math.cos(lat0)
    step = max(1, int(round(fps / hz))); idx = np.arange(0, n, step)
    cam_yaw = A['ac_yaw']; gp = A['g_pitch']
    r2 = lambda a: [None if (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 2) for x in a]
    data = dict(
        t=r2(idx / fps), e=r2(east[idx]), n=r2(north[idx]), u=r2(f['rel_alt'][idx]),
        sp=r2(f['speed_h'][idx]), cl=r2(-f['vz'][idx]), yr=r2(f['yaw_rate'][idx]), pr=r2(f['pitch_rate'][idx]),
        jit=r2(f['jitter_rms'][idx]), mpx=r2(f['motion_px'][idx]), macc=r2(f['motion_acc'][idx]), cons=r2(f['consist'][idx]),
        keep=[int(k) for k in keep[idx]], lab=[CAT.get(str(l).partition('+')[0], 'truck') for l in labels[idx]],
        cy=r2(cam_yaw[idx]), gp=r2(gp[idx]))
    base = os.path.splitext(os.path.basename(video))[0]
    sprite = os.path.join(os.path.dirname(out_html), base + '_thumbs.jpg')
    try:
        step, count, rows = thumb_sprite(video, n / fps, sprite, hwaccel=hwaccel)
        thumbs = dict(step=step, count=count, rows=rows, cols=THUMB_COLS, w=THUMB_W, h=THUMB_H,
                      uri='data:image/jpeg;base64,' + base64.b64encode(open(sprite, 'rb').read()).decode())
    except Exception as e:
        print('thumbnail sprite failed:', e); thumbs = None
    meta = dict(name=base, fps=fps, duration=n / fps, thumbs=thumbs,
                kept=float(keep.mean()), thr={k: thr[k] for k in ('jitter_px', 'motion_acc', 'yaw_acc', 'acc_h')},
                segs=[dict(start=round(s['start'], 2), end=round(s['end'], 2), label=s['label'], moves=s.get('moves', [s['label']]),
                           duration=round(s['duration'], 1), base=s.get('base', ''), speed=round(s['speed_mps'], 1),
                           yaw=round(s['yaw_rate_dps'], 1), radius=s.get('radius_m'),
                           max_playback_speed=s['max_playback_speed'],
                           playback_speed_profile=s['playback_speed_profile'],
                           speed_output_fps=s['playback_speed_profile'].get('output_fps', 30),
                           speed_limit_label=playback_speed_text(s['playback_speed_profile'])) for s in segs])
    html = TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':'))).replace('__META__', json.dumps(meta, ensure_ascii=False))
    html = html.replace('__TITLE__', meta['name'] + ' 轨迹')
    with open(out_html, 'w', encoding='utf-8') as fh: fh.write(html)
    return out_html

TEMPLATE = r'''<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* layout: one column; 3D view on top, control strip, stacked shared-x timelines below */
:root{--bg:#f7f6f2;--panel:#ffffff;--fg:#1b1a17;--fg2:#5d5b55;--muted:#9a9890;--line:#e2e0d8;--grid:#ebe9e2;
 --accent:#2a78d6;--range:rgba(42,120,214,.10);--drop:rgba(227,73,72,.10);
 --c-push:#2a78d6;--c-pull:#eb6834;--c-orbit:#1baf7a;--c-truck:#eda100;--c-vertical:#e87ba4;--c-static:#7a7873;--c-pan:#4a3aa7;--ghost:#c9c7bf;
 --font:"IBM Plex Sans",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15161a;--panel:#1d1e23;--fg:#ececea;--fg2:#b9b8b2;--muted:#7d7c76;--line:#2d2e34;--grid:#26272c;
 --accent:#3987e5;--range:rgba(57,135,229,.16);--drop:rgba(230,103,103,.14);
 --c-push:#3987e5;--c-pull:#d95926;--c-orbit:#199e70;--c-truck:#c98500;--c-vertical:#d55181;--c-static:#8d8c86;--c-pan:#9085e9;--ghost:#3a3b42;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15161a;--panel:#1d1e23;--fg:#ececea;--fg2:#b9b8b2;--muted:#7d7c76;--line:#2d2e34;--grid:#26272c;
 --accent:#3987e5;--range:rgba(57,135,229,.16);--drop:rgba(230,103,103,.14);
 --c-push:#3987e5;--c-pull:#d95926;--c-orbit:#199e70;--c-truck:#c98500;--c-vertical:#d55181;--c-static:#8d8c86;--c-pan:#9085e9;--ghost:#3a3b42;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:14px/1.5 var(--font);margin:0}
.wrap{max-width:1400px;margin:0 auto;padding-block:16px 32px;padding-inline:16px;display:flex;flex-direction:column;gap:14px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 20px}
h1{font-size:18px;font-weight:600;margin:0;font-family:var(--mono)}
.stat{color:var(--fg2);font-variant-numeric:tabular-nums}
.stat b{color:var(--fg);font-weight:500}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;min-width:0}
#p3d{height:56vh;min-height:380px}
.ctl{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:10px 12px}
.ctl label{color:var(--fg2);font-size:12px;letter-spacing:.04em;text-transform:uppercase}
.ctl input[type=number]{width:84px;font:13px var(--mono);padding:4px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg)}
.ctl button{font:13px var(--font);padding:5px 10px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg);cursor:pointer}
.ctl button:hover{border-color:var(--accent)}
.ctl button:focus-visible,.chip:focus-visible,.ctl input:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
.slider{position:relative;flex:1 1 260px;height:28px;min-width:0}
.slider input[type=range]{position:absolute;left:0;right:0;top:0;width:100%;height:28px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
.slider input[type=range]::-webkit-slider-thumb{pointer-events:auto;-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);border:2px solid var(--panel);cursor:ew-resize}
.slider input[type=range]::-moz-range-thumb{pointer-events:auto;width:14px;height:14px;border-radius:50%;background:var(--accent);border:2px solid var(--panel);cursor:ew-resize}
.slider .track{position:absolute;left:8px;right:8px;top:12px;height:4px;background:var(--line);border-radius:2px}
.slider .fill{position:absolute;top:12px;height:4px;background:var(--accent);border-radius:2px}
.chips{display:flex;flex-wrap:wrap;gap:6px;padding:0 12px 10px}
.chip{display:inline-flex;align-items:center;gap:6px;font:12px var(--mono);padding:3px 8px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--fg);cursor:pointer}
.chip:hover{border-color:var(--accent)}
.chip.on{border-color:var(--accent);background:var(--range)}
.chip i{width:9px;height:9px;border-radius:2px;display:inline-block}
.chip span{color:var(--fg2)}
.chip .speed-cap{color:var(--accent);font-weight:600}
#p2d{height:620px}
.legend{display:flex;flex-wrap:wrap;gap:4px 14px;padding:8px 12px 0;color:var(--fg2);font-size:12px}
.legend i{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:-1px}
.help{color:var(--muted);font-size:12px;padding:0 2px}
.strip{display:flex;gap:4px;padding:10px 12px 12px;overflow:hidden}
.th{flex:0 0 auto;width:var(--tw);border-radius:3px;overflow:hidden;cursor:pointer;background:var(--line);position:relative}
.th .im{width:100%;aspect-ratio:16/9;background-repeat:no-repeat}
.th .tt{position:absolute;left:0;bottom:0;font:10px var(--mono);color:#fff;background:rgba(0,0,0,.55);padding:1px 4px;border-radius:0 3px 0 0}
.th:hover{outline:2px solid var(--accent)}
.p2dwrap{position:relative}
#preview{position:absolute;top:44px;right:20px;width:200px;border:1px solid var(--line);border-radius:4px;background:var(--panel);box-shadow:0 2px 8px rgba(0,0,0,.18);pointer-events:none;z-index:2;overflow:hidden}
#preview .im{width:100%;aspect-ratio:16/9;background-repeat:no-repeat}
#preview .tt{font:11px var(--mono);color:var(--fg2);padding:2px 6px;text-align:center}
@media (max-width:600px){#p3d{height:46vh}#p2d{height:560px}#preview{width:120px;top:40px;right:12px}}
</style>
<div class="wrap">
<header>
 <h1 id="title"></h1>
 <span class="stat">时长 <b id="s-dur"></b></span>
 <span class="stat">保留帧 <b id="s-kept"></b></span>
 <span class="stat">片段 <b id="s-nseg"></b></span>
 <span class="stat">选区 <b id="s-range"></b></span>
</header>
<div class="panel"><div id="p3d"></div>
 <div class="legend" id="legend"></div>
 <div class="ctl">
  <label for="t0">起</label><input type="number" id="t0" step="0.1" min="0">
  <label for="t1">止</label><input type="number" id="t1" step="0.1" min="0">
  <div class="slider"><div class="track"></div><div class="fill" id="fill"></div><input type="range" id="r0" step="0.1" min="0"><input type="range" id="r1" step="0.1" min="0"></div>
  <button id="reset" type="button">全程</button>
  <button id="fit" type="button">3D 聚焦选区</button>
 </div>
 <div class="chips" id="chips"></div>
 <div class="strip" id="strip"></div>
</div>
<div class="panel p2dwrap"><div id="p2d"></div><div id="preview" hidden><div class="im"></div><div class="tt"></div></div></div>
<p class="help">3D 图：拖动旋转，滚轮缩放，右键平移。粗线是所选时间区间，细线是其余轨迹，短刺指向相机朝向。缩略图序列均匀覆盖所选区间，点击可在 3D 中定位。下方任一时间轴上框选即可更改区间，双击恢复；悬停可在 3D 中定位。红底是被剔除的帧，绿色条带是连续运镜片段。</p>
<p class="help">倍速上限：按输出帧率的运动规则估计（默认 30 fps），仅作片段属性，不改变原速切片；“待评估”表示证据不足，“无通过档位”表示没有评估档位通过。上限以已评估的档位为界，不代表允许任意更高倍速。</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.0/plotly.min.js"></script>
<script>
const D = __DATA__; const M = __META__;
const N = D.t.length, T_END = D.t[N-1];
const CATS = ['push','pull','orbit','truck','vertical','static','pan'];
const CAT_NAME = {push:'推进 push/dronie',pull:'后退 pull',orbit:'环绕 orbit/spiral',truck:'横移 truck/diagonal',vertical:'升降',static:'悬停',pan:'摇/俯仰'};
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
let range = [0, T_END];
const fmt = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;

document.getElementById('title').textContent = M.name;
document.getElementById('s-dur').textContent = fmt(M.duration);
document.getElementById('s-kept').textContent = (M.kept*100).toFixed(0)+'%';
document.getElementById('s-nseg').textContent = M.segs.length;
document.getElementById('legend').innerHTML = CATS.map(c=>`<span><i style="background:var(--c-${c})"></i>${CAT_NAME[c]}</span>`).join('') + `<span><i style="background:var(--ghost)"></i>选区外 / 剔除</span>`;

// ---------- thumbnails ----------
const TH = M.thumbs;
function thumbStyle(el, t, w){ if(!TH) return; const k=Math.min(TH.count-1, Math.max(0, Math.round(t/TH.step))); const c=k%TH.cols, r=Math.floor(k/TH.cols); const h=w*TH.h/TH.w;
  el.style.backgroundImage=`url(${TH.uri})`; el.style.backgroundSize=`${TH.cols*w}px ${TH.rows*h}px`; el.style.backgroundPosition=`-${c*w}px -${r*h}px`; }
const strip=document.getElementById('strip'), preview=document.getElementById('preview');
function buildStrip(){ if(!TH) return; strip.innerHTML=''; const W=strip.clientWidth-24; const tw=Math.max(72, Math.min(128, Math.floor(W/14)-4)); const n=Math.max(2, Math.floor((W+4)/(tw+4)));
  strip.style.setProperty('--tw', tw+'px');
  for(let k=0;k<n;k++){ const t=range[0]+(range[1]-range[0])*(n===1?0:k/(n-1)); const d=document.createElement('div'); d.className='th'; d.innerHTML='<div class="im"></div><div class="tt"></div>';
    thumbStyle(d.firstChild, t, tw); d.lastChild.textContent=fmt(t); d.title=fmt(t);
    d.addEventListener('click', ()=>{ let i=0; while(i<N-1 && D.t[i]<t) i++; mark3d(i); showPreview(t); }); strip.appendChild(d); } }
function showPreview(t){ if(!TH) return; preview.hidden=false; thumbStyle(preview.firstChild, t, preview.clientWidth); preview.lastChild.textContent=fmt(t); }
let resizeTimer; window.addEventListener('resize', ()=>{ clearTimeout(resizeTimer); resizeTimer=setTimeout(buildStrip, 150); });

// ---------- range controls ----------
const t0=document.getElementById('t0'), t1=document.getElementById('t1'), r0=document.getElementById('r0'), r1=document.getElementById('r1');
for (const el of [t0,t1,r0,r1]) el.max = T_END.toFixed(1);
function setRange(a, b, src){
  a = Math.max(0, Math.min(a, b-0.5)); b = Math.min(T_END, Math.max(b, a+0.5)); range=[a,b];
  if (src!=='num'){ t0.value=a.toFixed(1); t1.value=b.toFixed(1); }
  r0.value=a; r1.value=b;
  document.getElementById('fill').style.left = `calc(8px + ${a/T_END*100}% - ${a/T_END*16}px)`;
  document.getElementById('fill').style.width = `${(b-a)/T_END*100}%`;
  document.getElementById('s-range').textContent = `${fmt(a)} – ${fmt(b)} (${(b-a).toFixed(1)}s)`;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on', Math.abs(+c.dataset.s-a)<0.05 && Math.abs(+c.dataset.e-b)<0.05));
  draw3d(); buildStrip(); if (src!=='plot') Plotly.relayout('p2d', {'xaxis.range':[a,b]});
}
t0.addEventListener('change', ()=>setRange(+t0.value, +t1.value, 'num'));
t1.addEventListener('change', ()=>setRange(+t0.value, +t1.value, 'num'));
r0.addEventListener('input', ()=>{ setRange(+r0.value, +r1.value); showPreview(+r0.value); });
r1.addEventListener('input', ()=>{ setRange(+r0.value, +r1.value); showPreview(+r1.value); });
document.getElementById('reset').addEventListener('click', ()=>setRange(0, T_END));
document.getElementById('fit').addEventListener('click', ()=>fit3d());
const chips = document.getElementById('chips');
M.segs.forEach((s,i)=>{
  const b=document.createElement('button'); b.type='button'; b.className='chip'; b.dataset.s=s.start; b.dataset.e=s.end;
  b.innerHTML = `<i style="background:var(--c-${s.base||'truck'})"></i>${String(i+1).padStart(2,'0')} ${s.label} <span>${fmt(s.start)} · ${s.duration}s${s.radius?` · r≈${Math.round(s.radius)}m`:''}</span> <span class="speed-cap">${s.speed_limit_label||'待评估'}${s.speed_output_fps!==30?` @${s.speed_output_fps}fps`:''}</span>`;
  b.addEventListener('click', ()=>setRange(s.start, s.end)); chips.appendChild(b);
});

// ---------- 3D ----------
const idxRange = () => { let a=0,b=N-1; while(a<N && D.t[a]<range[0]) a++; while(b>0 && D.t[b]>range[1]) b--; return [a,b]; };
function tracks3d(){
  const [a,b] = idxRange(); const traces=[];
  traces.push({type:'scatter3d', mode:'lines', x:D.e, y:D.n, z:D.u, line:{color:css('--ghost'), width:2}, hoverinfo:'skip', name:'全程'});
  for (const c of CATS){
    const x=[],y=[],z=[],h=[];
    for (let i=a;i<=b;i++){ const on = D.keep[i] && D.lab[i]===c; if(on){x.push(D.e[i]);y.push(D.n[i]);z.push(D.u[i]);h.push(`${fmt(D.t[i])}  ${CAT_NAME[c]}<br>速度 ${D.sp[i]} m/s  高度 ${D.u[i]} m`);} else {x.push(null);y.push(null);z.push(null);h.push('');}
      if (on && i<b && !(D.keep[i+1] && D.lab[i+1]===c)) {x.push(D.e[i+1]);y.push(D.n[i+1]);z.push(D.u[i+1]);h.push('');} }
    traces.push({type:'scatter3d', mode:'lines', x,y,z, text:h, hoverinfo:'text', line:{color:css('--c-'+c), width:7}, name:CAT_NAME[c], connectgaps:false});
  }
  // dropped frames inside the range
  { const x=[],y=[],z=[]; for (let i=a;i<=b;i++){ if(!D.keep[i]){x.push(D.e[i]);y.push(D.n[i]);z.push(D.u[i]);} else {x.push(null);y.push(null);z.push(null);} }
    traces.push({type:'scatter3d', mode:'lines', x,y,z, line:{color:css('--ghost'), width:7}, hoverinfo:'skip', name:'剔除', connectgaps:false}); }
  // camera look direction ticks every ~2 s
  { const x=[],y=[],z=[]; const ext = Math.max(Math.max(...D.e)-Math.min(...D.e), Math.max(...D.n)-Math.min(...D.n), 10); const L = ext*0.04;
    const every = Math.max(1, Math.round(2*N/T_END));
    for (let i=a;i<=b;i+=every){ const yaw=D.cy[i]*Math.PI/180, p=D.gp[i]*Math.PI/180; const dx=Math.sin(yaw)*Math.cos(p)*L, dy=Math.cos(yaw)*Math.cos(p)*L, dz=Math.sin(p)*L;
      x.push(D.e[i], D.e[i]+dx, null); y.push(D.n[i], D.n[i]+dy, null); z.push(D.u[i], D.u[i]+dz, null); }
    traces.push({type:'scatter3d', mode:'lines', x,y,z, line:{color:css('--fg2'), width:2}, hoverinfo:'skip', name:'相机朝向'}); }
  traces.push({type:'scatter3d', mode:'markers', x:[D.e[a]], y:[D.n[a]], z:[D.u[a]], marker:{size:6, color:css('--fg'), symbol:'diamond'}, hoverinfo:'skip', name:'悬停位置'});
  return traces;
}
function layout3d(){
  const ax = {gridcolor:css('--grid'), zerolinecolor:css('--line'), backgroundcolor:'rgba(0,0,0,0)', color:css('--fg2'), showspikes:false};
  return {paper_bgcolor:'rgba(0,0,0,0)', font:{family:css('--font'), color:css('--fg2'), size:12}, showlegend:false, margin:{l:0,r:0,t:8,b:0},
    scene:{aspectmode:'data', xaxis:{...ax, title:{text:'东 (m)'}}, yaxis:{...ax, title:{text:'北 (m)'}}, zaxis:{...ax, title:{text:'相对高度 (m)'}}, dragmode:'orbit', camera:{eye:{x:0.25,y:-1.9,z:1.3}, up:{x:0,y:0,z:1}}},
    uirevision:'keep'};
}
let built3d=false;
function draw3d(){ if(!built3d){ Plotly.newPlot('p3d', tracks3d(), layout3d(), {responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage']}); built3d=true; } else Plotly.react('p3d', tracks3d(), layout3d(), {responsive:true, displaylogo:false}); }
function fit3d(){ const [a,b]=idxRange(); const sub=k=>D[k].slice(a,b+1); const r=k=>[Math.min(...sub(k)),Math.max(...sub(k))];
  const pad=v=>{const d=Math.max(v[1]-v[0],5)*0.1; return [v[0]-d,v[1]+d];};
  Plotly.relayout('p3d', {'scene.xaxis.range':pad(r('e')), 'scene.yaxis.range':pad(r('n')), 'scene.zaxis.range':pad(r('u')), 'scene.aspectmode':'data'}); }
function mark3d(i){ Plotly.restyle('p3d', {x:[[D.e[i]]], y:[[D.n[i]]], z:[[D.u[i]]]}, [CATS.length+3]); }

// ---------- 2D timelines ----------
function draw2d(){
  const fg2=css('--fg2'), grid=css('--grid');
  const ln=(y,name,color,ax,extra={})=>({type:'scatter', mode:'lines', x:D.t, y, name, line:{color, width:1.6}, yaxis:ax, hovertemplate:'%{y}', ...extra});
  const tr=[
    ln(D.sp,'地速 m/s',css('--c-push'),'y'), ln(D.cl,'爬升 m/s',css('--c-orbit'),'y'),
    ln(D.yr,'云台偏航 °/s',css('--c-pan'),'y2'), ln(D.pr,'云台俯仰 °/s',css('--c-pull'),'y2'),
    ln(D.jit,'抖动 px/s',css('--c-vertical'),'y3'), ln(D.macc,'画面加速度 px/s²',css('--c-truck'),'y3'), ln(D.cons.map(c=>c==null?null:c*100),'方向一致性 %',css('--c-static'),'y3'),
    {type:'scatter', mode:'lines', x:D.t, y:D.keep.map(k=>k?1:0), name:'保留', line:{color:css('--c-orbit'), width:0}, fill:'tozeroy', fillcolor:css('--c-orbit')+'55', yaxis:'y4', hoverinfo:'skip'},
    ln(D.u,'高度 m',css('--fg2'),'y5'),
  ];
  const shapes=[], ann=[];
  // dropped frames: red wash across all panels
  let i=0; while(i<N){ if(!D.keep[i]){ let j=i; while(j+1<N && !D.keep[j+1]) j++; shapes.push({type:'rect', xref:'x', yref:'paper', x0:D.t[i], x1:D.t[j], y0:0, y1:1, fillcolor:css('--drop'), line:{width:0}, layer:'below'}); i=j+1;} else i++; }
  // segments
  M.segs.forEach((s,k)=>{ shapes.push({type:'rect', xref:'x', yref:'y4', x0:s.start, x1:s.end, y0:0, y1:1, fillcolor:css('--c-'+(s.base||'truck'))+'66', line:{width:0}});
    ann.push({x:(s.start+s.end)/2, y:0.305+(k%2)*0.02, xref:'x', yref:'paper', text:`${String(k+1).padStart(2,'0')} ${s.label}`, showarrow:false, font:{size:10, color:fg2, family:css('--mono')}, yanchor:'bottom'}); });
  const Y=(dom,title,extra={})=>({domain:dom, title:{text:title, font:{size:11}}, gridcolor:grid, zerolinecolor:grid, color:fg2, fixedrange:true, ...extra});
  const lay={paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:{family:css('--font'), color:fg2, size:12}, margin:{l:56,r:16,t:36,b:40},
    hovermode:'x unified', hoverlabel:{bgcolor:css('--panel'), bordercolor:css('--line'), font:{color:css('--fg'), family:css('--mono'), size:11}},
    legend:{orientation:'h', y:1.02, x:0, font:{size:11}}, dragmode:'zoom', shapes, annotations:ann,
    xaxis:{anchor:'y5', title:{text:'时间 (s)'}, gridcolor:grid, color:fg2, range:range, showspikes:true, spikemode:'across', spikethickness:1, spikecolor:css('--muted'), spikedash:'solid'},
    yaxis:Y([0.78,1],'速度'), yaxis2:Y([0.56,0.74],'云台 °/s'), yaxis3:Y([0.36,0.52],'画面'), yaxis4:Y([0.24,0.30],'',{showticklabels:false, showgrid:false, zeroline:false, range:[0,1]}), yaxis5:Y([0,0.19],'高度 m')};
  Plotly.newPlot('p2d', tr, lay, {responsive:true, displaylogo:false, modeBarButtonsToRemove:['toImage','lasso2d','select2d']});
  const el=document.getElementById('p2d');
  el.on('plotly_relayout', ev=>{ if(ev['xaxis.range[0]']!==undefined) setRange(+ev['xaxis.range[0]'], +ev['xaxis.range[1]'], 'plot'); else if(ev['xaxis.autorange']) setRange(0, T_END, 'plot'); });
  el.on('plotly_hover', ev=>{ const p=ev.points[0]; if(p){ mark3d(p.pointIndex); showPreview(D.t[p.pointIndex]); } });
}
draw3d(); draw2d(); setRange(0, T_END); showPreview(0);
const mq = window.matchMedia('(prefers-color-scheme: dark)');
mq.addEventListener('change', ()=>{ Plotly.react('p3d', tracks3d(), layout3d()); draw2d(); Plotly.relayout('p2d',{'xaxis.range':range}); });
new MutationObserver(()=>{ Plotly.react('p3d', tracks3d(), layout3d()); draw2d(); Plotly.relayout('p2d',{'xaxis.range':range}); }).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});
</script>
'''

if __name__ == '__main__':
    import sys
    video = sys.argv[1]; base = os.path.splitext(os.path.basename(video))[0]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(video), base + '_cuts')
    p = build(video, f'{out}/{base}_telemetry.csv', f'{out}/{base}_flow.csv', f'{out}/{base}_segments.json', f'{out}/{base}_viz.html')
    print(p, os.path.getsize(p) // 1024, 'KB')
