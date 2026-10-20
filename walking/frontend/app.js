const canvas = document.getElementById('stage');
const ctx    = canvas.getContext('2d');
const chart  = document.getElementById('chart');
const cctx   = chart.getContext('2d');

const el = id => document.getElementById(id);
let state = null;
let W = 0, H = 0, CW = 0, CH = 0;
let camera_x = 0;
let envBuilt = false;
let envInputs = {};

const SCALE     = 250;
const GROUND_PX = 70;

// ------------------------------------------------------------------ //
function fit() {
  const dpr = window.devicePixelRatio || 1;
  const r = canvas.getBoundingClientRect();
  canvas.width  = r.width * dpr;
  canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  W = r.width; H = r.height;

  const cr = chart.getBoundingClientRect();
  chart.width  = cr.width * dpr;
  chart.height = cr.height * dpr;
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  CW = cr.width; CH = cr.height;
}
window.addEventListener('resize', fit);

// ------------------------------------------------------------------ //
function toScreen(p) {
  return [ W / 2 + (p[0] - camera_x) * SCALE, H - GROUND_PX - p[1] * SCALE ];
}

function seg(a, b, color, w) {
  const A = toScreen(a), B = toScreen(b);
  ctx.strokeStyle = color;
  ctx.lineWidth   = w;
  ctx.lineCap     = 'round';
  ctx.beginPath();
  ctx.moveTo(A[0], A[1]);
  ctx.lineTo(B[0], B[1]);
  ctx.stroke();
}

function joint(p, color, r = 4.5) {
  const S = toScreen(p);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(S[0], S[1], r, 0, Math.PI * 2);
  ctx.fill();
}

// ------------------------------------------------------------------ //
function drawGrid() {
  const gy = H - GROUND_PX;
  const originX = W / 2 - camera_x * SCALE;

  ctx.strokeStyle = 'rgba(38,45,58,.55)';
  ctx.lineWidth = 1;
  const startM = Math.floor(camera_x - (W/2)/SCALE) - 1;
  const endM   = Math.ceil (camera_x + (W/2)/SCALE) + 1;
  for (let m = startM; m <= endM; m += 0.5) {
    const x = originX + m * SCALE;
    if (x < -50 || x > W + 50) continue;
    ctx.globalAlpha = (Math.abs(m) % 1 < 0.01) ? 0.55 : 0.22;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, gy); ctx.stroke();
  }
  for (let m = 0; m <= 2.2; m += 0.25) {
    const y = gy - m * SCALE;
    if (y < 0) break;
    ctx.globalAlpha = (Math.abs(m) % 1 < 0.01) ? 0.55 : 0.22;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = 'rgba(88,166,255,.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(W/2, gy - 300); ctx.lineTo(W/2, gy);
  ctx.stroke();

  ctx.strokeStyle = '#39425a';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke();
}

// ------------------------------------------------------------------ //
function drawAgent(ag, isFirst) {
  const b = ag.body;
  const fallen = ag.fallen;
  const gy = H - GROUND_PX;
  const alpha = ag.done ? 0.35 : (state.nAgents > 1 ? 0.85 : 1.0);

  const hx = W / 2 + (b.hip[0] - camera_x) * SCALE;

  ctx.globalAlpha = alpha;

  ctx.fillStyle = 'rgba(88,166,255,.08)';
  ctx.beginPath();
  ctx.ellipse(hx, gy + 4, 90, 9, 0, 0, Math.PI * 2);
  ctx.fill();

  const cLegL  = fallen ? '#7d3b3b' : '#79c0ff';
  const cLegR  = fallen ? '#7d3b3b' : '#56d364';
  const cTorso = fallen ? '#7d3b3b' : (state.watchBest ? '#ffd66e' : '#ff7b72');
  const cArmL  = '#d2a8ff';
  const cArmR  = '#a371f7';

  seg(b.hip, b.kneeL, cLegL, 9); seg(b.kneeL, b.ankleL, cLegL, 7);
  seg(b.ankleL, b.footL, cLegL, 5);
  seg(b.hip, b.kneeR, cLegR, 9); seg(b.kneeR, b.ankleR, cLegR, 7);
  seg(b.ankleR, b.footR, cLegR, 5);

  seg(b.neck, b.elbowL, cArmL, 6); seg(b.elbowL, b.handL, cArmL, 5);
  seg(b.neck, b.elbowR, cArmR, 6); seg(b.elbowR, b.handR, cArmR, 5);

  seg(b.hip, b.neck, cTorso, 13);

  const hd = toScreen(b.head);
  ctx.fillStyle = fallen ? '#7d3b3b' : '#e6edf3';
  ctx.beginPath();
  ctx.arc(hd[0], hd[1], b.headR * SCALE, 0, Math.PI * 2);
  ctx.fill();

  joint(b.hip, '#2b3446', 6);
  joint(b.neck, '#2b3446', 5);
  joint(b.kneeL, cLegL); joint(b.kneeR, cLegR);
  joint(b.ankleL, cLegL); joint(b.ankleR, cLegR);

  const fLx = W/2 + (b.footL[0] - camera_x) * SCALE;
  const fRx = W/2 + (b.footR[0] - camera_x) * SCALE;
  ctx.fillStyle = b.groundL ? '#79c0ff' : 'rgba(121,192,255,.25)';
  ctx.beginPath(); ctx.arc(fLx, gy, 5, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = b.groundR ? '#56d364' : 'rgba(86,211,100,.25)';
  ctx.beginPath(); ctx.arc(fRx, gy, 5, 0, Math.PI*2); ctx.fill();

  ctx.globalAlpha = 1;

  if (isFirst) {
    ctx.fillStyle = 'rgba(139,148,158,.75)';
    ctx.font = '11px ui-monospace';
    ctx.fillText(`x=${ag.hip_x.toFixed(2)} vx=${ag.hip_vx.toFixed(2)} tilt=${ag.torso.toFixed(2)}`,
                 hx - 70, gy - 6);
  }
}

// ------------------------------------------------------------------ //
function drawScene() {
  ctx.clearRect(0, 0, W, H);
  drawGrid();

  if (!state || !state.agents) return;

  for (let i = 0; i < state.agents.length; i++) {
    drawAgent(state.agents[i], i === 0);
  }

  if (state.watchBest) {
    ctx.fillStyle = '#ffd66e';
    ctx.font = 'bold 13px ui-monospace';
    ctx.fillText('👁 WATCH BEST', 20, 30);
  }
  if (state.nAgents > 1) {
    ctx.fillStyle = 'rgba(139,148,158,.85)';
    ctx.font = '11px ui-monospace';
    ctx.fillText(`agents: ${state.nAgents}`, 20, 50);
  }
}

// ------------------------------------------------------------------ //
function drawChart() {
  cctx.clearRect(0, 0, CW, CH);
  if (!state || !state.log || !state.log.length) return;

  const log = state.log;
  const vals  = log.map(d => d.reward);
  const bests = log.map(d => d.best);

  const mn = Math.min(...vals, ...bests.filter(v => v !== null));
  const mx = Math.max(...vals, ...bests.filter(v => v !== null));
  const pad = (mx - mn) * 0.12 || 1;
  const lo = mn - pad, hi = mx + pad;
  const n = log.length;

  const X = i => (n <= 1 ? 0 : (i / (n - 1)) * CW);
  const Y = v => CH - ((v - lo) / (hi - lo)) * (CH - 8) - 4;

  if (lo < 0 && hi > 0) {
    cctx.strokeStyle = 'rgba(139,148,158,.22)';
    cctx.beginPath(); cctx.moveTo(0, Y(0)); cctx.lineTo(CW, Y(0)); cctx.stroke();
  }

  cctx.strokeStyle = '#58a6ff';
  cctx.lineWidth = 1.5;
  cctx.beginPath();
  bests.forEach((v, i) => v === null ? null :
    (i ? cctx.lineTo(X(i), Y(v)) : cctx.moveTo(X(i), Y(v))));
  cctx.stroke();

  cctx.strokeStyle = '#56d364';
  cctx.lineWidth = 1.2;
  cctx.beginPath();
  vals.forEach((v, i) => i ? cctx.lineTo(X(i), Y(v)) : cctx.moveTo(X(i), Y(v)));
  cctx.stroke();

  log.forEach((d, i) => {
    const X_ = X(i), Y_ = Y(d.reward);
    if (d.watch) {
      cctx.fillStyle = '#ffd66e';
      cctx.beginPath(); cctx.arc(X_, Y_, 2.2, 0, Math.PI * 2); cctx.fill();
    } else if (d.improved) {
      cctx.fillStyle = '#58a6ff';
      cctx.beginPath(); cctx.arc(X_, Y_, 3.2, 0, Math.PI * 2); cctx.fill();
    } else if (d.replaced > 0) {
      cctx.fillStyle = '#ffa657';
      cctx.beginPath(); cctx.arc(X_, Y_, 1.8, 0, Math.PI * 2); cctx.fill();
    }
  });
}

// ------------------------------------------------------------------ //
function buildEnv(env) {
  const box = el('env');
  box.innerHTML = '';
  envInputs = {};
  const labelMap = {
    g_torso:'гравитация торса', g_leg:'гравитация ног',
    inertia_torso:'инерция торса', ctrl_torso:'контроль торса',
    ctrl_leg_hip:'контроль бедра', ctrl_leg_knee:'контроль колена',
    ankle_kp:'ankle Kp', ankle_kd:'ankle Kd',
    L_torso:'длина торса', L_thigh:'бедро', L_shin:'голень',
    target_h:'целевая высота', w_torso:'штраф наклона',
    w_omega:'штраф угл.скор', w_stay_bonus:'бонус ровной стойки',
    w_fall_warn:'предупрежд. падения', w_vx:'штраф vx', w_drift:'штраф дрифта',
    fall_penalty:'штраф падения', survive_bonus:'бонус выживания',
    temperature:'temperature', epsilon:'epsilon',
    max_steps:'макс. шагов', improved_tail:'доля хвоста',
    n_agents:'кол-во агентов',
  };
  const ranges = {
    g_torso:[1,20,0.5], g_leg:[1,15,0.5], inertia_torso:[0.5,4,0.1],
    ctrl_torso:[0,30,0.5], ctrl_leg_hip:[0,20,0.5], ctrl_leg_knee:[0,25,0.5],
    ankle_kp:[0,20,0.5], ankle_kd:[0,10,0.1],
    L_torso:[0.3,1.0,0.02], L_thigh:[0.2,0.7,0.01], L_shin:[0.2,0.7,0.01],
    target_h:[0.6,1.0,0.01],
    w_torso:[0,20,0.5], w_omega:[0,10,0.5], w_stay_bonus:[0,20,0.5],
    w_fall_warn:[0,80,1], w_vx:[0,0.5,0.01], w_drift:[0,5,0.1],
    fall_penalty:[0,500,10], survive_bonus:[0,500,10],
    // ★ temperature теперь от 0
    temperature:[0,1,0.01], epsilon:[0,0.3,0.01],
    max_steps:[50,1000,10], improved_tail:[0.05,0.8,0.05],
    n_agents:[1,20,1],
  };
  for (const [k, v] of Object.entries(env)) {
    const lab = document.createElement('label');
    lab.innerHTML = `<span>${labelMap[k] || k}</span><span class="val" id="val-${k}">${v}</span>`;
    box.appendChild(lab);
    const inp = document.createElement('input');
    inp.type = 'range';
    const [lo, hi, step] = ranges[k] || [0, Math.max(1, v*2), 0.01];
    inp.min = lo; inp.max = hi; inp.step = step; inp.value = v;
    // ★ n_agents дёргаем только когда пользователь отпустил ползунок
    if (k === 'n_agents') {
      inp.oninput = () => { el(`val-${k}`).textContent = inp.value; };
      inp.onchange = () => {
        send('update_env', { patch: { [k]: parseInt(inp.value, 10) } });
      };
    } else {
      inp.oninput = () => {
        el(`val-${k}`).textContent = inp.value;
        send('update_env', { patch: { [k]: parseFloat(inp.value) } });
      };
    }
    box.appendChild(inp);
    envInputs[k] = inp;
  }
  envBuilt = true;
}

function syncEnv(env) {
  for (const [k, v] of Object.entries(env)) {
    const inp = envInputs[k];
    if (inp && document.activeElement !== inp) {
      inp.value = v;
      const valEl = el(`val-${k}`);
      if (valEl) valEl.textContent = v;
    }
  }
}

// ------------------------------------------------------------------ //
function updateUI() {
  if (!state) return;
  el('status').textContent = state.running
    ? (state.watchBest ? 'смотрю лучшую' : 'обучение идёт')
    : 'пауза';
  el('status').classList.toggle('on', state.running);

  el('hudEp').textContent   = state.episode;
  el('hudRew').textContent  = state.reward.toFixed(1);
  el('hudBest').textContent = state.best === null ? '—' : state.best.toFixed(1);
  el('hudStep').textContent = state.step + '/' + state.maxStep;

  const box = el('models');
  if (box.dataset.n !== String(state.models.length)) {
    box.dataset.n = state.models.length;
    box.innerHTML = state.models.map(m => `
      <div class="m" data-k="${m.key}">
        <div class="top">
          <span class="dotc" style="background:${m.color}"></span>
          <span class="name">${m.label}</span>
          <span class="cnt"><span class="ctx">0</span> · q <span class="q">0</span></span>
        </div>
        <span class="act">—</span>
      </div>`).join('');
  }
  state.models.forEach(m => {
    const node = box.querySelector(`.m[data-k="${m.key}"]`);
    if (!node) return;
    node.querySelector('.act').textContent = m.action;
    node.querySelector('.ctx').textContent = m.contexts;
    node.querySelector('.q').textContent   = m.quality;
  });

  if (!envBuilt) buildEnv(state.env);
  else syncEnv(state.env);

  el('btnWatchBest').textContent = state.watchBest ? '👁 Watch best: ON' : '👁 Watch best';
  el('btnWatchBest').classList.toggle('primary', state.watchBest);
  el('btnWatchBest').disabled = !state.hasBest;
}

// ------------------------------------------------------------------ //
function frame() {
  if (state && state.agents && state.agents.length) {
    let sum = 0, n = 0;
    for (const ag of state.agents) {
      if (!ag.done) { sum += ag.body.hip[0]; n++; }
    }
    if (n === 0) {
      for (const ag of state.agents) { sum += ag.body.hip[0]; n++; }
    }
    if (n > 0) {
      const target = sum / n;
      camera_x += (target - camera_x) * 0.08;
    }
  }
  drawScene();
  drawChart();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------ //
const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
ws.onopen    = () => { el('status').textContent = 'connected'; send('pause'); };
ws.onmessage = ev => { state = JSON.parse(ev.data); updateUI(); };
ws.onclose   = () => { el('status').textContent = 'offline'; el('status').classList.remove('on'); };

function send(cmd, extra = {}) {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ cmd, ...extra }));
}

el('btnStart').onclick     = () => send('start');
el('btnPause').onclick     = () => send('pause');
el('btnReset').onclick     = () => send('reset');
el('btnResetBrain').onclick= () => { if (confirm('Сбросить все мозги моделей?')) send('reset_brain'); };
el('btnSave').onclick      = () => send('save');
el('btnLoad').onclick      = () => send('load');
el('speed').oninput = e => {
  el('speedVal').textContent = e.target.value + '×';
  send('speed', { value: Number(e.target.value) });
};

document.addEventListener('DOMContentLoaded', () => {
  el('btnWatchBest').onclick = () => send('watch_best', { value: !state.watchBest });
  el('btnLoadBest').onclick  = () => { if (confirm('Загрузить лучший снапшот в текущие модели?')) send('load_best_into_current'); };
  el('btnResetEnv').onclick  = () => { if (confirm('Сбросить параметры окружения?')) send('reset_env'); };
});

fit();
frame();