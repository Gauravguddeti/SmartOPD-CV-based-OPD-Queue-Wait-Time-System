/* SmartOPD - shared front-end helpers (all pages) */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

async function api(path, opts) {
  const res = await fetch(path, Object.assign({
    headers: { "Content-Type": "application/json" },
  }, opts || {}));
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-json */ }
  if (!res.ok) throw new Error((body && body.error) || res.statusText);
  return body;
}

function hhmm(epoch) {
  if (!epoch) return "--:--";
  const d = new Date(epoch * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function hhmmss(epoch) {
  if (!epoch) return "--:--:--";
  return new Date(epoch * 1000).toLocaleTimeString();
}
function mins(sec) { return Math.max(0, Math.round(sec / 60)); }

function toast(msg, ms) {
  let el = $("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast"; el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), ms || 2600);
}

/* ---------------------------------------------------------------- polling */
const OPD = {
  state: null,
  handlers: [],
  onState(fn) { this.handlers.push(fn); },
  async poll() {
    try {
      const s = await api("/api/state");
      this.state = s;
      this.handlers.forEach((fn) => { try { fn(s); } catch (e) { console.error(e); } });
    } catch (e) {
      const c = $("connBadge");
      if (c) c.textContent = "server offline";
    }
  },
  start() { this.poll(); setInterval(() => this.poll(), 1000); },
};

/* ------------------------------------------------------- voice announcer */
const Voice = {
  enabled: true,
  lastNo: null,
  maybeAnnounce(s) {
    if (!this.enabled || !("speechSynthesis" in window)) return;
    const calling = s.tokens.filter((t) => t.status === "called");
    const t = calling[0];
    if (t && t.no !== this.lastNo) {
      this.lastNo = t.no;
      const u = new SpeechSynthesisUtterance(
        `Token number ${t.no}, ${t.name}, please proceed to the doctor room.`);
      u.rate = 0.95;
      speechSynthesis.cancel();
      speechSynthesis.speak(u);
    }
    if (!t) this.lastNo = null;
  },
};

/* -------------------------------------------------- zone-drawing overlay */
class ZoneCanvas {
  constructor(canvas, getZones, onSave) {
    this.cv = canvas;
    this.getZones = getZones;
    this.onSave = onSave;
    this.mode = "wait";
    this.drag = null;
    canvas.addEventListener("pointerdown", (e) => this._down(e));
    canvas.addEventListener("pointermove", (e) => this._move(e));
    canvas.addEventListener("pointerup", (e) => this._up(e));
    requestAnimationFrame(() => this._draw());
  }
  _pt(e) {
    const b = this.cv.getBoundingClientRect();
    return [(e.clientX - b.left) / b.width, (e.clientY - b.top) / b.height];
  }
  _down(e) {
    const p = this._pt(e);
    this.drag = { x: p[0], y: p[1], x2: p[0], y2: p[1] };
    this.cv.setPointerCapture(e.pointerId);
  }
  _move(e) {
    if (!this.drag) return;
    const p = this._pt(e);
    this.drag.x2 = p[0]; this.drag.y2 = p[1];
  }
  _up() {
    if (!this.drag) return;
    const d = this.drag;
    const r = [Math.min(d.x, d.x2), Math.min(d.y, d.y2),
               Math.max(d.x, d.x2), Math.max(d.y, d.y2)];
    this.drag = null;
    if (r[2] - r[0] > 0.02 && r[3] - r[1] > 0.02) this.onSave(this.mode, r);
  }
  _rect(c, r, col, label) {
    const W = this.cv.width, H = this.cv.height;
    c.strokeStyle = col; c.lineWidth = 3;
    c.strokeRect(r[0] * W, r[1] * H, (r[2] - r[0]) * W, (r[3] - r[1]) * H);
    c.fillStyle = col + "2e";
    c.fillRect(r[0] * W, r[1] * H, (r[2] - r[0]) * W, (r[3] - r[1]) * H);
    if (label) {
      c.font = "bold 15px system-ui";
      c.fillStyle = col;
      c.fillText(label, r[0] * W + 6, r[1] * H + 19);
    }
  }
  _draw() {
    const c = this.cv.getContext("2d");
    const W = this.cv.width, H = this.cv.height;
    c.clearRect(0, 0, W, H);
    const z = this.getZones() || {};
    if (z.wait) this._rect(c, z.wait, "#3b82f6", "WAITING AREA");
    if (z.doc) this._rect(c, z.doc, "#f59e0b", "DOCTOR ROOM");
    if (this.drag) {
      const d = this.drag;
      this._rect(c, [Math.min(d.x, d.x2), Math.min(d.y, d.y2),
                     Math.max(d.x, d.x2), Math.max(d.y, d.y2)],
        this.mode === "wait" ? "#3b82f6" : "#f59e0b", "");
    }
    requestAnimationFrame(() => this._draw());
  }
}

/* live snapshot refresh (monitor page) + overlay that tracks the
   letterboxed video content area (works for any source aspect ratio) */
function startSnapshot(imgId, cvId, hz) {
  const img = $(imgId);
  const cv = cvId ? $(cvId) : null;
  const period = 1000 / (hz || 4);

  function fit() {
    if (!cv) return;
    const nw = img.naturalWidth, nh = img.naturalHeight;
    if (!nw || !nh) return;
    const b = img.getBoundingClientRect();
    const scale = Math.min(b.width / nw, b.height / nh);
    const w = nw * scale, h = nh * scale;
    cv.style.left = ((b.width - w) / 2) + "px";
    cv.style.top = ((b.height - h) / 2) + "px";
    cv.style.width = w + "px";
    cv.style.height = h + "px";
    cv.style.right = "auto";
    cv.style.bottom = "auto";
  }
  if (cv) {
    img.addEventListener("load", fit);
    window.addEventListener("resize", fit);
    fit();
  }

  function refresh() { img.src = "/api/snapshot.jpg?t=" + Date.now(); }
  refresh();
  setInterval(refresh, period);
  img.onerror = () => { /* placeholder stays until next poll */ };
}
