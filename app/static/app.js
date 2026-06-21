"use strict";

const api = {
  async health() { return (await fetch("/api/health")).json(); },
  async kb() { return (await fetch("/api/kb")).json(); },
  async recent() { return (await fetch("/api/documents")).json(); },
  async document(id) { return (await fetch(`/api/documents/${id}`)).json(); },
  async upload(file) {
    const fd = new FormData();
    fd.append("file", file);
    const r = await fetch("/api/documents", { method: "POST", body: fd });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Upload failed (${r.status})`);
    return r.json();
  },
  async chat(message, threadId, documentIds) {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, thread_id: threadId, document_ids: documentIds || [] }),
    });
    if (!r.ok) throw new Error(`Chat failed (${r.status})`);
    return r.json();
  },
  async deleteDoc(id) { return (await fetch(`/api/documents/${id}`, { method: "DELETE" })).ok; },
  async seedKB(offline) {
    const r = await fetch("/api/kb/seed", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ offline: !!offline }),
    });
    return r.json();
  },
};

// ---------- state ----------
let threadId = localStorage.getItem("thread_id");
if (!threadId) { threadId = "t-" + Math.random().toString(36).slice(2, 12); localStorage.setItem("thread_id", threadId); }
let activeDocIds = new Set();   // documents included in the chat context
let viewedDocId = null;         // document shown in the left panel
const docMeta = {};             // id -> { filename, doc_type }

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const humanize = (k) => k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const TYPE_ICON = { invoice: "🧾", air_waybill: "✈️", cmr: "🚚", packing_list: "📦", customs_declaration: "🛃", letter: "✉️", other: "📄" };

// ---------- upload ----------
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); });
["dragenter", "dragover"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) => dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (ev) => { const f = ev.dataTransfer.files[0]; if (f) handleUpload(f); });

async function handleUpload(file) {
  dropzone.classList.add("hidden");
  $("#result").classList.add("hidden");
  const proc = $("#processing");
  proc.classList.remove("hidden");
  $("#processing-name").textContent = file.name;
  animateSteps();
  try {
    const result = await api.upload(file);
    stopSteps();
    proc.classList.add("hidden");
    dropzone.classList.remove("hidden");
    renderDocument(result);
    docMeta[result.id] = { filename: result.filename, doc_type: result.doc_type };
    activeDocIds.add(result.id);
    viewedDocId = result.id;
    addSystem(`Processed “${esc(result.filename)}” → ${badgeLabel(result.doc_type)}. Added to the chat — ask me about it!`);
    renderSuggestions(result.doc_type);
    renderActiveChips();
    updateModePill();
    loadRecent();
  } catch (err) {
    stopSteps();
    proc.classList.add("hidden");
    dropzone.classList.remove("hidden");
    addSystem(`⚠️ ${esc(err.message)}`);
  }
}

let stepTimer = null;
function animateSteps() {
  const order = ["ingest", "ocr", "classify", "extract"];
  const spans = [...document.querySelectorAll(".steps span")];
  spans.forEach((s) => s.classList.remove("active", "done"));
  let i = 0;
  spans[0]?.classList.add("active");
  stepTimer = setInterval(() => {
    if (i > 0) spans[i - 1]?.classList.add("done");
    spans[i - 1]?.classList.remove("active");
    if (i < spans.length) spans[i].classList.add("active");
    i++;
    if (i > spans.length) i = spans.length;
  }, 1400);
}
function stopSteps() { clearInterval(stepTimer); document.querySelectorAll(".steps span").forEach((s) => { s.classList.remove("active"); s.classList.add("done"); }); }

// ---------- render extracted document ----------
function badgeLabel(t) { return `${TYPE_ICON[t] || "📄"} ${humanize(t || "other")}`; }

function renderDocument(r) {
  const el = $("#result");
  const conf = r.confidence != null ? Math.round(r.confidence * 100) + "%" : "—";
  const chips = [
    r.language ? `🌐 ${esc(r.language)}` : null,
    `🎯 ${conf}`,
    r.ocr_method ? `OCR: ${esc(r.ocr_method)}` : null,
    r.extraction_method ? `extract: ${esc(r.extraction_method)}` : null,
    r.num_pages ? `${r.num_pages} page(s)` : null,
  ].filter(Boolean).map((c) => `<span class="chip">${c}</span>`).join("");

  let body = "";
  const extracted = r.extracted || {};
  if (Object.keys(extracted).length === 0) {
    body = `<p class="muted">No structured fields extracted. ${esc((r.errors || []).join(" "))}</p>`;
  } else {
    body = renderFields(extracted);
  }

  el.innerHTML = `
    <div class="result-head">
      <span class="badge">${badgeLabel(r.doc_type)}</span>
      <h3>${esc(r.filename || "Document")}</h3>
    </div>
    <div class="meta-chips">${chips}</div>
    <div class="fields">${body}</div>
    <div class="actions">
      <button class="btn primary" onclick="downloadExport('${r.id}','xlsx')">⬇︎ Excel</button>
      <button class="btn secondary" onclick="downloadExport('${r.id}','csv')">⬇︎ CSV</button>
    </div>`;
  el.classList.remove("hidden");
}

function isEmpty(v) { return v == null || v === "" || (Array.isArray(v) && v.length === 0); }

function renderFields(obj) {
  let html = "";
  const objectArrays = [];
  for (const [k, v] of Object.entries(obj)) {
    if (isEmpty(v)) continue;
    if (Array.isArray(v) && v.every((x) => x && typeof x === "object")) { objectArrays.push([k, v]); continue; }
    if (k === "extra_fields") continue;
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const inner = renderFields(v);
      if (inner) html += `<div class="subsection">${esc(humanize(k))}</div>${inner}`;
      continue;
    }
    const val = Array.isArray(v) ? v.join(", ") : v;
    html += `<div class="field-row"><div class="k">${esc(humanize(k))}</div><div class="v">${esc(val)}</div></div>`;
  }
  for (const [k, arr] of objectArrays) {
    html += `<div class="subsection">${esc(humanize(k))} (${arr.length})</div>${renderTable(arr)}`;
  }
  if (Array.isArray(obj.extra_fields) && obj.extra_fields.length) {
    const rows = obj.extra_fields
      .filter((f) => f && !isEmpty(f.value))
      .map((f) => `<div class="field-row"><div class="k">${esc(f.name)}</div><div class="v">${esc(f.value)}</div></div>`)
      .join("");
    if (rows) html += `<details class="extra"><summary>Other fields (${obj.extra_fields.length})</summary>${rows}</details>`;
  }
  return html;
}

function renderTable(arr) {
  const cols = [...new Set(arr.flatMap((o) => Object.keys(o)))].filter((c) => arr.some((o) => !isEmpty(o[c])));
  const head = cols.map((c) => `<th>${esc(humanize(c))}</th>`).join("");
  const rows = arr.map((o) => `<tr>${cols.map((c) => `<td>${esc(o[c] ?? "")}</td>`).join("")}</tr>`).join("");
  return `<table class="items"><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
}

function downloadExport(id, fmt) { window.location.href = `/api/documents/${id}/export?format=${fmt}`; }

// ---------- chat ----------
const messages = $("#messages");
function addMsg(role, text, opts = {}) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  let inner = "";
  if (opts.route) inner += `<span class="route-tag">${routeLabel(opts.route)}</span>`;
  inner += esc(text);
  if (opts.sources && opts.sources.length) inner += renderSources(opts.sources);
  div.innerHTML = inner;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}
function addSystem(text) { addMsg("system", text); }
function routeLabel(r) { return { doc_qa: "Document Q&A", rag: "Customs Law", general: "Assistant" }[r] || r; }
function renderSources(sources) {
  const items = sources.map((s) => {
    const cite = s.citation || `${s.article_number ? "Article " + s.article_number + " · " : ""}${s.title || ""}`;
    const link = s.source_url ? ` <a href="${esc(s.source_url)}" target="_blank">source</a>` : "";
    return `<div class="src"><strong>${esc(cite)}</strong>${link}<br><em>${esc(s.snippet || "")}</em></div>`;
  }).join("");
  return `<details class="sources"><summary>📚 ${sources.length} citation(s)</summary>${items}</details>`;
}

function showTyping() {
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.innerHTML = "<span></span><span></span><span></span>";
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  return el;
}

// A bot bubble that grows token-by-token during streaming.
function createBotBubble() {
  const div = document.createElement("div");
  div.className = "msg bot";
  const tag = document.createElement("span");
  tag.className = "route-tag";
  tag.style.display = "none";
  const body = document.createElement("span");
  div.append(tag, body);
  messages.appendChild(div);
  let raw = "";
  return {
    setRoute(route) { tag.textContent = routeLabel(route); tag.style.display = "block"; },
    append(text) { raw += text; body.textContent = raw; messages.scrollTop = messages.scrollHeight; },
    addSources(sources) {
      if (sources && sources.length) {
        div.insertAdjacentHTML("beforeend", renderSources(sources));
        messages.scrollTop = messages.scrollHeight;
      }
    },
    isEmpty() { return raw.length === 0; },
  };
}

// Read the SSE stream from /api/chat/stream and dispatch each event.
async function streamChat(message, threadId, documentIds, onEvent) {
  const r = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId, document_ids: documentIds || [] }),
  });
  if (!r.ok || !r.body) throw new Error(`Chat failed (${r.status})`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let sep;
    while ((sep = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, sep);
      buf = buf.slice(sep + 2);
      const line = block.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      try { onEvent(JSON.parse(line.slice(5).trim())); } catch {}
    }
  }
}

let chatBusy = false;
$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (chatBusy) return;
  const input = $("#chat-input");
  const sendBtn = document.querySelector(".send");
  const text = input.value.trim();
  if (!text) return;

  // lock the composer while the assistant is thinking
  chatBusy = true;
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  sendBtn.classList.add("loading");

  addMsg("user", text);
  const typing = showTyping();
  let bubble = null;
  let pendingRoute = null;
  try {
    await streamChat(text, threadId, Array.from(activeDocIds), (ev) => {
      if (ev.type === "route") {
        pendingRoute = ev.route;
      } else if (ev.type === "token") {
        if (!bubble) {
          typing.remove();
          bubble = createBotBubble();
          if (pendingRoute) bubble.setRoute(pendingRoute);
        }
        bubble.append(ev.text);
      } else if (ev.type === "sources") {
        if (bubble) bubble.addSources(ev.sources);
      } else if (ev.type === "error") {
        if (!bubble) { typing.remove(); bubble = createBotBubble(); }
        bubble.append("⚠️ " + ev.message);
      }
    });
    if (!bubble) { typing.remove(); addMsg("bot", "(no response)"); }
  } catch (err) {
    typing.remove();
    if (bubble) bubble.append("\n⚠️ " + err.message);
    else addMsg("bot", "⚠️ " + err.message);
  } finally {
    chatBusy = false;
    input.disabled = false;
    sendBtn.disabled = false;
    sendBtn.classList.remove("loading");
    input.focus();
  }
});

function updateModePill() {
  const n = activeDocIds.size;
  $("#mode-pill").textContent = n > 0 ? `${n} doc${n > 1 ? "s" : ""} in chat` : "General";
}

// Chips above the composer showing which documents are in the chat context.
function renderActiveChips() {
  const el = $("#active-docs");
  if (!el) return;
  if (activeDocIds.size === 0) { el.innerHTML = ""; return; }
  el.innerHTML = `<span class="active-label">In chat:</span>` +
    [...activeDocIds].map((id) => {
      const m = docMeta[id] || {};
      const icon = TYPE_ICON[m.doc_type] || "📄";
      return `<span class="doc-chip" data-id="${id}">${icon} ${esc(m.filename || id)}` +
             `<button class="chip-x" title="Remove from chat">×</button></span>`;
    }).join("");
  el.querySelectorAll(".doc-chip .chip-x").forEach((btn) =>
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      activeDocIds.delete(btn.closest(".doc-chip").dataset.id);
      renderActiveChips(); updateModePill(); loadRecent();
    }));
}

const SUGGEST = {
  invoice: ["What is the total amount?", "List all line items", "Kim sotuvchi va xaridor?"],
  air_waybill: ["What is the AWB number?", "Airport of destination?", "Какой вес груза?"],
  customs_declaration: ["What is the MRN?", "List the commodity codes", "Yetkazib berish sharti qanday?"],
  packing_list: ["Total gross weight?", "How many packages?"],
  cmr: ["Who is the carrier?", "Vehicle registration number?"],
  letter: ["Summarize this letter", "What documents are referenced?"],
  _general: ["What can you do?", "What is the customs territory of Uzbekistan?", "Bojxona to‘lovlari nima?"],
};
function renderSuggestions(type) {
  const list = SUGGEST[type] || SUGGEST._general;
  $("#suggestions").innerHTML = list.map((s) => `<span class="suggestion">${esc(s)}</span>`).join("");
  document.querySelectorAll(".suggestion").forEach((el) =>
    el.addEventListener("click", () => { $("#chat-input").value = el.textContent; $("#chat-form").requestSubmit(); }));
}

// ---------- recent docs ----------
async function loadRecent() {
  const { documents } = await api.recent();
  const el = $("#recent");
  if (!documents || !documents.length) { el.innerHTML = ""; return; }
  documents.forEach((d) => { docMeta[d.id] = { filename: d.filename, doc_type: d.doc_type }; });
  el.innerHTML = `<div class="recent-title">Recent documents</div>` +
    documents.slice(0, 8).map((d) =>
      `<div class="recent-item ${activeDocIds.has(d.id) ? "active" : ""}" data-id="${d.id}">
         <span class="ri-icon">${TYPE_ICON[d.doc_type] || "📄"}</span>
         <span class="ri-name">${esc(d.filename || d.id)}</span>
         <button class="recent-del" title="Delete document" data-id="${d.id}">🗑</button>
       </div>`).join("");
  el.querySelectorAll(".recent-item").forEach((it) => {
    // Click the row → view it on the left AND add it to the chat context.
    it.addEventListener("click", async () => {
      const r = await api.document(it.dataset.id);
      docMeta[r.id] = { filename: r.filename, doc_type: r.doc_type };
      activeDocIds.add(r.id); viewedDocId = r.id;
      renderDocument(r); renderSuggestions(r.doc_type);
      renderActiveChips(); updateModePill();
      addSystem(`Added “${esc(r.filename)}” to the chat.`);
      loadRecent();
    });
    // Delete button (appears on hover, right side) → remove the document for good.
    it.querySelector(".recent-del").addEventListener("click", async (e) => {
      e.stopPropagation();
      const id = e.currentTarget.dataset.id;
      if (!confirm("Delete this document permanently?")) return;
      await api.deleteDoc(id);
      activeDocIds.delete(id); delete docMeta[id];
      if (viewedDocId === id) { viewedDocId = null; $("#result").classList.add("hidden"); }
      renderActiveChips(); updateModePill(); loadRecent();
    });
  });
}

// ---------- boot ----------
(async function boot() {
  try {
    const h = await api.health();
    $("#model-pill").textContent = h.models?.chat || "local model";
  } catch {}
  await refreshKB();
  renderSuggestions("_general");
  updateModePill();
  addSystem("👋 Upload one or more customs documents, then ask me about them — in Uzbek, Russian or English. You can add several files to the chat and compare them. I can also answer general customs-law questions.");
  loadRecent();
})();

async function refreshKB() {
  try {
    const k = await api.kb();
    const pill = $("#kb-pill");
    if (k.count == null) { pill.textContent = "KB n/a"; return; }
    pill.textContent = `KB ${k.count}`;
    if (k.count === 0) {
      pill.classList.add("warn"); pill.style.cursor = "pointer"; pill.title = "Click to seed the knowledge base";
      pill.onclick = async () => { pill.textContent = "KB …"; await api.seedKB(false).catch(() => api.seedKB(true)); refreshKB(); };
    }
  } catch {}
}
