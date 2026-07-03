from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent import answer
from src.kb import ingest, search


app = FastAPI(title="全发首页 AI 小助手", version="0.1.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.on_event("startup")
def startup() -> None:
    search("全发平台是什么", k=1)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    return answer(payload.question)


@app.post("/api/ingest")
def rebuild_knowledge_base() -> dict:
    return ingest(reset=True)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>全发 AI 小助手</title>
  <style>
    :root { --brand:#ff4848; --ink:#1f2937; --muted:#667085; --line:#e8edf5; --bg:#f5f7fb; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"Microsoft YaHei", Arial, sans-serif; background:var(--bg); color:var(--ink); }
    .shell { max-width: 980px; margin: 0 auto; min-height: 100vh; display:grid; grid-template-rows:auto 1fr auto; }
    header { padding: 22px 18px 14px; background:#fff; border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:24px; }
    .sub { margin-top:7px; color:var(--muted); font-size:14px; }
    .chips { display:flex; gap:8px; flex-wrap:wrap; padding:14px 18px; background:#fff; border-bottom:1px solid var(--line); }
    .chip { border:1px solid #ffd3d3; color:var(--brand); background:#fff8f8; border-radius:999px; padding:8px 12px; cursor:pointer; font-size:13px; }
    main { padding:18px; overflow:auto; }
    .msg { max-width: 820px; margin: 0 0 14px; padding:14px 16px; border-radius:12px; line-height:1.7; white-space:pre-wrap; }
    .user { margin-left:auto; background:var(--brand); color:white; border-bottom-right-radius:4px; }
    .assistant { background:white; border:1px solid var(--line); border-bottom-left-radius:4px; }
    .sources { margin-top:12px; display:grid; gap:8px; }
    details { background:#fafbff; border:1px solid var(--line); border-radius:8px; padding:9px 10px; }
    summary { cursor:pointer; color:#475467; font-size:13px; }
    .rec { margin-top:12px; display:grid; gap:8px; }
    .card { border:1px solid #ffd7d7; background:#fff8f8; border-radius:10px; padding:10px 12px; }
    .card b { color:var(--brand); }
    form { display:flex; gap:10px; padding:14px 18px 18px; background:#fff; border-top:1px solid var(--line); }
    textarea { flex:1; resize:vertical; min-height:48px; max-height:140px; border:1px solid var(--line); border-radius:12px; padding:12px; font-size:15px; outline:none; }
    textarea:focus { border-color:#ff9d9d; box-shadow:0 0 0 3px rgba(255,72,72,.1); }
    button { border:0; background:var(--brand); color:white; border-radius:12px; padding:0 20px; font-weight:700; cursor:pointer; }
    .empty { color:var(--muted); text-align:center; margin-top:12vh; }
  </style>
</head>
<body>
  <div class="shell">
    <header>
      <h1>全发 AI 小助手</h1>
      <div class="sub">基于 Chroma 静态知识库的首页导览与基础问答</div>
    </header>
    <section class="chips" id="chips">
      <span class="chip">全发平台是什么？</span>
      <span class="chip">我想找项目机会，应该去哪里？</span>
      <span class="chip">怎么发布项目需求？</span>
      <span class="chip">跨境光伏项目怎么找法律服务？</span>
      <span class="chip">项目供应和项目需求有什么区别？</span>
    </section>
    <main id="chat">
      <div class="empty">不清楚从哪里开始？可以直接问我平台、频道或操作流程。</div>
    </main>
    <form id="form">
      <textarea id="question" placeholder="输入你的问题，例如：我想找有印尼光伏经验的服务商"></textarea>
      <button type="submit">发送</button>
    </form>
  </div>
  <script>
    const chat = document.querySelector("#chat");
    const form = document.querySelector("#form");
    const question = document.querySelector("#question");
    document.querySelectorAll(".chip").forEach(chip => {
      chip.addEventListener("click", () => { question.value = chip.textContent; form.requestSubmit(); });
    });

    function addMessage(role, html) {
      const empty = chat.querySelector(".empty");
      if (empty) empty.remove();
      const div = document.createElement("div");
      div.className = "msg " + role;
      div.innerHTML = html;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      question.value = "";
      addMessage("user", escapeHtml(text));
      const pending = addMessage("assistant", "正在检索知识库...");
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: text})
        });
        const data = await response.json();
        let html = escapeHtml(data.answer || "暂无回答");
        if (data.recommendations && data.recommendations.length) {
          html += '<div class="rec">' + data.recommendations.map(item =>
            `<div class="card"><b>${escapeHtml(item.channel)}</b><br>${escapeHtml(item.reason)}</div>`
          ).join("") + "</div>";
        }
        if (data.sources && data.sources.length) {
          html += '<div class="sources">' + data.sources.slice(0, 5).map(src =>
            `<details><summary>${escapeHtml(src.source)} / 片段 ${src.chunk}</summary>${escapeHtml(src.text)}</details>`
          ).join("") + "</div>";
        }
        pending.innerHTML = html;
      } catch (error) {
        pending.textContent = "请求失败：" + error;
      }
    });
  </script>
</body>
</html>
"""
