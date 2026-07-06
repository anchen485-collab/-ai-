from __future__ import annotations

"""FastAPI 服务入口。

这个文件同时承担两件事：
1. 对外提供 HTTP API，例如 /api/chat 和 /api/ingest。
2. 返回一个内置的轻量网页，方便在没有前端工程的情况下直接体验问答。
"""

import logging
import json
from collections.abc import Iterator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agent.deep.service import deep_answer
from src.agent.service import answer as normal_answer
from src.rag.embeddings import ingest, search


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="全发首页 AI 小助手", version="0.1.0")


class ChatRequest(BaseModel):
    """聊天接口请求体。

    目前一期只需要用户问题，因此只有 question 一个字段。
    这里用 Pydantic 做长度校验，避免空问题或超长输入直接进入 Agent。
    """

    question: str = Field(min_length=1, max_length=2000)
    mode: str = Field(default="normal", max_length=20)


@app.on_event("startup")
def startup() -> None:
    # 启动时预热一次知识库，提前创建 Chroma collection。
    # 这样用户第一次提问时不会承担初始化开销。
    try:
        search("全发平台是什么", k=1)
        logger.info("启动预热知识库完成")
    except FileNotFoundError:
        # 知识库目录还没准备好时，不阻塞 Web 服务启动。
        # 前端提问时会返回清晰的配置提示。
        logger.warning("启动预热跳过：知识库目录不存在")
    except Exception as exc:
        # 知识库构建、在线 embedding 或单个文档异常都不应该阻塞 Web 服务启动。
        logger.warning("启动预热跳过：%s", exc)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    # 返回内置 HTML 页面。当前项目没有独立前端构建步骤。
    return INDEX_HTML


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    # 主聊天接口：把用户问题交给 Agent，Agent 内部会完成检索、生成和推荐。
    logger.info("收到聊天请求：mode=%s, question=%s", payload.mode, payload.question)
    if payload.mode == "deep":
        return deep_answer(payload.question)
    return normal_answer(payload.question)


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    # 流式聊天接口：保留原 Agent 逻辑，把最终答案拆成小片段返回给前端形成打字机效果。
    logger.info("收到流式聊天请求：mode=%s, question=%s", payload.mode, payload.question)
    return StreamingResponse(
        stream_chat_events(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def stream_chat_events(payload: ChatRequest) -> Iterator[str]:
    """把普通 Agent 或深度思考 Agent 的结果转换成前端可消费的事件流。"""

    mode_name = "深度思考" if payload.mode == "deep" else "普通问答"
    yield sse_event({"type": "status", "text": f"{mode_name}模式启动"})

    try:
        result = deep_answer(payload.question) if payload.mode == "deep" else normal_answer(payload.question)
    except Exception as exc:
        logger.exception("流式聊天请求失败")
        yield sse_event({"type": "error", "text": f"请求失败：{exc}"})
        yield sse_event({"type": "done"})
        return

    for index, step in enumerate(result.get("steps", []), start=1):
        yield sse_event({"type": "step", "index": index, "step": step})

    answer_text = result.get("answer") or "暂无回答"
    for chunk in split_text(answer_text, size=8):
        yield sse_event({"type": "delta", "text": chunk})

    yield sse_event(
        {
            "type": "metadata",
            "recommendations": result.get("recommendations", []),
            "sources": result.get("sources", []),
            "steps": result.get("steps", []),
            "mode": result.get("mode", payload.mode),
        }
    )
    yield sse_event({"type": "done"})


def sse_event(payload: dict) -> str:
    """序列化 SSE data 事件。"""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def split_text(text: str, size: int = 8) -> Iterator[str]:
    """按固定长度拆分文本，交给前端做打字机动画。"""

    for index in range(0, len(text), size):
        yield text[index:index + size]


@app.post("/api/ingest")
def rebuild_knowledge_base() -> dict:
    # 手动重建知识库。适合桌面“知识库”目录里的 docx 更新后调用。
    return ingest()


@app.get("/api/health")
def health() -> dict:
    # 最小健康检查接口，方便确认服务是否启动。
    return {"ok": True}


# 一期为了交付简单，把页面直接内嵌在 app.py。
# 后续如果要做复杂交互，可以把它拆成独立前端项目或 templates/static 目录。
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
    .input-wrap { flex:1; display:grid; gap:10px; }
    .modebar { display:flex; align-items:center; justify-content:space-between; gap:12px; color:var(--muted); font-size:13px; }
    .mode-toggle { border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:8px; padding:8px 12px; font-weight:700; cursor:pointer; }
    .mode-toggle.active { border-color:#ffb0b0; background:#fff3f3; color:var(--brand); }
    textarea { flex:1; resize:vertical; min-height:48px; max-height:140px; border:1px solid var(--line); border-radius:12px; padding:12px; font-size:15px; outline:none; }
    textarea:focus { border-color:#ff9d9d; box-shadow:0 0 0 3px rgba(255,72,72,.1); }
    button { border:0; background:var(--brand); color:white; border-radius:12px; padding:0 20px; font-weight:700; cursor:pointer; }
    .empty { color:var(--muted); text-align:center; margin-top:12vh; }
    .trace { margin-top:12px; }
    .trace pre { white-space:pre-wrap; font-family:Consolas, "Microsoft YaHei", monospace; font-size:12px; line-height:1.6; color:#344054; }
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
      <div class="input-wrap">
        <textarea id="question" placeholder="输入你的问题，例如：我想找有印尼光伏经验的服务商"></textarea>
        <div class="modebar">
          <span id="modeText">普通问答模式</span>
          <button id="deepToggle" class="mode-toggle" type="button">深度思考</button>
        </div>
      </div>
      <button type="submit">发送</button>
    </form>
  </div>
  <script>
    const chat = document.querySelector("#chat");
    const form = document.querySelector("#form");
    const question = document.querySelector("#question");
    const deepToggle = document.querySelector("#deepToggle");
    const modeText = document.querySelector("#modeText");
    let mode = "normal";

    deepToggle.addEventListener("click", () => {
      mode = mode === "deep" ? "normal" : "deep";
      deepToggle.classList.toggle("active", mode === "deep");
      modeText.textContent = mode === "deep" ? "深度思考模式：会调用推理模型和工具循环" : "普通问答模式";
    });

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

    function createTyper(target) {
      const queue = [];
      const timer = window.setInterval(() => {
        if (!queue.length) return;
        target.textContent += queue.shift();
        chat.scrollTop = chat.scrollHeight;
      }, 18);
      return {
        push(text) {
          queue.push(...Array.from(String(text)));
        },
        drain() {
          return new Promise(resolve => {
            const check = window.setInterval(() => {
              if (queue.length) return;
              window.clearInterval(check);
              window.clearInterval(timer);
              resolve();
            }, 18);
          });
        }
      };
    }

    function renderExtras(data, steps) {
      let html = "";
      const finalSteps = data.steps && data.steps.length ? data.steps : steps;
      if (finalSteps && finalSteps.length) {
        html += '<details class="trace"><summary>深度思考过程</summary><pre>' + escapeHtml(
          finalSteps.map((step, index) => {
            const item = step.step || step;
            const parts = [`第 ${index + 1} 步`, `Thought: ${item.thought || "无"}`];
            if (item.action) parts.push(`Action: ${item.action}`);
            if (item.action_input) parts.push(`Action Input: ${JSON.stringify(item.action_input, null, 2)}`);
            if (item.observation) parts.push(`Observation: ${item.observation}`);
            return parts.join("\n");
          }).join("\n\n")
        ) + '</pre></details>';
      }
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
      return html;
    }

    async function handleStream(response, pending, statusNode, answerNode) {
      if (!response.ok || !response.body) {
        throw new Error("HTTP " + response.status);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      const typer = createTyper(answerNode);
      const steps = [];
      let metadata = {};
      let buffer = "";

      while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        const events = buffer.split("\n\n");
        buffer = events.pop();

        for (const rawEvent of events) {
          const line = rawEvent.split("\n").find(item => item.startsWith("data:"));
          if (!line) continue;
          const data = JSON.parse(line.slice(5).trim());

          if (data.type === "status" && statusNode.parentNode) {
            statusNode.textContent = data.text;
          }
          if (data.type === "step") {
            steps.push(data.step);
            if (statusNode.parentNode) statusNode.textContent = `深度思考中：第 ${data.index} 步完成`;
          }
          if (data.type === "delta") {
            if (statusNode.parentNode) statusNode.remove();
            typer.push(data.text);
          }
          if (data.type === "metadata") {
            metadata = data;
          }
          if (data.type === "error") {
            if (statusNode.parentNode) statusNode.remove();
            typer.push(data.text);
          }
        }
      }

      await typer.drain();
      pending.insertAdjacentHTML("beforeend", renderExtras(metadata, steps));
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const text = question.value.trim();
      if (!text) return;
      question.value = "";
      addMessage("user", escapeHtml(text));
      const pendingText = mode === "deep" ? "正在深度思考..." : "正在检索知识库...";
      const pending = addMessage("assistant", "");
      const statusNode = document.createElement("span");
      const answerNode = document.createElement("span");
      statusNode.textContent = pendingText;
      pending.appendChild(statusNode);
      pending.appendChild(answerNode);
      try {
        const response = await fetch("/api/chat/stream", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: text, mode})
        });
        await handleStream(response, pending, statusNode, answerNode);
      } catch (error) {
        pending.textContent = "请求失败：" + error;
      }
    });
  </script>
</body>
</html>
"""
