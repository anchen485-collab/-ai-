# 全发首页 AI 小助手 Agent

这是一个基于 FastAPI 的 AI 问答项目，提供内置网页入口、知识库检索、普通问答 Agent 和深度思考 Agent。

当前能力：

- 普通问答：基于本地 Chroma 知识库检索，并可调用通义千问生成回答。
- 深度思考：使用 ReAct 流程，支持 `Thought -> Action -> Observation` 循环。
- 共享工具：普通 Agent 和深度思考 Agent 共用 `src/tools/` 下的工具。
- 流式输出：前端通过 `/api/chat/stream` 实现打字机效果。
- 内置页面：无需单独前端工程，启动 FastAPI 后即可在浏览器使用。

## 环境要求

建议使用 Python 3.11。

当前项目已存在虚拟环境：

```text
D:\PycharmProject\.venv
```

如果需要重新安装依赖：

```powershell
cd D:\PycharmProject
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置说明

项目会读取根目录下的 `.env` 文件。

参考配置见 `.env.example`：

```env
# 普通 Agent 可选配置。未配置时仍会返回检索证据。
DASHSCOPE_API_KEY=
LLM_MODEL=qwen-plus

# 知识库配置
KB_SOURCE_DIR=C:\Users\Administrator\Desktop\知识库
CHROMA_DIR=.\data\chroma
CHROMA_COLLECTION=qf_knowledge_base

# 深度思考 Agent 配置。DeepSeek-R1 示例：
OPENAI_COMPATIBLE_API_KEY=
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
DEEP_AGENT_MODEL=deepseek-reasoner
DEEP_AGENT_MAX_STEPS=8
DEEP_AGENT_TEMPERATURE=0.2
```

说明：

- `KB_SOURCE_DIR` 是原始知识库目录，默认读取桌面 `知识库` 文件夹。
- 知识库目录中目前主要读取 `.docx` 文件。
- `DASHSCOPE_API_KEY` 用于普通 Agent 的自然语言生成。
- `OPENAI_COMPATIBLE_API_KEY` 用于深度思考 Agent。DeepSeek 的接口兼容 OpenAI Chat Completions 格式。
- 如果知识库目录不存在，服务不会启动失败，但问答时会返回清晰提示。

## 启动项目

进入项目目录：

```powershell
cd D:\PycharmProject
```

如果需要先重建知识库：

```powershell
.\.venv\Scripts\python.exe scripts\ingest.py
```

启动 FastAPI：

```powershell
.\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

如果 `8000` 端口被占用，可以换成：

```powershell
.\.venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8001
```

## PyCharm 启动方式

也可以不用命令行，在 PyCharm 里配置运行项：

- 类型：Python
- Module name：`uvicorn`
- Parameters：`app:app --host 127.0.0.1 --port 8000`
- Working directory：`D:\PycharmProject`
- Python interpreter：`D:\PycharmProject\.venv\Scripts\python.exe`

点击运行后，在浏览器打开 `http://127.0.0.1:8000`。

## 使用方式

打开页面后，可以直接输入问题。

默认是普通问答模式。

点击输入框下方的“深度思考”按钮后，会切换到深度思考 Agent：

- 普通模式：更快，适合平台导览、基础问答。
- 深度思考模式：会调用推理模型，通过 ReAct 循环检索和反思，适合复杂问题。

两个模式都支持前端打字机效果。

## API

### 健康检查

```http
GET /api/health
```

### 普通 JSON 问答

```http
POST /api/chat
Content-Type: application/json
```

请求体：

```json
{
  "question": "全发平台是什么？",
  "mode": "normal"
}
```

`mode` 可选：

- `normal`：普通 Agent
- `deep`：深度思考 Agent

### 流式问答

```http
POST /api/chat/stream
Content-Type: application/json
```

请求体：

```json
{
  "question": "全发平台是什么？",
  "mode": "deep"
}
```

返回格式为 SSE 风格事件流：

```text
data: {"type":"status","text":"深度思考模式启动"}

data: {"type":"step","index":1,"step":{...}}

data: {"type":"delta","text":"全发平台是"}

data: {"type":"metadata","sources":[...],"recommendations":[...]}

data: {"type":"done"}
```

前端页面使用该接口实现打字机效果。

### 重建知识库

```http
POST /api/ingest
```

会重新读取 `KB_SOURCE_DIR` 中的 `.docx` 文件，并写入本地 Chroma。

## 项目结构

```text
app.py                         FastAPI 入口和内置网页
requirements.txt               Python 依赖
.env.example                   环境变量示例
scripts/
  ingest.py                    知识库入库脚本
src/
  core/
    config.py                  配置读取
  agent/
    service.py                 普通 Agent
    deep/
      client.py                OpenAI 兼容模型客户端
      parser.py                ReAct 输出解析
      prompts.py               深度思考 Prompt
      react_agent.py           ReAct 主循环和反思机制
      service.py               深度思考 Agent 服务入口
  tools/
    rag.py                     共享 RAG 工具，使用 LangChain @tool
  rag/
    documents.py               docx 文档读取和切分
    embeddings.py              本地 Hashing 中文 embedding
    store.py                   Chroma 检索和入库封装
    chroma_noop.py             关闭 Chroma 遥测的空实现
```

## Agent 设计

### 普通 Agent

位置：

```text
src/agent/service.py
```

流程：

1. 调用共享 RAG 工具检索知识库。
2. 如果配置了 `DASHSCOPE_API_KEY`，使用通义模型生成回答。
3. 如果没有配置模型 Key，则返回检索证据和基础推荐。

### 深度思考 Agent

位置：

```text
src/agent/deep/
```

核心流程：

```text
Thought -> Action -> Observation -> Thought -> ... -> Final Answer
```

特点：

- 使用 ReAct 格式约束模型输出。
- Action 当前支持 `rag_search`。
- 如果连续三次检索没有新进展，会自动加入反思提示。
- 最终答案会附带来源、推荐和思考步骤。

## 工具层

工具统一放在：

```text
src/tools/
```

当前工具：

- `rag_search`：检索本地 Chroma 知识库。

工具使用 `langchain_core.tools.tool` 装饰器定义，后续如果接入完整 LangChain Agent，可以直接放入工具列表。

## 调试日志

项目启动后，终端会打印关键日志，例如：

```text
收到聊天请求：mode=deep, question=全发平台是什么？
深度思考 Agent 进入第 1 步
调用 OpenAI 兼容模型：model=deepseek-reasoner
RAG 检索完成：query=全发平台, hits=5
深度思考 Agent 生成最终答案：step=2
```

这些日志可以帮助排查：

- 请求是否进入正确模式
- 模型是否调用成功
- 工具是否执行
- RAG 命中数量
- 深度思考执行到第几步

## 常见问题

### 1. 启动后浏览器打不开

先确认终端中是否出现：

```text
Uvicorn running on http://127.0.0.1:8000
```

如果提示端口被占用，换 `8001`。

### 2. 提示知识库目录不存在

检查 `.env`：

```env
KB_SOURCE_DIR=C:\Users\Administrator\Desktop\知识库
```

确保该目录存在，并且里面有 `.docx` 文件。

### 3. 深度思考模式提示缺少 API Key

检查 `.env`：

```env
OPENAI_COMPATIBLE_API_KEY=你的Key
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com
DEEP_AGENT_MODEL=deepseek-reasoner
```

### 4. 普通模式没有生成自然语言回答

如果没有配置 `DASHSCOPE_API_KEY`，普通 Agent 会使用检索兜底回答。

配置后可使用通义模型生成更自然的回答。

### 5. 终端中文乱码

这是 Windows PowerShell 编码显示问题，通常不影响 Web 页面和接口返回。

可尝试：

```powershell
chcp 65001
```

然后重新启动服务。
