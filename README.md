# 全发首页 AI 小助手 Agent

这是前端项目 `agent/` 目录下的 AI 小助手后端模块，基于 FastAPI、LangChain、Chroma、DashScope/OpenAI 兼容接口实现。

当前能力：

- 普通问答 Agent：适合平台导览、基础问答、附件分析和知识库问答。
- 深度思考 Agent：适合复杂问题，会使用工具检索和多步分析。
- 模型切换：前端只传模型名，后端按模型所属厂商选择对应 API Key 和 Base URL。
- 图片识别：用户上传图片时，后端自动使用视觉模型，不需要前端手动选择视觉模型。
- 文档处理：支持 `.txt`、`.md`、`.docx` 等文档内容提取；较大文档会建立临时索引供 Agent 检索。
- 图片生成：支持 DashScope 文生图；同一会话内可尝试基于最近生成图进行图片编辑。
- 会话记忆：通过 `conversation_id` 保存对话上下文。

## Python 版本

建议使用 **Python 3.11**。

查看本机已安装的 Python 版本：

```powershell
py -0p
```

查看当前默认 Python 版本：

```powershell
python --version
```

如果本机同时安装了多个 Python 版本，创建虚拟环境时优先使用：

```powershell
py -3.11 -m venv .venv
```

这样可以明确让 `agent/.venv` 使用 Python 3.11，避免误用其它版本导致依赖安装或运行异常。

## 创建虚拟环境

进入 agent 目录：

```powershell
cd D:\fronted\-\agent
```

创建虚拟环境：

```powershell
py -3.11 -m venv .venv
```

激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

确认虚拟环境 Python 版本：

```powershell
python --version
```

正常应显示：

```text
Python 3.11.x
```

如果 PowerShell 不允许激活脚本，也可以直接使用虚拟环境里的 Python：

```powershell
.\.venv\Scripts\python.exe --version
```

## 安装依赖

在 `agent/` 目录下执行：

```powershell
python -m pip install -r requirements.txt
```

如果没有激活虚拟环境，使用：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

当前主要依赖：

- `fastapi`：HTTP API 服务。
- `uvicorn`：FastAPI 运行服务。
- `python-dotenv`：读取 `.env` 配置。
- `python-multipart`：支持文件上传。
- `langchain` / `langchain-core` / `langchain-community` / `langchain-openai`：Agent、工具和模型调用。
- `langchain-chroma` / `chromadb`：向量库。
- `docx2txt`：提取 docx 文本。
- `dashscope`：DashScope 图片生成等能力。
- `Pillow`：图片压缩和格式处理。
- `oss2`：阿里云 OSS 文件上传、下载和删除。

依赖版本以 `requirements.txt` 为准。

## 配置文件

复制配置模板：

```powershell
Copy-Item .env.example .env
```

`.env` 只保存在本地，不要提交真实 Key。

常用配置：

```env
# 千问模型共用配置：qwen-plus、qwen-vl-plus 等。
DASHSCOPE_API_KEY=
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# DeepSeek 或其它 OpenAI 兼容模型共用配置。
OPENAI_COMPATIBLE_API_KEY=
OPENAI_COMPATIBLE_BASE_URL=https://api.deepseek.com

# 知识库配置。
KB_SOURCE_DIR=src/rag/data
CHROMA_DIR=data/chroma
CHROMA_COLLECTION=qf_knowledge_base

# 默认模型配置。
# MODEL_OPTIONS 只控制前端可手动选择的文本模型列表。
# VISION_MODEL 在用户上传图片时由后端自动使用，不需要放进 MODEL_OPTIONS。
LLM_MODEL=qwen-plus
MODEL_OPTIONS=qwen-plus,deepseek-chat,deepseek-reasoner
DEFAULT_MODEL=qwen-plus
TEXT_MODEL=qwen-plus
VISION_MODEL=qwen-vl-plus

# 上传附件配置。
UPLOAD_DIR=data/uploads
UPLOAD_MAX_MB=5
UPLOAD_IMAGE_MAX_COUNT=3
UPLOAD_TOTAL_MAX_MB=12
VISION_IMAGE_MAX_SIDE=1280
VISION_IMAGE_JPEG_QUALITY=80
VISION_PAYLOAD_MAX_MB=8
ATTACHMENT_TEXT_MAX_CHARS=8000
ATTACHMENT_INDEX_THRESHOLD_CHARS=4000

# 深度思考 Agent 参数。
DEEP_AGENT_MODEL=deepseek-reasoner
DEEP_AGENT_MAX_STEPS=8
DEEP_AGENT_TEMPERATURE=0.2

# 图片生成配置。
IMAGE_GENERATION_MODEL=wan2.7-image-pro
IMAGE_GENERATION_API_KEY=
IMAGE_GENERATION_BASE_URL=
IMAGE_GENERATION_SIZE=1280*1280
IMAGE_GENERATION_COUNT=1
IMAGE_GENERATION_PROMPT_EXTEND=true
IMAGE_GENERATION_WATERMARK=false
GENERATED_IMAGE_DIR=data/generated
GENERATED_IMAGE_HISTORY_DIR=data/generated/history

# 阿里云 OSS 配置。
# 启用后，上传附件写入 uploads，生成图片写入 image-result，图片生成历史写入 history。
OSS_ENABLED=false
OSS_ENDPOINT=
OSS_BUCKET=
OSS_ACCESS_KEY_ID=
OSS_ACCESS_KEY_SECRET=
OSS_UPLOADS_PREFIX=uploads
OSS_IMAGE_RESULT_PREFIX=image-result
OSS_HISTORY_PREFIX=history

# 图片编辑配置。用于“修改上一张图”这类请求。
IMAGE_EDIT_MODEL=wan2.6-image
IMAGE_EDIT_BASE_URL=
IMAGE_EDIT_PROMPT_EXTEND=true
IMAGE_EDIT_WATERMARK=false
```

模型配置规则：

- `qwen*` 模型默认使用 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。
- `deepseek*` 模型默认使用 `OPENAI_COMPATIBLE_API_KEY` 和 `OPENAI_COMPATIBLE_BASE_URL`。
- 前端切换模型时只传模型名，后端不接收也不暴露 API Key。
- 用户上传图片时，后端自动使用 `VISION_MODEL`。

OSS 存储规则：

- `uploads`：保存用户上传的图片和文档附件。
- `image-result`：保存图片生成/图片编辑接口返回给前端展示的结果图。
- `history`：保存图片生成历史，用于同一会话里继续“修改上一张图”。
- `OSS_ENABLED=false` 时完全走本地目录，适合本地开发。
- `OSS_ENABLED=true` 时必须填写 `OSS_ENDPOINT`、`OSS_BUCKET`、`OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`。
- 前端仍然访问原来的 `/api/attachments/{attachment_id}` 和 `/api/generated-images/{image_id}`，后端会在本地缺失时从 OSS 拉回文件。

## 启动服务

方式一：使用虚拟环境中的 uvicorn：

```powershell
cd D:\fronted\-\agent
.\.venv\Scripts\uvicorn.exe app:app --host 0.0.0.0 --port 8000 --reload
```

方式二：使用 Python 模块方式：

```powershell
cd D:\fronted\-\agent
.\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

健康检查：

```text
http://127.0.0.1:8000/api/health
```

如果 `8000` 端口被占用，查看占用进程：

```powershell
netstat -ano | findstr :8000
```

如果改用其它端口，需要同步修改前端根目录 `vite.config.js` 中 `/agent-api` 的 `target`。

## 前端联调

前端页面不会直接请求 `http://127.0.0.1:8000/api`，而是请求：

```text
/agent-api
```

根目录 `vite.config.js` 会把它代理到：

```text
http://127.0.0.1:8000/api
```

注意：

- AI 小助手接口走 `/agent-api`。
- Java 业务后端接口继续走 `/api`。
- 不要把根目录 `/api` 代理全部改成本地 FastAPI，否则其它业务页面会请求不到 Java 后端。

## 主要接口

### 健康检查

```http
GET /api/health
```

### 聊天

```http
POST /api/chat
POST /api/chat/stream
```

请求体示例：

```json
{
  "question": "全发平台是什么？",
  "mode": "normal",
  "model": "qwen-plus",
  "conversation_id": null,
  "attachment_ids": []
}
```

字段说明：

- `question`：用户问题。
- `mode`：`normal` 普通模式，`deep` 深度思考模式。
- `model`：文本模型名，可为空；为空时使用默认模型。
- `conversation_id`：会话 ID，用于记忆和图片生成历史。
- `attachment_ids`：本轮请求携带的附件 ID。

### 模型列表与切换

```http
GET /api/models
POST /api/models/switch
```

切换模型接口只记录模型名；实际回答时也可以在 `/api/chat` 或 `/api/chat/stream` 请求体中传 `model`，本轮优先使用该模型。

### 附件上传

```http
POST /api/attachments/images
POST /api/attachments/files
GET  /api/attachments/{attachment_id}
```

说明：

- `/api/attachments/images` 只允许上传图片。
- `/api/attachments/files` 只允许上传文档。
- 图片会走视觉模型识别。
- 文档会提取文本；超过 `ATTACHMENT_INDEX_THRESHOLD_CHARS` 的文档会建立会话级临时索引。

### 图片生成

```http
POST /api/images/generate
GET  /api/generated-images/{image_id}
```

说明：

- 文生图使用 DashScope 图片生成模型。
- 如果用户在同一 `conversation_id` 下提出“修改上一张图”这类请求，后端会尝试使用最近一次生成图作为参考图。
- 生成结果会下载到 `data/generated`，再返回给前端渲染。

### 调试追踪

```http
GET /api/traces
GET /api/traces/{trace_id}
```

用于查看一次请求中的预处理、附件识别、模型调用、工具调用等链路信息。

## 项目结构

```text
agent/
├─ app.py                       FastAPI 入口
├─ requirements.txt             Python 依赖
├─ .env.example                 环境变量示例
├─ data/                        运行时数据，通常不提交
└─ src/
   ├─ agent/
   │  ├─ normal/                普通 Agent
   │  ├─ deep/                  深度思考 Agent
   │  └─ streaming.py           流式输出
   ├─ attachments/              附件上传、加载、文档解析、会话索引
   ├─ core/
   │  └─ config.py              配置读取和模型配置解析
   ├─ media/                    图片分析、图片上传、图片生成、图片编辑
   ├─ memory/                   conversation_id 会话记忆
   ├─ rag/                      知识库加载、切分、向量检索
   ├─ tools/                    Agent 工具
   └─ trace.py                  调试追踪
```

## 常见问题

### 1. 修改 `.env` 后没有生效

`.env` 在服务启动时读取。修改后需要停止并重新启动 FastAPI。

如果 PyCharm Run/Debug Configuration 配置了同名环境变量，它可能会覆盖 `.env`。

### 2. 前端请求 AI 小助手失败

确认 FastAPI 已启动：

```text
http://127.0.0.1:8000/api/health
```

再确认前端 `vite.config.js` 中 `/agent-api` 的 `target` 端口和 FastAPI 实际端口一致。

### 3. 上传后出现 `agent/storage` 目录

当前默认上传目录是：

```env
UPLOAD_DIR=data/uploads
```

如果请求后仍出现 `agent/storage`，一般说明运行中的后端进程读到了旧的 `UPLOAD_DIR=storage` 配置。检查 PyCharm 环境变量、系统环境变量，然后重启服务。

### 4. 知识库目录不存在

检查：

```env
KB_SOURCE_DIR=src/rag/data
```

如果目录不存在，部分 RAG 流程可能无法检索。可以先创建该目录，或者把 `KB_SOURCE_DIR` 改为真实知识库目录。

### 5. 终端中文乱码

Windows PowerShell 可以先执行：

```powershell
chcp 65001
```

然后重新启动服务。

## 提交提醒

当前 agent 开发分支是：

```text
feature/ai-agent
```

不要直接推送到 `master`。提交 agent 文档时可以执行：

```powershell
git add agent/README.md
git commit -m "docs: update agent readme"
git push origin feature/ai-agent
```
