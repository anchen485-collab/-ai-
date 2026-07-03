# 全发首页 AI 小助手 Agent

一期最小可用版本：FastAPI 聊天接口 + Chroma 静态知识库检索 + 简洁网页入口。

## 启动

```powershell
cd D:\software\project
.\.venv311\Scripts\python.exe scripts\ingest.py
.\.venv311\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 配置

默认读取桌面知识库：

```text
C:\Users\Administrator\Desktop\知识库
```

如需自然语言生成回答，在 `.env` 中配置：

```text
DASHSCOPE_API_KEY=你的Key
LLM_MODEL=qwen-plus
```

如果不配置 Key，系统仍会从 Chroma 返回检索证据和基础推荐。

> 说明：项目原有 `.venv` 是 Python 3.14，当前 Chroma/onnxruntime 在该环境下不稳定。
> 已创建 `.venv311` 作为本项目运行环境。
> 代码通过 `src.kb` 封装 Chroma 导入，屏蔽默认 ONNX embedding 的初始化；
> 实际检索使用 `src.embeddings.HashingChineseEmbedding`。

## API

- `POST /api/chat`：问答
- `POST /api/ingest`：重建知识库
- `GET /api/health`：健康检查
