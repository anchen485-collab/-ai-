# Python 文件说明

本文档说明当前项目中每个 `.py` 文件的职责。项目已按模块拆分为 `core`、`agent`、`rag` 三块，避免所有代码堆在 `src` 根目录。

## 当前目录结构

```text
D:\software\project
├─ app.py
├─ scripts
│  └─ ingest.py
└─ src
   ├─ __init__.py
   ├─ core
   │  ├─ __init__.py
   │  └─ config.py
   ├─ agent
   │  ├─ __init__.py
   │  └─ service.py
   └─ rag
      ├─ __init__.py
      ├─ chroma_noop.py
      ├─ documents.py
      ├─ embeddings.py
      └─ store.py
```

## 调用链路

问答链路：

```text
浏览器页面
  ↓
app.py /api/chat
  ↓
src.agent.service.answer()
  ↓
src.rag.store.search()
  ↓
Chroma
```

入库链路：

```text
scripts/ingest.py
  ↓
src.rag.store.ingest()
  ↓
src.rag.documents.build_chunks()
  ↓
src.rag.embeddings.HashingChineseEmbedding
  ↓
Chroma data/chroma
```

## app.py

FastAPI 服务入口。

它负责：

- 创建 FastAPI 应用。
- 提供内置网页 `GET /`。
- 提供聊天接口 `POST /api/chat`。
- 提供重建知识库接口 `POST /api/ingest`。
- 提供健康检查接口 `GET /api/health`。
- 启动时预热一次知识库检索。

核心导入：

```python
from src.agent.service import answer
from src.rag.store import ingest, search
```

也就是说，`app.py` 只负责 HTTP 层，不直接处理文档解析、向量库细节或大模型调用。

## scripts/ingest.py

知识库入库脚本。

运行方式：

```powershell
.\.venv311\Scripts\python.exe scripts\ingest.py
```

它负责：

- 把项目根目录加入 `sys.path`。
- 调用 `src.rag.store.ingest(reset=True)`。
- 打印入库结果。

适用场景：

- 桌面“知识库”文件夹里的 docx 更新后。
- 首次克隆项目后需要重新生成本地 Chroma 数据。

## src/__init__.py

`src` 包标记文件。

它没有业务逻辑，只用于让 Python 识别 `src` 为包。

## src/core/__init__.py

`core` 模块标记文件。

当前 `core` 只放项目通用配置。

## src/core/config.py

项目配置中心。

它负责：

- 加载项目根目录 `.env`。
- 读取知识库路径。
- 读取 Chroma 存储路径。
- 读取 collection 名称。
- 读取通义模型名。
- 读取检索数量和切块参数。

关键对象：

```python
settings = Settings()
```

其他模块都通过 `settings` 获取配置，避免路径和参数散落在各处。

默认配置：

- 知识库目录：`C:\Users\Administrator\Desktop\知识库`
- Chroma 目录：`D:\software\project\data\chroma`
- collection：`qf_knowledge_base`
- LLM 模型：`qwen-plus`

## src/agent/__init__.py

`agent` 模块标记文件。

当前 agent 逻辑集中在 `service.py`。

## src/agent/service.py

问答编排模块。

它负责：

- 接收用户问题。
- 调用 RAG 检索。
- 尝试调用通义千问生成回答。
- 没有 API Key 或模型失败时返回检索兜底回答。
- 根据关键词给出频道推荐卡片。

主要函数：

- `answer(question)`

  Agent 主入口。返回：

  ```python
  {
      "answer": "...",
      "recommendations": [...],
      "sources": [...]
  }
  ```

- `recommendations(question)`

  根据关键词匹配频道或操作路径，例如：

  - 发布与入驻路径
  - 项目需求频道
  - 项目供应频道
  - 法律服务频道
  - 金融供应频道
  - 物流/仓储供需频道

- `llm_answer(question, hits)`

  如果配置了 `DASHSCOPE_API_KEY`，就使用 `ChatTongyi` 生成自然语言回答。

  如果没有 Key 或调用失败，返回 `None`。

- `fallback_answer(question, hits)`

  不调用大模型，直接把检索到的知识片段和来源整理成回答。

- `compact(text, max_len)`

  压缩长片段，避免回答过长。

## src/rag/__init__.py

`rag` 模块标记文件。

RAG 相关代码都放在这个目录。

## src/rag/documents.py

文档加载和切块模块。

它负责：

- 读取 docx 文件。
- 从 docx 内部 XML 中提取文本。
- 将长文本切成适合检索的 chunk。
- 为 chunk 附加来源文件名、来源路径、片段编号。

主要函数：

- `extract_docx_text(path)`

  从 docx 中提取文本。

- `load_documents(source_dir=None)`

  读取知识库目录下全部 `.docx` 文件。

- `split_text(text)`

  按中文标点和换行切块，并保留 overlap。

- `build_chunks(source_dir=None)`

  返回可写入 Chroma 的 `Chunk` 列表。

## src/rag/embeddings.py

本地 embedding 模块。

它负责：

- 把中文文本转换成向量。
- 避免依赖外部 embedding API。
- 避免下载模型。
- 避免使用 Chroma 默认 ONNX embedding。

核心类：

```python
HashingChineseEmbedding
```

工作方式：

- 提取中文单字。
- 提取 2/3/4-gram。
- 用 hash 映射到固定维度向量。
- 做向量归一化。

这是一期为了简单稳定采用的方案。后续如果要提升效果，可以替换为专业 embedding 模型。

## src/rag/chroma_noop.py

Chroma 遥测空实现。

它负责：

- 禁用 Chroma 的遥测事件。
- 避免本地运行时出现无关 posthog warning。

核心类：

```python
NoopTelemetry
```

## src/rag/store.py

Chroma 向量库访问层。

它是项目中唯一直接操作 Chroma 的模块。

它负责：

- 创建 Chroma collection。
- 将知识库 chunk 写入 Chroma。
- 根据用户问题检索相关 chunk。
- 绕开 Chroma 默认 ONNX embedding 初始化。

主要数据结构：

- `SearchHit`

字段：

- `text`：命中的知识片段。
- `source`：来源 docx 文件名。
- `chunk`：片段编号。
- `distance`：Chroma 距离值。

主要函数：

- `get_collection()`

  获取或创建 Chroma collection。

- `ingest(source_dir=None, reset=True)`

  重建知识库索引。

- `search(query, k=None)`

  检索相关知识片段。

- `import_chromadb()`

  导入 Chroma 前注入一个很小的 `onnxruntime` stub。

  原因是当前项目不使用 Chroma 默认 ONNX embedding，但 Chroma 0.5 在导入时会初始化它；本机 onnxruntime DLL 不稳定，所以这里屏蔽掉默认初始化。

## 常用命令

重建知识库：

```powershell
cd D:\software\project
.\.venv311\Scripts\python.exe scripts\ingest.py
```

启动服务：

```powershell
cd D:\software\project
.\.venv311\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```

访问页面：

```text
http://127.0.0.1:8000
```

## 当前版本边界

已经具备：

- FastAPI 服务
- 简洁网页入口
- docx 静态知识库解析
- Chroma 检索
- 本地 hashing embedding
- 通义千问回答增强
- 无 API Key 兜底回答
- 基础频道推荐

暂未实现：

- 后台知识库管理页面
- 图片识别
- 用户登录和权限
- 动态企业/项目数据推荐
- 多轮状态管理
- 埋点统计后台

