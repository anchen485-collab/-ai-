# Python 文件说明

本文档说明当前项目中每个 `.py` 文件的作用、关键函数和它们之间的调用关系。它和 `README.md` 同级，主要用于后续维护、交接和二次开发。

## 项目整体流程

当前项目是一个“全发首页 AI 小助手”的最小可用版本，核心链路如下：

```text
用户浏览器
  ↓
app.py 的 FastAPI 接口 /api/chat
  ↓
src.agent.answer()
  ↓
src.kb.search()
  ↓
Chroma 向量库
  ↓
返回知识库片段
  ↓
src.agent 组织回答
  ↓
返回给前端页面
```

知识库入库流程如下：

```text
桌面“知识库”目录下的 docx 文件
  ↓
src.documents.extract_docx_text()
  ↓
src.documents.split_text()
  ↓
src.embeddings.HashingChineseEmbedding
  ↓
src.kb.ingest()
  ↓
写入 data/chroma
```

## app.py

位置：`D:\software\project\app.py`

这是 FastAPI 服务入口，也是浏览器页面入口。

主要职责：

- 创建 FastAPI 应用实例。
- 提供网页聊天界面。
- 提供聊天 API。
- 提供知识库重建 API。
- 提供健康检查 API。
- 服务启动时预热一次知识库检索。

关键对象和函数：

- `app = FastAPI(...)`
  创建 FastAPI 应用。

- `ChatRequest`
  Pydantic 请求模型，用来校验 `/api/chat` 的请求体。
  当前只包含一个字段：

  ```python
  question: str
  ```

- `startup()`
  服务启动时执行。
  它会调用：

  ```python
  search("全发平台是什么", k=1)
  ```

  作用是提前初始化 Chroma 集合，避免第一次用户提问时才初始化。

- `index()`
  对应接口：

  ```text
  GET /
  ```

  返回内置的 `INDEX_HTML` 页面。这个页面是一个简洁聊天界面，不需要额外前端项目。

- `chat(payload: ChatRequest)`
  对应接口：

  ```text
  POST /api/chat
  ```

  调用：

  ```python
  answer(payload.question)
  ```

  返回结构包括：

  - `answer`：回答内容
  - `recommendations`：频道或路径推荐
  - `sources`：检索来源片段

- `rebuild_knowledge_base()`
  对应接口：

  ```text
  POST /api/ingest
  ```

  调用：

  ```python
  ingest(reset=True)
  ```

  用于重新读取知识库 docx 并重建 Chroma 索引。

- `health()`
  对应接口：

  ```text
  GET /api/health
  ```

  返回：

  ```json
  {"ok": true}
  ```

- `INDEX_HTML`
  内嵌的 HTML/CSS/JS 页面。页面内会调用 `/api/chat`，并展示回答、推荐卡片和知识来源。

启动方式：

```powershell
cd D:\software\project
.\.venv311\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000
```

## scripts/ingest.py

位置：`D:\software\project\scripts\ingest.py`

这是命令行入库脚本，用来把桌面“知识库”目录里的 docx 文件写入 Chroma。

主要职责：

- 把项目根目录加入 `sys.path`，确保脚本从 `scripts` 目录运行时也能导入 `src` 包。
- 调用 `src.kb.ingest(reset=True)` 重建知识库。
- 打印入库结果。

关键逻辑：

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kb import ingest
```

这段是为了保证下面命令可以正常运行：

```powershell
.\.venv311\Scripts\python.exe scripts\ingest.py
```

主入口：

```python
if __name__ == "__main__":
    result = ingest(reset=True)
    print(result)
```

成功时会输出类似：

```text
{'collection': 'qf_knowledge_base', 'documents': 4, 'chunks': 25, 'path': 'D:\\software\\project\\data\\chroma'}
```

## src/__init__.py

位置：`D:\software\project\src\__init__.py`

这是 Python 包标记文件。

主要职责：

- 让 `src` 目录被 Python 识别为一个包。
- 支持如下导入方式：

  ```python
  from src.agent import answer
  from src.kb import ingest
  ```

当前文件内容为空，这是正常的。

## src/config.py

位置：`D:\software\project\src\config.py`

这是项目配置模块，负责统一读取环境变量和默认配置。

主要职责：

- 读取项目根目录 `.env`。
- 定义知识库路径、Chroma 存储路径、集合名称、模型名称、检索数量、分块参数等配置。
- 向其他模块暴露统一的 `settings` 对象。

关键变量：

- `ROOT_DIR`
  项目根目录：

  ```python
  D:\software\project
  ```

- `load_dotenv(ROOT_DIR / ".env")`
  自动加载 `.env` 文件。

关键函数：

- `_path_from_env(name, default)`
  从环境变量读取路径。如果环境变量不存在，就使用默认路径。

核心配置类：

- `Settings`

字段说明：

- `kb_source_dir`
  默认知识库目录：

  ```text
  C:\Users\Administrator\Desktop\知识库
  ```

- `chroma_dir`
  Chroma 持久化目录：

  ```text
  D:\software\project\data\chroma
  ```

- `chroma_collection`
  Chroma 集合名称：

  ```text
  qf_knowledge_base
  ```

- `llm_model`
  调用通义千问时使用的模型，默认：

  ```text
  qwen-plus
  ```

- `retrieval_k`
  每次检索返回的片段数量，默认 5。

- `chunk_size`
  文档切块大小，默认 650 字符。

- `chunk_overlap`
  文档切块重叠长度，默认 90 字符。

其他模块通过下面方式使用配置：

```python
from .config import settings
```

## src/documents.py

位置：`D:\software\project\src\documents.py`

这是文档解析和切块模块。

主要职责：

- 读取 docx 文件。
- 从 docx 内部 XML 中提取文本。
- 将长文本切成适合向量检索的小片段。
- 为每个片段生成元数据。

核心数据结构：

- `Chunk`

```python
@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int]
```

字段说明：

- `id`：片段唯一 ID。
- `text`：片段正文。
- `metadata`：片段来源信息，包括文件名、文件路径、片段编号。

关键函数：

- `extract_docx_text(path: Path) -> str`

  读取 docx 文件正文。

  docx 本质上是一个 zip 包，正文在：

  ```text
  word/document.xml
  ```

  函数会从 XML 中提取 `<w:t>` 文本节点，并按段落组织成文本。

- `load_source_documents(source_dir: Path | None = None)`

  读取知识库目录下所有 `.docx` 文件。

  默认目录来自：

  ```python
  settings.kb_source_dir
  ```

  如果目录不存在，会抛出：

  ```python
  FileNotFoundError
  ```

- `split_text(text, chunk_size=None, overlap=None)`

  将长文本切成多个片段。

  主要切分依据：

  - 句号
  - 感叹号
  - 问号
  - 分号
  - 换行

  同时保留一定重叠内容，避免上下文被切断。

- `build_chunks(source_dir: Path | None = None)`

  完整构建入库片段。

  调用链路：

  ```text
  load_source_documents()
    ↓
  split_text()
    ↓
  Chunk(...)
  ```

输出会被 `src.kb.ingest()` 使用。

## src/embeddings.py

位置：`D:\software\project\src\embeddings.py`

这是本地 embedding 模块。

主要职责：

- 为 Chroma 提供一个无需外部模型、无需 API Key、无需下载模型的向量化函数。
- 支持中文文本的基础检索。
- 避免依赖 Chroma 默认的 ONNX embedding，因为当前机器上的 onnxruntime DLL 初始化不稳定。

核心类：

- `HashingChineseEmbedding`

它是一个轻量级哈希向量 embedding。

实现思路：

1. 清理文本空白。
2. 提取英文、数字、中文字符。
3. 生成中文 2-gram、3-gram、4-gram。
4. 用 `blake2b` 哈希将 token 映射到固定维度向量。
5. 对向量做归一化。

关键方法：

- `__init__(dimensions=768)`
  设置向量维度，默认 768。

- `name()`
  返回 embedding 函数名称，供 Chroma 识别。

- `build_from_config(config)`
  Chroma 需要的配置恢复方法。

- `get_config()`
  返回可序列化配置。

- `default_space()`
  返回向量空间类型：

  ```text
  cosine
  ```

- `__call__(input)`
  Chroma 调用 embedding 函数时会走这里。

- `_embed(text)`
  把单个字符串变成向量。

- `_tokens(text)`
  把文本拆成 token，包括中文单字和 n-gram。

注意：

这个 embedding 适合当前一期的静态知识库检索和本地验证。后续如果要提升召回质量，可以替换成更强的中文 embedding 模型，比如 DashScope embedding、bge-m3、text2vec 等。

## src/kb.py

位置：`D:\software\project\src\kb.py`

这是知识库和 Chroma 交互的核心模块。

主要职责：

- 创建或获取 Chroma collection。
- 将 docx 文档片段写入 Chroma。
- 根据用户问题检索相关知识片段。
- 封装 Chroma 的 Windows 兼容处理。

核心数据结构：

- `SearchHit`

```python
@dataclass(frozen=True)
class SearchHit:
    text: str
    source: str
    chunk: int
    distance: float | None = None
```

字段说明：

- `text`：命中的知识片段正文。
- `source`：来源 docx 文件名。
- `chunk`：来源片段编号。
- `distance`：Chroma 返回的距离值。

关键函数：

- `get_collection()`

  获取 Chroma collection。

  关键配置：

  ```python
  name=settings.chroma_collection
  embedding_function=HashingChineseEmbedding()
  ```

  也就是说，当前项目不使用 Chroma 默认 embedding，而是使用 `HashingChineseEmbedding`。

- `ingest(source_dir=None, reset=True)`

  重建或追加知识库。

  当 `reset=True` 时，会先删除旧 collection，再重新创建。

  调用链路：

  ```text
  build_chunks()
    ↓
  collection.add(...)
  ```

  返回结果包括：

  - collection 名称
  - 文档数量
  - chunk 数量
  - Chroma 存储路径

- `search(query, k=None)`

  根据问题检索知识库。

  如果 collection 为空，会自动调用：

  ```python
  ingest(reset=False)
  ```

  查询后将 Chroma 原始结果转换成 `SearchHit` 列表。

- `_import_chromadb()`

  Chroma 兼容处理函数。

  背景：

  当前项目使用的 Chroma 0.5 会在导入时初始化默认 ONNX embedding；但本机 onnxruntime DLL 初始化存在问题。项目实际不需要默认 ONNX embedding，所以这里提前注入一个很小的 `onnxruntime` stub，让 Chroma 可以正常导入。

  这段兼容逻辑只用于绕开默认 embedding 初始化，实际检索仍然使用 `HashingChineseEmbedding`。

- `_make_client(chromadb)`

  创建 Chroma `PersistentClient`。

  同时关闭匿名遥测，并把 Chroma 的 telemetry 实现切换到 `src.chroma_noop.NoopTelemetry`，避免日志刷屏。

## src/agent.py

位置：`D:\software\project\src\agent.py`

这是问答 Agent 模块。

主要职责：

- 接收用户问题。
- 调用知识库检索。
- 根据关键词做简单频道推荐。
- 如果配置了 `DASHSCOPE_API_KEY`，调用通义千问生成自然语言回答。
- 如果没有配置 Key 或模型调用失败，返回基于检索片段的兜底回答。

核心配置：

- `CHANNEL_RULES`

这是一个简单关键词规则表，用于判断应该推荐哪个频道或路径。

当前覆盖：

- 发布与入驻路径
- 项目需求频道
- 项目供应频道
- 法律服务频道
- 金融供应频道
- 物流/仓储供需频道

关键函数：

- `detect_recommendations(question)`

  根据关键词匹配推荐卡片。

  例如：

  - 用户问“怎么发布项目需求”
    推荐：发布与入驻路径

  - 用户问“我想找项目机会”
    推荐：项目需求频道

  - 用户问“跨境光伏项目怎么找法律服务”
    推荐：法律服务频道

- `answer(question)`

  Agent 主入口。

  调用链路：

  ```text
  search(question)
    ↓
  _llm_answer(question, hits)
    ↓ 如果失败或未配置 Key
  _fallback_answer(question, hits)
  ```

  返回结构：

  ```python
  {
      "answer": "...",
      "recommendations": [...],
      "sources": [...]
  }
  ```

- `_llm_answer(question, hits)`

  如果环境变量中存在 `DASHSCOPE_API_KEY`，尝试使用：

  ```python
  ChatTongyi(model=settings.llm_model)
  ```

  它会把检索到的知识库片段放进 prompt，要求模型：

  - 只基于知识库回答
  - 不编造企业、项目、资质、案例或交易结果
  - 无依据时明确说明
  - 给出可操作下一步

  如果没有 Key、依赖不可用或调用失败，返回 `None`。

- `_fallback_answer(question, hits)`

  兜底回答逻辑。

  不调用大模型，直接把检索到的前三个片段整理出来，并附上来源。

  这个设计保证了：

  - 没有 API Key 也能跑通
  - Chroma 检索能力可以独立验证
  - 用户至少能看到知识库依据

- `_compact(text, max_len)`

  压缩长文本片段，避免兜底回答过长。

## src/chroma_noop.py

位置：`D:\software\project\src\chroma_noop.py`

这是 Chroma telemetry 的空实现。

主要职责：

- 禁用 Chroma 的产品遥测事件发送。
- 避免本地运行时出现 posthog 相关 warning。

核心类：

- `NoopTelemetry`

继承：

```python
ProductTelemetryClient
```

核心方法：

```python
def capture(self, event: ProductTelemetryEvent) -> None:
    return None
```

这个类在 `src.kb._make_client()` 里被使用：

```python
chroma_product_telemetry_impl="src.chroma_noop.NoopTelemetry"
```

注意：

这里的 `@override` 装饰器是必须的，因为 Chroma 依赖的 `overrides` 包会检查子类方法是否显式声明覆盖父类方法。

## 文件之间的依赖关系

```text
app.py
  ├─ src.agent.answer
  └─ src.kb.ingest / src.kb.search

scripts/ingest.py
  └─ src.kb.ingest

src.agent
  ├─ src.kb.search
  └─ src.config.settings

src.kb
  ├─ src.config.settings
  ├─ src.documents.build_chunks
  ├─ src.embeddings.HashingChineseEmbedding
  └─ src.chroma_noop.NoopTelemetry

src.documents
  └─ src.config.settings

src.embeddings
  └─ Python 标准库 hashlib / math / re

src.config
  └─ .env / 环境变量
```

## 常用运行命令

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

健康检查：

```text
GET http://127.0.0.1:8000/api/health
```

聊天接口：

```text
POST http://127.0.0.1:8000/api/chat
```

请求体示例：

```json
{
  "question": "跨境光伏项目怎么找法律服务？"
}
```

## 当前版本边界

当前版本是一期最小可用版本，重点是静态知识库问答和基础频道推荐。

已经具备：

- FastAPI 服务
- 简单网页聊天入口
- docx 知识库读取
- Chroma 持久化检索
- 本地 hashing embedding
- 通义千问回答增强
- 无 API Key 兜底回答
- 简单关键词频道推荐

暂未实现：

- 用户登录和权限控制
- 后台知识库管理页面
- 图片识别和多模态
- 真实项目/企业动态数据推荐
- 复杂多轮状态管理
- 埋点统计后台
- 可配置的运营模板和规则管理

