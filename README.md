# 文旅场景数字人运营系统 - Graph RAG与知识图谱完整方案

## 📋 项目概述

本项目为**双服务架构**的统一底层平台，基于 Graph RAG 与知识图谱技术：

| 服务 | 说明 | 状态 |
|------|------|:----:|
| 🏛️ **文旅数字人对话** | 景区推荐、路线规划、文化问答、数字人交互 | 原有 |
| ✂️ **非遗剪纸智能生成** | 关键词驱动、图谱约束、AI 图像生成 | ✅ 新增 |

两者共享 `CulturalHeritage` 文化遗产实体作为领域桥梁，通过统一的 Graph RAG 架构实现知识检索与推理。

---

## 🏗️ 系统全景架构

```
                    用户/前端
                        │
              ┌─────────┴──────────┐
              │  FastAPI 服务层      │  papercut_api_service.py
              │  POST /generate      │
              │  GET  /patterns      │
              └─────────┬──────────┘
                        │
     ┌──────────────────┼──────────────────┐
     │                  │                   │
     ▼                  ▼                   ▼
┌────────────┐  ┌──────────────┐  ┌──────────────────┐
│ 文旅数字人   │  │ 非遗剪纸约束   │  │ 图像生成管线     │
│ 对话系统    │  │ Graph RAG    │  │ SD/ControlNet    │
│ digital_   │  │ papercut_    │  │ Mock/Local/API   │
│ human_     │  │ graph_rag_   │  │ papercut_image_  │
│ integra-   │  │ engine.py    │  │ generator.py     │
│ tion.py    │  └──────┬───────┘  └──────────────────┘
└──────┬─────┘         │
       │               │ 结构化约束参数
       │     ┌─────────┴──────────┐
       │     │ 知识抽取管道        │
       │     │ 断点续传 6 阶段     │
       │     │ papercut_extrac-   │
       │     │ tion_pipeline.py   │
       │     └─────────┬──────────┘
       │               │
       ▼               ▼
┌───────────────────────────────────────────┐
│              知识图谱本体层                 │
│  ┌────────────────┐ ┌──────────────────┐  │
│  │  文旅本体        │ │  剪纸本体          │  │
│  │  7 实体 / 10 关系│ │  8 实体 / 15 关系  │  │
│  │  knowledge_     │ │  papercut_       │  │
│  │  graph_ontology │ │  knowledge_      │  │
│  │  .py            │ │  ontology.py     │  │
│  └────────────────┘ └──────────────────┘  │
│  ┌────────────────────────────────────┐   │
│  │  共享文化遗产桥接 CulturalHeritage  │   │
│  └────────────────────────────────────┘   │
└───────────────────────────────────────────┘
                        │
               ┌────────┴────────┐
               │   数据存储层     │
               │ Neo4j / Milvus  │
               └─────────────────┘
```

---

## 📁 项目文件说明

### 🏛️ 文旅数字人系统（原有，保持不动）

| 文件 | 说明 |
|------|------|
| `knowledge_graph_ontology.py` | 文旅本体：7 实体(景区/文化遗产/路线等) + 10 关系 + Neo4j Schema |
| `knowledge_extraction_pipeline.py` | 文旅数据抽取 Pipeline |
| `graph_rag_engine.py` | 文旅 Graph RAG 检索与推理引擎（向量+图+推理三路融合） |
| `digital_human_integration.py` | 数字人对话接口（对话管理、用户画像、响应生成） |
| `graph_rag_architecture.md` | Graph RAG 架构设计文档 |
| `tourism_knowledge_graph_schema.md` | 知识图谱 Schema 设计文档 |
| `ARCHITECTURE_EXPLAINED.md` | 架构详解 |

### ✂️ 非遗剪纸生成系统（新增）

| 文件 | 说明 | 代码量 |
|------|------|:------:|
| `papercut_knowledge_ontology.py` | **剪纸知识图谱本体** — 8 实体 + 15 关系 + 5 条推理规则 + 52 条 Neo4j Schema | ~850 行 |
| `papercut_seed_data.py` | **内置种子数据** — 14 纹样 + 12 母题 + 7 技法 + 6 流派 + 8 象征 + 6 传承人 + 91 条关系映射 | ~900 行 |
| `papercut_extraction_pipeline.py` | **断点续传抽取管道** — 6 阶段 + 状态持久化 + 数据源可插拔 | ~890 行 |
| `papercut_graph_rag_engine.py` | **Graph RAG 约束生成引擎** — 查询分析 → 三路检索 → 6 维推理 → 结构化参数 | ~1350 行 |
| `papercut_image_generator.py` | **图像生成管线** — PromptBuilder + 三后端(Mock/Local/API) + 后处理 + 结果管理 | ~1100 行 |
| `papercut_api_service.py` | **RESTful API 服务** — FastAPI 7 端点 + Swagger 文档 | ~590 行 |
| `papercut_data_sources.json` | **数据源配置** — 可插拔数据源 JSON 配置 | — |

---

## 🚀 快速开始

### 环境要求

**文旅部分：**
- Python 3.8+
- Neo4j 5.11+ / Milvus 2.3+

**剪纸部分新增依赖：**
```bash
pip install fastapi uvicorn python-multipart  # API 服务
pip install pillow                              # 图像处理（Mock 模式必需）
pip install torch diffusers transformers        # 本地 SD 生成（可选）
pip install replicate                           # Replicate API（可选）
```

### 剪纸服务快速启动

#### 1. 运行知识抽取管道（加载种子数据）

```bash
# 完整运行
python papercut_extraction_pipeline.py

# 中断后恢复
python papercut_extraction_pipeline.py --resume

# 输出: pipeline_output/entities_*.json, relations_*.json
```

#### 2. CLI 约束生成测试

```bash
# 单次查询
python papercut_graph_rag_engine.py "婚庆"

# JSON 输出
python papercut_graph_rag_engine.py "龙年吉祥" -j

# 交互模式
python papercut_graph_rag_engine.py

# 详细调试
python papercut_graph_rag_engine.py "花开富贵" -v
```

#### 3. CLI 图像生成（Mock 模式）

```bash
# Mock 模式（不需 GPU，生成占位图）
python papercut_image_generator.py "花开富贵" --backend mock

# 批量生成
python papercut_image_generator.py "春节福字" -n 4 --backend mock
```

#### 4. 启动 API 服务

```bash
uvicorn papercut_api_service:app --host 0.0.0.0 --port 8000

# 打开浏览器访问:
#   http://localhost:8000/docs   — Swagger API 文档
#   http://localhost:8000/redoc  — ReDoc 文档

# API 测试:
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"keyword": "婚庆"}'
```

### 文旅服务快速启动

```bash
# 1. 启动 Neo4j
docker run -d --name tourism_neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15.0

# 2. 初始化知识图谱 Schema
python -c "
from knowledge_graph_ontology import TourismOntology
ontology = TourismOntology()
print(ontology.generate_neo4j_schema())
"

# 3. 启动数字人对话系统
python -c "
from digital_human_integration import DigitalHumanDialogueSystem
# dialogue_system = DigitalHumanDialogueSystem(graph_rag_engine, neo4j_driver, llm_client)
"
```

---

## 💡 核心特性

### 1. 双领域知识图谱

| 维度 | 文旅领域 | 剪纸领域 |
|------|---------|---------|
| 实体类型 | 7 种 | 8 种 |
| 关系类型 | 10 种 | 15 种 |
| 推理规则 | 4 条 | 5 条 |
| 核心实体 | ScenicSpot, CulturalHeritage | PaperCutPattern(30 属性) |
| 领域特色 | 景区等级/路线规划/用户画像 | 纹样母题/技法/流派/文化象征 |

**桥梁：** 两个图谱通过 `CulturalHeritage` 实体和跨域查询方法桥接，文旅侧的"文化遗产"与剪纸侧的"文化象征"语义互通。

### 2. Graph RAG 约束生成引擎

```
输入: "婚庆"
  → 查询分析 (意图+实体识别+同义词扩展)
  → 三路检索 (精确匹配/关键词索引/图遍历2跳)
  → 6 维推理 (母题/风格/技法/配色/构图/文化)
  → 输出结构化约束参数 -> 传给图像模型
```

典型输出：
```json
{
  "primary_motif": "龙凤呈祥",
  "secondary_motifs": ["龙纹", "凤纹", "云纹"],
  "style": "蔚县剪纸",
  "technique": "阴阳刻结合",
  "recommended_colors": ["大红", "金色"],
  "color_count": 2,
  "symmetry_type": "轴对称",
  "complexity": "复杂",
  "meaning": "婚姻美满、龙凤和鸣、幸福吉祥",
  "applicable_scenes": ["婚庆"],
  "generation_prompt": "中国传统剪纸，蔚县剪纸风格，阴阳刻结合技法..."
}
```

### 3. 图像生成管线

- **三后端切换**：Mock(测试) / LocalDiffusers(本地 GPU) / API(Replicate/SD WebUI)
- **后处理**：背景白化 + 边缘锐化 + 红白二值化 + 纸面纹理
- **批量生成**：多种子探索 + 网格图输出
- **结果管理**：图像 / 缩略图 / 元数据 JSON / GenerationRecord

### 4. 断点续传管道

- 6 阶段（加载种子→采集→实体→关系→融合→输出）
- 状态持久化到文件，`--resume` 参数恢复
- 数据源可插拔（JSON 配置，找到真实网站后启用即可）

### 5. RESTful API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/constraints` | POST | 仅获取约束参数 |
| `/api/v1/generate` | POST | 端到端生成（约束+图像） |
| `/api/v1/generate-batch` | POST | 批量多关键词生成 |
| `/api/v1/patterns` | GET | 纹样/母题查询 |
| `/api/v1/knowledge/stats` | GET | 知识库统计 |
| `/api/v1/records` | GET | 生成记录查询 |

---

## 📊 种子数据规模

| 类别 | 数量 | 说明 |
|------|:----:|------|
| **纹样** | 14 | 龙凤呈祥、福寿双全、年年有余、四君子、花开富贵等 |
| **母题** | 12 | 龙纹、凤纹、牡丹、云纹、蝙蝠纹、寿桃等 |
| **技法** | 7 | 阴刻、阳刻、阴阳刻结合、套色、染色等 |
| **地域流派** | 6 | 蔚县、扬州、陕北、佛山、漳浦、高密 |
| **文化象征** | 8 | 福、寿、喜、财、吉祥如意、辟邪等 |
| **传承人** | 6 | 王老赏、库淑兰、周淑英、张永寿等 |
| **材料工具** | 6 | 大红宣纸、刻刀、蜡盘等 |
| **关系** | 91 | 纹样↔母题/技法/流派/象征/材料的全连接 |

---

## 🔧 配置说明

### 数据源配置 `papercut_data_sources.json`

```json
{
  "sources": [
    {"id": "seed_builtin", "type": "seed", "enabled": true},
    {"id": "file_import",  "type": "file", "enabled": false, "path": "./data/papercut_import.json"},
    {"id": "web_crawl",   "type": "web",  "enabled": false, "url": "https://..."},
    {"id": "api_feed",    "type": "api",  "enabled": false, "url": "https://..."}
  ]
}
```

### 图像生成配置

```python
from papercut_image_generator import GeneratorConfig

config = GeneratorConfig(
    backend="api",           # "mock" | "local" | "api"
    output_dir="./output",
    api_endpoint="https://api.replicate.com/v1/",
    api_key="your-key",
    api_model="stability-ai/sdxl:39ed52f2...",
)
```

---

## 📈 后续规划

| 优先级 | 计划 | 状态 |
|:------:|------|:----:|
| 1 | 剪纸知识图谱本体定义 | ✅ |
| 2 | 断点续传抽取管道（含种子数据） | ✅ |
| 3 | Graph RAG 约束生成引擎 | ✅ |
| 4 | 图像生成管线对接 | ✅ |
| 5 | RESTful API 封装 | ✅ |
| 6 | 非遗传承人审核工作流 | ⏳ |
| 7 | 真实数据源接入（网站爬取） | ⏳ |
| 8 | 对接真实 SD / ControlNet 模型 | ⏳ |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request。在提交 PR 前，请确保：
1. 代码通过单元测试
2. 遵循代码规范
3. 更新相关文档

剪纸领域知识欢迎非遗传承人、民俗学者参与审核和补充。

---

## 📄 许可证

本项目采用 MIT 许可证。

---

**技术支持**: 如有任何问题，请提交 Issue 或联系开发团队。
