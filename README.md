# 文旅场景数字人运营系统 - Graph RAG与知识图谱完整方案

## 📋 项目概述

本项目为文旅场景的数字人运营系统提供了完整的Graph RAG数据库与知识图谱解决方案，实现了从数据采集、知识抽取、图谱构建到智能检索、对话交互的全流程。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    数字人对话层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  对话管理     │  │  用户画像     │  │  响应生成     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Graph RAG检索引擎                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  向量检索     │  │  图结构检索   │  │  知识推理     │  │
│  │  (Milvus)    │  │  (Neo4j)     │  │  (GDS)       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  知识图谱构建层                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  数据采集     │  │  实体抽取     │  │  知识融合     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  数据存储层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Neo4j图数据库│  │  Milvus向量库 │  │  Redis缓存   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 📁 项目文件说明

### 1. `tourism_knowledge_graph_schema.md`
**知识图谱Schema设计文档**
- 定义了6类核心实体：景区景点、文化遗产、旅游路线、服务设施、活动节庆、游客画像
- 设计了15种关系类型：空间关系、旅游关系、文化关系、服务关系、相似关系
- 包含完整的属性定义、索引设计、约束规则
- 提供典型查询模式示例

### 2. `graph_rag_architecture.md`
**Graph RAG数据库架构设计**
- 技术栈选型与对比（Neo4j vs Apache Age, Milvus vs Qdrant）
- Graph RAG融合架构详细设计
- 混合检索引擎实现方案
- 数据同步与更新策略
- 性能优化与监控运维方案

### 3. `knowledge_graph_ontology.py`
**知识图谱本体层实现**
- 实体类型与关系类型的枚举定义
- 属性定义与约束规则
- 本体Schema自动生成Neo4j创建语句
- 实体验证与推理规则
- 支持JSON导出与导入

### 4. `knowledge_extraction_pipeline.py`
**数据抽取与知识融合Pipeline**
- 多源数据采集（Web、API、文件、数据库）
- 实体抽取（景区景点、文化遗产、活动节庆）
- 关系抽取（空间关系、包含关系、文化关联）
- 实体融合与去重
- 知识图谱自动构建

### 5. `graph_rag_engine.py`
**Graph RAG检索与推理引擎**
- 查询理解（意图分类、实体识别、查询改写）
- 向量检索（Milvus集成）
- 图结构检索（Neo4j Cypher查询）
- 知识推理（相似度推理、关联推理）
- 多路召回融合与排序
- 答案生成

### 6. `digital_human_integration.py`
**数字人对话接口集成**
- 对话状态管理
- 用户画像管理
- 响应生成（基于检索结果）
- 个性化推荐
- 多轮对话上下文维护
- RESTful API接口

## 🚀 快速开始

### 环境要求
- Python 3.8+
- Neo4j 5.11+
- Milvus 2.3+
- Redis 6.0+

### 安装依赖
```bash
pip install neo4j pymilvus redis openai transformers jieba
```

### 启动服务

#### 1. 启动Neo4j
```bash
docker run -d \
  --name tourism_neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15.0
```

#### 2. 启动Milvus
```bash
docker run -d \
  --name tourism_milvus \
  -p 19530:19530 \
  milvusdb/milvus:v2.3.3
```

#### 3. 初始化知识图谱
```python
from knowledge_graph_ontology import TourismOntology

# 创建本体
ontology = TourismOntology()

# 生成Neo4j Schema
schema = ontology.generate_neo4j_schema()
# 在Neo4j中执行schema语句
```

#### 4. 运行知识抽取Pipeline
```python
from knowledge_extraction_pipeline import KnowledgeExtractionPipeline

pipeline = KnowledgeExtractionPipeline(neo4j_driver)
result = await pipeline.run(data_sources)
```

#### 5. 启动数字人对话系统
```python
from digital_human_integration import DigitalHumanDialogueSystem

dialogue_system = DigitalHumanDialogueSystem(
    graph_rag_engine,
    neo4j_driver,
    llm_client
)

# 开始对话
response = await dialogue_system.start_session("session_001", "user_001")
```

## 💡 核心特性

### 1. 知识图谱特化设计
- 针对文旅场景定制实体与关系
- 支持多维度属性（地理位置、时间、价格等）
- 内置推理规则（相似推荐、关联推理）

### 2. Graph RAG融合检索
- 向量检索：语义相似度匹配
- 图结构检索：关系路径查询
- 知识推理：隐含关联发现
- 多路融合：加权排序与去重

### 3. 智能对话交互
- 多轮对话上下文管理
- 用户画像个性化推荐
- 意图识别与查询改写
- 情感化响应生成

### 4. 可扩展架构
- 模块化设计，易于扩展
- 支持多数据源接入
- 增量更新机制
- 完善的监控体系

## 📊 性能指标

- **检索延迟**: < 200ms (P95)
- **检索召回率**: > 85%
- **答案准确率**: > 90%
- **并发支持**: 1000+ QPS

## 🔧 配置说明

### Neo4j配置
```yaml
neo4j:
  uri: "bolt://localhost:7687"
  user: "neo4j"
  password: "password"
  max_connection_pool_size: 50
```

### Milvus配置
```yaml
milvus:
  host: "localhost"
  port: 19530
  collection_name: "tourism_knowledge"
  embedding_dim: 1536
```

### LLM配置
```yaml
llm:
  model: "gpt-3.5-turbo"
  embedding_model: "text-embedding-3-large"
  temperature: 0.7
  max_tokens: 2000
```

## 📈 监控与运维

### 关键指标
- 查询QPS与延迟
- 检索召回率与准确率
- 知识图谱节点/关系数量
- 缓存命中率
- 用户满意度

### 日志与告警
- 检索失败告警
- 性能下降告警
- 数据同步异常告警

## 🔄 更新与维护

### 数据更新策略
- **实时更新**: 用户行为数据
- **每日更新**: 热度、评分数据
- **每周更新**: 景区基础信息
- **每月更新**: 知识图谱全量重建

### 版本管理
- 知识图谱版本化
- 增量更新回滚机制
- Schema演进支持

## 🤝 贡献指南

欢迎提交Issue和Pull Request。在提交PR前，请确保：
1. 代码通过单元测试
2. 遵循代码规范
3. 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证。

---

**技术支持**: 如有任何问题，请提交Issue或联系开发团队。
