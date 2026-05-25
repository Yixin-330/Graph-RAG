# Graph RAG系统代码架构详解

## 🏗️ 整体架构设计思路

### 核心设计理念

这套系统采用**分层架构**设计，从底层到上层依次为：

```
数据存储层 → 知识构建层 → 检索引擎层 → 对话交互层
```

每一层职责明确，层与层之间通过清晰的接口通信，便于维护和扩展。

---

## 一、数据存储层设计

### 1.1 为什么选择Neo4j + Milvus？

#### Neo4j（图数据库）
**解决的问题**：存储实体间的复杂关系

**选择理由**：
```python
# 传统关系型数据库查询多跳关系很复杂
# SQL: 需要多次JOIN，性能差
SELECT s1.name, s2.name, s3.name
FROM scenic_spots s1
JOIN relations r1 ON s1.id = r1.source
JOIN scenic_spots s2 ON r1.target = s2.id
JOIN relations r2 ON s2.id = r2.source
JOIN scenic_spots s3 ON r2.target = s3.id
WHERE s1.name = '故宫'

# Neo4j Cypher: 简洁直观
MATCH path = (s1:ScenicSpot {name: '故宫'})-[*1..3]->(s3)
RETURN path
```

**核心优势**：
- 关系查询性能优异（O(1)到O(log n)）
- 支持图算法（相似度、社区发现、路径规划）
- Cypher查询语言直观易懂

#### Milvus（向量数据库）
**解决的问题**：语义相似度检索

**选择理由**：
```python
# 传统数据库无法做语义检索
# 只能做精确匹配或模糊匹配
SELECT * FROM spots WHERE name LIKE '%故宫%'

# Milvus向量检索：理解语义
# "故宫博物院" ≈ "紫禁城" ≈ "明清皇宫"
query_vector = embed("明清皇宫")
results = milvus.search(query_vector, top_k=10)
```

**核心优势**：
- 支持高维向量高效检索（HNSW、IVF索引）
- 毫秒级响应
- 支持多种相似度度量（余弦、欧氏、内积）

### 1.2 双写策略设计

**核心挑战**：如何保持Neo4j和Milvus数据一致性？

**解决方案**：同步双写 + ID映射

```python
class DataSyncManager:
    async def insert_entity(self, entity: Entity):
        # 1. 写入Neo4j（主存储）
        node_id = await self._insert_to_neo4j(entity)
        
        # 2. 写入Milvus（向量索引）
        vector_id = await self._insert_to_milvus(entity)
        
        # 3. 维护ID映射关系
        await self._update_id_mapping(node_id, vector_id)
        
        # 4. 事务保证（失败回滚）
        if not node_id or not vector_id:
            await self._rollback(node_id, vector_id)
```

**为什么这样设计？**
- Neo4j作为主存储：存储完整实体信息
- Milvus作为索引：只存储向量+关键属性
- ID映射：快速关联两个存储系统

---

## 二、知识构建层设计

### 2.1 Pipeline架构

采用**责任链模式**，每个处理环节独立且可组合：

```python
数据采集 → 实体抽取 → 关系抽取 → 实体融合 → 图谱构建
```

**设计优势**：
- 每个环节可独立测试
- 可灵活替换某个环节的实现
- 支持断点续传和增量处理

### 2.2 实体抽取设计

#### 为什么用规则 + NLP混合方法？

**纯NLP方法的问题**：
```python
# 通用NER模型对领域实体识别效果差
# 例如："故宫博物院"可能被识别为ORG（组织）
# 但我们需要识别为ScenicSpot（景点）
```

**混合方法设计**：
```python
class EntityExtractor:
    def extract_scenic_spots(self, text: str):
        # 第一步：规则匹配（高精度）
        patterns = [
            r'([\u4e00-\u9fa5]{2,}(景区|景点|公园|博物馆))',
            r'([\u4e00-\u9fa5]{2,}(博物院|纪念馆|故居))'
        ]
        rule_entities = self._rule_extract(text, patterns)
        
        # 第二步：NLP补充（高召回）
        nlp_entities = self._nlp_extract(text)
        
        # 第三步：融合去重
        return self._merge(rule_entities, nlp_entities)
```

**为什么这样设计？**
- 规则匹配：精确度高，适合固定模式
- NLP补充：召回率高，发现规则遗漏的实体
- 融合去重：取长补短

### 2.3 知识融合设计

**核心问题**：同一实体可能有多个表述

```
"故宫博物院" = "故宫" = "紫禁城" = "Palace Museum"
```

**解决方案**：多级融合策略

```python
class EntityFusion:
    def fuse_entities(self, entities: List[Entity]):
        # 第一级：精确匹配
        groups = self._exact_match(entities)
        
        # 第二级：别名匹配（基于词典）
        groups = self._alias_match(groups)
        
        # 第三级：相似度匹配（基于embedding）
        groups = self._similarity_match(groups)
        
        # 合并每组实体
        return [self._merge_group(g) for g in groups]
```

**为什么多级融合？**
- 精确匹配：速度快，处理完全相同的实体
- 别名匹配：处理已知别名（如"故宫"="紫禁城"）
- 相似度匹配：发现未知别名，但计算成本高

---

## 三、检索引擎层设计

### 3.1 Graph RAG核心思想

**传统RAG的局限**：
```python
# 传统RAG：只做向量检索
query_vector = embed("故宫附近有什么景点")
results = milvus.search(query_vector)

# 问题：
# 1. 无法理解"附近"的空间关系
# 2. 无法推理隐含关联
# 3. 缺少结构化知识
```

**Graph RAG增强**：
```python
# Graph RAG：向量 + 图结构 + 推理
# 1. 向量检索：找到语义相关实体
vector_results = milvus.search(query_vector)

# 2. 图结构检索：基于关系扩展
graph_results = neo4j.query("""
    MATCH (s:ScenicSpot {name: '故宫'})-[:NEARBY]->(nearby)
    RETURN nearby
""")

# 3. 知识推理：发现隐含关联
reasoning_results = infer_relations(vector_results)

# 4. 多路融合
final_results = merge(vector_results, graph_results, reasoning_results)
```

### 3.2 查询理解设计

**为什么需要查询理解？**

用户查询往往是模糊的、不完整的：
```
"推荐一些好玩的" → 需要推断用户意图和偏好
"故宫门票" → 需要识别实体和属性需求
```

**设计实现**：
```python
class QueryUnderstanding:
    def process(self, query: str):
        # 1. 意图分类（推荐/查询/规划/比较）
        intent = self.intent_classifier.predict(query)
        
        # 2. 实体识别（景点名/地名/时间等）
        entities = self.entity_extractor.extract(query)
        
        # 3. 查询改写（补充缺失信息）
        rewritten = self.query_rewriter.rewrite(query, intent, entities)
        
        # 4. 向量化
        query_vector = self.embedding_model.encode(rewritten)
        
        return QueryIntent(intent, entities, rewritten, query_vector)
```

**查询改写示例**：
```
原查询: "推荐一些好玩的"
↓
改写后: "推荐一些热门的、评分高的、适合游客的景点"
（补充了"热门"、"评分高"等关键信息）
```

### 3.3 多路检索设计

**核心思想**：不同检索方式各有优劣，需要组合使用

```python
class HybridRetriever:
    async def retrieve(self, query_intent):
        # 并行执行三种检索
        tasks = [
            self.vector_retriever.search(query_intent.vector),  # 语义检索
            self.graph_retriever.search(query_intent.entities), # 结构检索
            self.knowledge_reasoner.infer(query_intent)         # 知识推理
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 融合排序
        return self.merge_and_rank(results)
```

**为什么并行执行？**
- 三种检索相互独立，无依赖关系
- 并行可显著降低延迟（从600ms降到200ms）

### 3.4 结果融合设计

**核心挑战**：如何合并不同来源的结果？

**解决方案**：加权融合 + 多样性重排

```python
class ResultFusion:
    def merge(self, vector_results, graph_results, reasoning_results):
        # 1. 实体对齐（同一实体可能出现在多个结果中）
        aligned = self.align_entities([
            vector_results,
            graph_results,
            reasoning_results
        ])
        
        # 2. 分数融合（加权平均）
        for entity_id, scores in aligned.items():
            final_score = (
                scores.get('vector', 0) * 0.4 +    # 向量检索权重
                scores.get('graph', 0) * 0.4 +    # 图检索权重
                scores.get('reasoning', 0) * 0.2  # 推理权重
            )
        
        # 3. 多样性重排（避免结果过于相似）
        return self.diversity_rerank(merged_results)
```

**为什么这样设计权重？**
- 向量和图检索同等重要：各占40%
- 推理作为补充：占20%
- 可根据实际效果调整权重

---

## 四、对话交互层设计

### 4.1 对话状态管理

**为什么需要状态管理？**

多轮对话需要维护上下文：
```
用户: "介绍一下故宫"
助手: "故宫是明清两代的皇宫..."
用户: "它门票多少钱？"  ← 需要知道"它"指故宫
```

**状态机设计**：
```python
class DialogueState(Enum):
    GREETING = "greeting"        # 初始问候
    QUERYING = "querying"        # 信息查询
    RECOMMENDING = "recommending" # 景点推荐
    PLANNING = "planning"        # 行程规划
    CLARIFYING = "clarifying"    # 澄清确认
    CLOSING = "closing"          # 结束对话
```

**状态转换逻辑**：
```python
def update_state(self, context, query, results):
    if not results:
        context.state = DialogueState.QUERYING
    
    elif results[0].type == "ScenicSpot":
        context.state = DialogueState.RECOMMENDING
    
    elif results[0].type == "TravelRoute":
        context.state = DialogueState.PLANNING
    
    # 根据状态生成不同风格的回复
    return self.generate_response(context)
```

### 4.2 用户画像设计

**核心作用**：实现个性化推荐

```python
class TouristProfile:
    # 静态属性
    age_group: str              # 年龄段
    travel_preference: List[str] # 偏好类型
    
    # 动态属性（从历史行为学习）
    history: List[Visit]         # 游览历史
    interests: List[str]         # 兴趣标签
    
    # 推理属性（计算得出）
    suitable_spots: List[str]    # 适合景点
```

**画像更新策略**：
```python
async def update_profile(self, user_id, interaction):
    # 1. 记录行为
    profile.history.append(interaction)
    
    # 2. 更新兴趣（从查询中提取）
    new_interests = extract_interests(interaction.query)
    profile.interests.update(new_interests)
    
    # 3. 重新计算推荐（异步）
    asyncio.create_task(
        recalculate_recommendations(user_id)
    )
```

### 4.3 响应生成设计

**核心原则**：基于检索结果生成，而非凭空编造

```python
class ResponseGenerator:
    async def generate(self, query, retrieval_results):
        # 1. 构建上下文（从检索结果）
        context = self.build_context(retrieval_results)
        
        # 2. 构建提示词
        prompt = f"""
        用户问题: {query}
        
        知识图谱信息:
        {context}
        
        要求：
        1. 基于提供的信息回答
        2. 信息不足时诚实告知
        3. 保持亲切专业的语气
        """
        
        # 3. 调用LLM生成
        response = await self.llm.generate(prompt)
        
        # 4. 后处理（添加建议问题等）
        return self.post_process(response, retrieval_results)
```

**为什么这样设计？**
- 确保回答有据可查（基于知识图谱）
- 避免LLM幻觉问题
- 保持回答的专业性和准确性

---

## 五、关键技术决策解析

### 5.1 为什么用异步编程？

**性能对比**：
```python
# 同步方式：串行执行
result1 = search_milvus(query)  # 100ms
result2 = search_neo4j(query)   # 100ms
result3 = inference(query)      # 100ms
# 总耗时: 300ms

# 异步方式：并行执行
result1, result2, result3 = await asyncio.gather(
    search_milvus(query),
    search_neo4j(query),
    inference(query)
)
# 总耗时: 100ms（最慢的那个）
```

**设计实现**：
```python
# 所有I/O操作都设计为异步
async def search(self, query):
    results = await self.milvus.search(query)
    return results

# CPU密集型操作保持同步
def merge_results(self, results):
    return sorted(results, key=lambda x: x.score)
```

### 5.2 为什么用DataClass？

**代码对比**：
```python
# 传统方式：冗长且易出错
class RetrievalResult:
    def __init__(self, entity_id, entity_type, name, score, source):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.name = name
        self.score = score
        self.source = source
    
    def __eq__(self, other):
        return self.entity_id == other.entity_id
    
    def __hash__(self):
        return hash(self.entity_id)

# DataClass：简洁且功能完整
@dataclass
class RetrievalResult:
    entity_id: str
    entity_type: str
    name: str
    score: float
    source: str
```

**优势**：
- 自动生成`__init__`、`__eq__`、`__hash__`等方法
- 类型提示清晰
- 代码可读性强

### 5.3 为什么用枚举类型？

**代码对比**：
```python
# 字符串方式：易出错
intent = "recommendation"
if intent == "recomendation":  # 拼写错误，难以发现
    pass

# 枚举方式：类型安全
class Intent(Enum):
    RECOMMENDATION = "recommendation"
    QUERY = "query"

intent = Intent.RECOMMENDATION
if intent == Intent.RECOMMENDATION:  # IDE会提示，编译时检查
    pass
```

---

## 六、性能优化设计

### 6.1 缓存策略

**多级缓存设计**：
```python
class CacheStrategy:
    # L1: 本地内存缓存（最快）
    local_cache = {}
    
    # L2: Redis缓存（中等）
    redis_cache = Redis()
    
    async def get(self, key):
        # 先查L1
        if key in self.local_cache:
            return self.local_cache[key]
        
        # 再查L2
        value = await self.redis_cache.get(key)
        if value:
            self.local_cache[key] = value
            return value
        
        # 最后查数据库
        value = await self.query_database(key)
        await self.redis_cache.set(key, value)
        self.local_cache[key] = value
        return value
```

### 6.2 批量处理

**为什么批量处理？**
```python
# 单条插入：频繁I/O，性能差
for entity in entities:
    await insert_one(entity)  # 每次都建立连接

# 批量插入：减少I/O，性能优
await insert_batch(entities)  # 一次连接插入多条
```

**实现**：
```python
async def batch_import(entities, batch_size=100):
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        await insert_batch(batch)
        print(f"进度: {i+len(batch)}/{len(entities)}")
```

---

## 七、可扩展性设计

### 7.1 插件化架构

**设计思想**：核心功能固定，扩展功能可插拔

```python
class GraphRAGEngine:
    def __init__(self):
        # 核心组件（必需）
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        
        # 扩展组件（可选）
        self.plugins = []
    
    def register_plugin(self, plugin):
        """注册扩展插件"""
        self.plugins.append(plugin)
    
    async def retrieve(self, query):
        # 核心检索
        results = await self._base_retrieve(query)
        
        # 应用插件
        for plugin in self.plugins:
            results = await plugin.process(results)
        
        return results
```

**使用示例**：
```python
# 添加个性化插件
engine.register_plugin(PersonalizationPlugin())

# 添加多样性插件
engine.register_plugin(DiversityPlugin())
```

### 7.2 配置化设计

**所有参数可配置**：
```yaml
# config.yaml
retrieval:
  vector_weight: 0.4
  graph_weight: 0.4
  reasoning_weight: 0.2
  top_k: 10

models:
  embedding: "text-embedding-3-large"
  llm: "gpt-3.5-turbo"

cache:
  enabled: true
  ttl: 300
```

**加载配置**：
```python
class Config:
    def __init__(self, config_file):
        with open(config_file) as f:
            self.config = yaml.load(f)
    
    def get(self, key, default=None):
        return self.config.get(key, default)

# 使用
config = Config("config.yaml")
top_k = config.get("retrieval.top_k", 10)
```

---

## 八、错误处理设计

### 8.1 优雅降级

**设计原则**：部分功能失败不影响整体服务

```python
async def retrieve(self, query):
    results = []
    
    # 向量检索失败不影响图检索
    try:
        vector_results = await self.vector_search(query)
        results.extend(vector_results)
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
    
    # 图检索继续执行
    try:
        graph_results = await self.graph_search(query)
        results.extend(graph_results)
    except Exception as e:
        logger.error(f"图检索失败: {e}")
    
    # 至少返回部分结果
    return results if results else self._default_results()
```

### 8.2 重试机制

**网络请求重试**：
```python
async def search_with_retry(self, query, max_retries=3):
    for i in range(max_retries):
        try:
            return await self.search(query)
        except Exception as e:
            if i == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** i)  # 指数退避
```

---

## 九、测试策略

### 9.1 单元测试

```python
def test_entity_extraction():
    extractor = EntityExtractor()
    text = "故宫博物院位于北京市东城区"
    
    entities = extractor.extract_scenic_spots(text)
    
    assert len(entities) == 1
    assert entities[0].name == "故宫博物院"
    assert entities[0].type == "ScenicSpot"
```

### 9.2 集成测试

```python
async def test_retrieval_pipeline():
    engine = GraphRAGEngine()
    
    results = await engine.retrieve("推荐北京景点")
    
    assert len(results) > 0
    assert all(r.score > 0 for r in results)
```

---

## 十、总结：设计原则

### 核心设计原则

1. **分层架构**：职责清晰，易于维护
2. **异步优先**：提升性能，降低延迟
3. **类型安全**：使用DataClass和枚举
4. **可扩展性**：插件化、配置化
5. **容错设计**：优雅降级、重试机制
6. **性能优化**：缓存、批量处理
7. **测试友好**：依赖注入、接口清晰

### 为什么这样设计？

**不是为了炫技，而是为了解决实际问题**：

- **性能问题**：异步+缓存+批量处理
- **可维护问题**：分层+类型安全+清晰接口
- **可扩展问题**：插件化+配置化
- **可靠性问题**：容错设计+优雅降级

这套架构经过深思熟虑，每个设计决策都有明确的理由和权衡。
