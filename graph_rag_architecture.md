# Graph RAG数据库架构设计方案

## 一、技术栈选型与对比

### 1. 图数据库选型

#### Neo4j（推荐方案）
**优势：**
- 成熟的Cypher查询语言，语法直观
- 内置向量索引支持（5.11+版本）
- 丰富的图算法库（APOC、GDS）
- 完善的Python/Java客户端
- 企业级性能与稳定性

**适用场景：**
- 复杂关系查询与推理
- 实时推荐系统
- 知识图谱可视化

**部署方案：**
```yaml
# docker-compose.yml
version: '3.8'
services:
  neo4j:
    image: neo4j:5.15.0
    container_name: tourism_neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/password
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_dbms_memory_heap_initial__size: "512m"
      NEO4J_dbms_memory_heap_max__size: "2G"
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
```

#### Apache Age（备选方案）
**优势：**
- 基于PostgreSQL，运维成本低
- 支持SQL+Cypher混合查询
- 开源免费

**劣势：**
- 生态不如Neo4j成熟
- 向量检索需额外集成

### 2. 向量数据库选型

#### Milvus（推荐方案）
**优势：**
- 高性能分布式架构
- 支持多种索引类型（IVF、HNSW）
- 丰富的相似度度量
- 云原生设计

**部署方案：**
```yaml
# docker-compose.yml
services:
  milvus-standalone:
    image: milvusdb/milvus:v2.3.3
    container_name: tourism_milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus/data:/var/lib/milvus
```

#### Qdrant（轻量级方案）
**优势：**
- 单机部署简单
- Rust实现，性能优异
- 内置过滤功能

### 3. Embedding模型选型

#### 文本Embedding
- **OpenAI text-embedding-3-large**：1536维，质量最优
- **BGE-large-zh**：1024维，中文优化
- **M3E-base**：768维，轻量级方案

#### 多模态Embedding
- **CLIP**：图文联合embedding
- **Chinese-CLIP**：中文图文embedding

## 二、Graph RAG融合架构设计

### 1. 整体架构图
```
┌─────────────────────────────────────────────────────────┐
│                    用户查询层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  自然语言查询  │  │  多轮对话     │  │  个性化推荐   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  查询理解与路由层                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  意图识别     │  │  实体抽取     │  │  查询改写     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  混合检索引擎层                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Graph RAG检索策略                         │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐ │  │
│  │  │ 向量检索    │  │ 图结构检索  │  │ 知识推理    │ │  │
│  │  │ (Milvus)   │  │ (Neo4j)    │  │ (GDS)      │ │  │
│  │  └────────────┘  └────────────┘  └────────────┘ │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  结果融合与排序层                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  多路召回融合  │  │  相关性排序   │  │  多样性重排   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  答案生成层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  上下文构建   │  │  LLM生成     │  │  答案优化     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2. 核心组件设计

#### 查询理解模块
```python
class QueryUnderstanding:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.query_rewriter = QueryRewriter()
    
    def process(self, query: str, context: dict) -> QueryIntent:
        # 1. 意图识别
        intent = self.intent_classifier.predict(query)
        
        # 2. 实体抽取
        entities = self.entity_extractor.extract(query)
        
        # 3. 查询改写
        rewritten_query = self.query_rewriter.rewrite(query, context)
        
        return QueryIntent(
            original_query=query,
            intent=intent,
            entities=entities,
            rewritten_query=rewritten_query
        )
```

#### 混合检索引擎
```python
class HybridRetriever:
    def __init__(self, neo4j_driver, milvus_collection):
        self.graph_retriever = GraphRetriever(neo4j_driver)
        self.vector_retriever = VectorRetriever(milvus_collection)
        self.knowledge_reasoner = KnowledgeReasoner(neo4j_driver)
    
    def retrieve(self, query_intent: QueryIntent, top_k: int = 10):
        # 1. 向量检索
        vector_results = self.vector_retriever.search(
            query_intent.rewritten_query,
            top_k=top_k * 2
        )
        
        # 2. 图结构检索
        graph_results = self.graph_retriever.search(
            query_intent.entities,
            query_intent.intent
        )
        
        # 3. 知识推理
        reasoning_results = self.knowledge_reasoner.infer(
            query_intent.entities,
            query_intent.intent
        )
        
        # 4. 结果融合
        merged_results = self.merge_results(
            vector_results,
            graph_results,
            reasoning_results
        )
        
        return merged_results[:top_k]
```

#### 向量检索实现
```python
class VectorRetriever:
    def __init__(self, milvus_collection, embedding_model):
        self.collection = milvus_collection
        self.embedding_model = embedding_model
    
    def search(self, query: str, top_k: int = 20, filters: dict = None):
        # 生成查询向量
        query_vector = self.embedding_model.encode(query)
        
        # Milvus检索
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=self._build_filter_expr(filters) if filters else None
        )
        
        return [self._format_result(r) for r in results[0]]
    
    def _build_filter_expr(self, filters: dict) -> str:
        # 构建Milvus过滤表达式
        expr_parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                expr_parts.append(f"{key} in {value}")
            else:
                expr_parts.append(f"{key} == {value}")
        return " and ".join(expr_parts)
```

#### 图结构检索实现
```python
class GraphRetriever:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def search(self, entities: List[Entity], intent: str) -> List[GraphResult]:
        with self.driver.session() as session:
            # 根据意图选择查询模板
            query_template = self._get_query_template(intent)
            
            # 执行图查询
            result = session.run(
                query_template,
                entities=entities
            )
            
            return [self._format_record(record) for record in result]
    
    def _get_query_template(self, intent: str) -> str:
        templates = {
            "spot_recommend": """
                MATCH (s:ScenicSpot)
                WHERE s.name IN $entities OR s.category IN $entities
                OPTIONAL MATCH (s)-[:NEARBY]->(nearby:ScenicSpot)
                OPTIONAL MATCH (s)-[:HAS_THEME]->(theme:Theme)
                RETURN s, collect(DISTINCT nearby) as nearby_spots, 
                       collect(DISTINCT theme) as themes
            """,
            "route_planning": """
                MATCH (r:TravelRoute)-[:PASSES_THROUGH]->(s:ScenicSpot)
                WHERE s.name IN $entities
                WITH r, count(DISTINCT s) as matched_count
                ORDER BY matched_count DESC
                LIMIT 10
                MATCH (r)-[p:PASSES_THROUGH]->(s:ScenicSpot)
                RETURN r, collect({spot: s, order: p.order}) as route_spots
            """,
            "cultural_query": """
                MATCH (c:CulturalHeritage)-[:RELATED_TO]->(s:ScenicSpot)
                WHERE c.name IN $entities OR c.era IN $entities
                OPTIONAL MATCH (c)-[:RELATED_TO]->(related:ScenicSpot)
                RETURN c, s, collect(DISTINCT related) as related_spots
            """
        }
        return templates.get(intent, templates["spot_recommend"])
```

#### 知识推理引擎
```python
class KnowledgeReasoner:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
    
    def infer(self, entities: List[Entity], intent: str) -> List[ReasoningResult]:
        with self.driver.session() as session:
            # 1. 相似实体推理
            similar_entities = self._find_similar_entities(session, entities)
            
            # 2. 关联路径推理
            association_paths = self._find_associations(session, entities)
            
            # 3. 推荐推理
            recommendations = self._infer_recommendations(session, entities, intent)
            
            return ReasoningResult(
                similar_entities=similar_entities,
                association_paths=association_paths,
                recommendations=recommendations
            )
    
    def _find_similar_entities(self, session, entities: List[Entity]):
        # 使用图算法计算相似度
        query = """
            CALL gds.nodeSimilarity.stream('myGraph')
            YIELD node1, node2, similarity
            WHERE gds.util.asNode(node1).name IN $entities
            AND similarity > 0.5
            RETURN gds.util.asNode(node1).name AS entity1,
                   gds.util.asNode(node2).name AS entity2,
                   similarity
            ORDER BY similarity DESC
        """
        return session.run(query, entities=[e.name for e in entities]).data()
```

### 3. 结果融合策略

#### 多路召回融合
```python
class ResultFusion:
    def __init__(self):
        self.weights = {
            "vector": 0.4,
            "graph": 0.4,
            "reasoning": 0.2
        }
    
    def merge_results(self, vector_results, graph_results, reasoning_results):
        # 1. 实体对齐
        aligned_results = self._align_entities(
            vector_results, graph_results, reasoning_results
        )
        
        # 2. 分数融合
        merged_scores = {}
        for entity_id, scores in aligned_results.items():
            merged_scores[entity_id] = (
                scores.get("vector", 0) * self.weights["vector"] +
                scores.get("graph", 0) * self.weights["graph"] +
                scores.get("reasoning", 0) * self.weights["reasoning"]
            )
        
        # 3. 排序
        sorted_entities = sorted(
            merged_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_entities
```

## 三、数据同步与更新策略

### 1. 双写策略
```python
class DataSyncManager:
    def __init__(self, neo4j_driver, milvus_collection):
        self.neo4j = neo4j_driver
        self.milvus = milvus_collection
    
    async def insert_entity(self, entity: Entity):
        # 1. 写入Neo4j
        node_id = await self._insert_to_neo4j(entity)
        
        # 2. 写入Milvus
        vector_id = await self._insert_to_milvus(entity)
        
        # 3. 维护ID映射
        await self._update_id_mapping(node_id, vector_id)
    
    async def _insert_to_neo4j(self, entity: Entity):
        query = f"""
            CREATE (n:{entity.label} $properties)
            RETURN id(n) as node_id
        """
        with self.neo4j.session() as session:
            result = session.run(query, properties=entity.properties)
            return result.single()["node_id"]
    
    async def _insert_to_milvus(self, entity: Entity):
        # 生成embedding
        embedding = self.embedding_model.encode(entity.text)
        
        # 插入Milvus
        data = [{
            "id": entity.id,
            "embedding": embedding,
            "text": entity.text,
            "metadata": entity.metadata
        }]
        
        self.milvus.insert(data)
        return entity.id
```

### 2. 增量更新机制
```python
class IncrementalUpdater:
    def __init__(self, neo4j_driver, milvus_collection):
        self.neo4j = neo4j_driver
        self.milvus = milvus_collection
    
    async def update_hot_data(self):
        """每日更新热度数据"""
        # 1. 计算新的热度分数
        popularity_scores = await self._calculate_popularity()
        
        # 2. 批量更新Neo4j
        query = """
            UNWIND $updates as update
            MATCH (s:ScenicSpot {id: update.id})
            SET s.popularity = update.popularity,
                s.updated_at = datetime()
        """
        
        with self.neo4j.session() as session:
            session.run(query, updates=popularity_scores)
    
    async def update_embeddings(self, entity_ids: List[str]):
        """更新指定实体的embedding"""
        # 1. 获取实体文本
        entities = await self._get_entity_texts(entity_ids)
        
        # 2. 重新生成embedding
        for entity in entities:
            new_embedding = self.embedding_model.encode(entity.text)
            
            # 3. 更新Milvus
            self.milvus.upsert([{
                "id": entity.id,
                "embedding": new_embedding
            }])
```

## 四、性能优化策略

### 1. 缓存策略
```python
class CacheManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.cache_ttl = {
            "hot_spots": 3600,      # 热门景点缓存1小时
            "user_profile": 86400,  # 用户画像缓存1天
            "query_result": 300     # 查询结果缓存5分钟
        }
    
    async def get_cached_result(self, query_hash: str):
        cached = await self.redis.get(f"query:{query_hash}")
        if cached:
            return json.loads(cached)
        return None
    
    async def cache_result(self, query_hash: str, result: dict):
        await self.redis.setex(
            f"query:{query_hash}",
            self.cache_ttl["query_result"],
            json.dumps(result)
        )
```

### 2. 查询优化
- **Neo4j查询优化**：
  - 使用PROFILE分析查询计划
  - 合理使用索引
  - 避免笛卡尔积
  
- **Milvus检索优化**：
  - 选择合适的索引类型（HNSW适合高精度，IVF适合高吞吐）
  - 调整nprobe参数平衡精度与速度
  - 使用分区减少搜索空间

## 五、监控与运维

### 1. 性能监控指标
```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "query_latency": [],
            "retrieval_recall": [],
            "graph_traversal_depth": [],
            "cache_hit_rate": []
        }
    
    def record_query_metrics(self, query_result: QueryResult):
        self.metrics["query_latency"].append(query_result.latency)
        self.metrics["retrieval_recall"].append(query_result.recall)
        
        # 上报到监控系统
        self._report_to_prometheus(query_result)
```

### 2. 健康检查
```python
class HealthChecker:
    async def check_system_health(self) -> HealthStatus:
        neo4j_health = await self._check_neo4j()
        milvus_health = await self._check_milvus()
        redis_health = await self._check_redis()
        
        return HealthStatus(
            neo4j=neo4j_health,
            milvus=milvus_health,
            redis=redis_health,
            overall=all([neo4j_health, milvus_health, redis_health])
        )
```

## 六、部署架构建议

### 生产环境部署
```yaml
# kubernetes部署示例
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graph-rag-service
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: graph-rag-api
        image: tourism-graph-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
        - name: MILVUS_HOST
          value: "milvus-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

这套架构设计实现了：
1. **图结构与向量检索的深度融合**
2. **多路召回与智能融合**
3. **知识推理能力**
4. **高性能与可扩展性**
5. **完善的监控运维体系**
