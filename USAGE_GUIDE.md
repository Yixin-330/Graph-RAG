# 文旅Graph RAG系统使用指南

## 📖 目录
1. [环境准备](#环境准备)
2. [系统部署](#系统部署)
3. [数据初始化](#数据初始化)
4. [功能使用](#功能使用)
5. [集成到现有系统](#集成到现有系统)
6. [常见问题](#常见问题)

---

## 一、环境准备

### 1.1 硬件要求
```
最低配置：
- CPU: 4核
- 内存: 16GB
- 硬盘: 100GB SSD

推荐配置：
- CPU: 8核+
- 内存: 32GB+
- 硬盘: 500GB SSD
```

### 1.2 软件环境
```bash
# Python环境
Python 3.8+ (推荐3.10)

# Docker环境（用于部署数据库）
Docker 20.0+
Docker Compose 2.0+
```

### 1.3 安装Python依赖
```bash
# 创建虚拟环境
python -m venv tourism_env
source tourism_env/bin/activate  # Linux/Mac
# 或
tourism_env\\Scripts\\activate  # Windows

# 安装核心依赖
pip install neo4j==5.15.0
pip install pymilvus==2.3.3
pip install redis==5.0.0
pip install openai==1.6.0
pip install transformers==4.36.0
pip install jieba==0.42.1
pip install torch==2.1.0
pip install fastapi==0.109.0
pip install uvicorn==0.25.0
```

---

## 二、系统部署

### 2.1 部署Neo4j图数据库

#### 方式一：Docker部署（推荐）
```bash
# 创建数据目录
mkdir -p ./data/neo4j

# 启动Neo4j
docker run -d \\
  --name tourism_neo4j \\
  -p 7474:7474 \\
  -p 7687:7687 \\
  -v $(pwd)/data/neo4j:/data \\
  -e NEO4J_AUTH=neo4j/your_password \\
  -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \\
  -e NEO4J_dbms_memory_heap_initial__size=512m \\
  -e NEO4J_dbms_memory_heap_max__size=2G \\
  neo4j:5.15.0

# 验证部署
# 浏览器访问 http://localhost:7474
# 用户名: neo4j, 密码: your_password
```

#### 方式二：本地安装
```bash
# 下载Neo4j Community Edition
# https://neo4j.com/download/

# 配置neo4j.conf
dbms.memory.heap.initial_size=512m
dbms.memory.heap.max_size=2G
dbms.security.procedures.unrestricted=apoc.*,gds.*

# 启动服务
./bin/neo4j console
```

### 2.2 部署Milvus向量数据库

#### Docker Compose部署
```bash
# 创建docker-compose.yml
cat > docker-compose-milvus.yml <<EOF
version: '3.8'
services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ./data/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ./data/minio:/minio_data
    command: minio server /minio_data
    ports:
      - "9000:9000"

  milvus:
    image: milvusdb/milvus:v2.3.3
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./data/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
EOF

# 启动Milvus
docker-compose -f docker-compose-milvus.yml up -d

# 验证部署
curl http://localhost:9091/healthz
```

### 2.3 部署Redis缓存
```bash
docker run -d \\
  --name tourism_redis \\
  -p 6379:6379 \\
  -v $(pwd)/data/redis:/data \\
  redis:7.0-alpine \\
  redis-server --appendonly yes
```

### 2.4 配置OpenAI API
```bash
# 设置环境变量
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 或您的代理地址
```

---

## 三、数据初始化

### 3.1 初始化知识图谱Schema

创建初始化脚本 `init_knowledge_graph.py`：

```python
import asyncio
from neo4j import GraphDatabase
from knowledge_graph_ontology import TourismOntology

async def init_schema():
    # 连接Neo4j
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    # 创建本体
    ontology = TourismOntology()
    
    # 生成Schema语句
    schema_cypher = ontology.generate_neo4j_schema()
    
    # 执行Schema创建
    with driver.session() as session:
        # 分批执行（避免一次性执行过多语句）
        statements = schema_cypher.split(';')
        for stmt in statements:
            if stmt.strip():
                try:
                    session.run(stmt)
                    print(f"执行成功: {stmt[:50]}...")
                except Exception as e:
                    print(f"执行失败: {e}")
    
    driver.close()
    print("知识图谱Schema初始化完成！")

if __name__ == "__main__":
    asyncio.run(init_schema())
```

执行初始化：
```bash
python init_knowledge_graph.py
```

### 3.2 初始化Milvus向量集合

创建 `init_milvus.py`：

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def init_milvus():
    # 连接Milvus
    connections.connect("default", host="localhost", port="19530")
    
    # 定义集合Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
    ]
    
    schema = CollectionSchema(fields=fields, description="文旅知识向量库")
    
    # 创建集合
    collection = Collection(name="tourism_knowledge", schema=schema)
    
    # 创建向量索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    
    # 加载集合
    collection.load()
    
    print("Milvus向量集合初始化完成！")

if __name__ == "__main__":
    init_milvus()
```

执行初始化：
```bash
python init_milvus.py
```

### 3.3 导入初始数据

#### 方式一：从结构化数据导入

创建 `import_data.py`：

```python
import asyncio
import json
from neo4j import GraphDatabase
from knowledge_extraction_pipeline import KnowledgeExtractionPipeline, DataSource

async def import_sample_data():
    # 准备示例数据
    sample_spots = [
        {
            "id": "spot_001",
            "name": "故宫博物院",
            "level": "5A",
            "category": ["历史文化", "博物馆"],
            "description": "故宫是中国明清两代的皇家宫殿，位于北京中轴线的中心...",
            "location": {
                "province": "北京市",
                "city": "北京市",
                "district": "东城区",
                "address": "北京市东城区景山前街4号"
            },
            "ticket": {"price": 60.0},
            "rating": 4.8
        },
        {
            "id": "spot_002",
            "name": "长城",
            "level": "5A",
            "category": ["历史文化", "世界遗产"],
            "description": "长城是中国古代的军事防御工程...",
            "location": {
                "province": "北京市",
                "city": "北京市",
                "district": "延庆区"
            },
            "rating": 4.9
        }
    ]
    
    # 连接Neo4j
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    # 导入数据
    with driver.session() as session:
        for spot in sample_spots:
            query = """
                MERGE (s:ScenicSpot {id: $id})
                SET s += $properties,
                    s.updated_at = datetime()
            """
            session.run(query, id=spot["id"], properties=spot)
            print(f"导入景点: {spot['name']}")
    
    driver.close()
    print("数据导入完成！")

if __name__ == "__main__":
    asyncio.run(import_sample_data())
```

#### 方式二：从文本数据抽取

```python
import asyncio
from knowledge_extraction_pipeline import KnowledgeExtractionPipeline, DataSource

async def extract_from_text():
    # 创建Pipeline
    pipeline = KnowledgeExtractionPipeline(neo4j_driver)
    
    # 添加数据源
    data_sources = [
        DataSource(
            source_type="file",
            source_url="data/tourism_text.txt",
            data_format="text",
            update_frequency="monthly",
            priority=1
        )
    ]
    
    for source in data_sources:
        pipeline.collector.add_source(source)
    
    # 运行抽取
    result = await pipeline.run(data_sources)
    
    print(f"抽取完成: {result['entities_count']}个实体, {result['relations_count']}个关系")

if __name__ == "__main__":
    asyncio.run(extract_from_text())
```

---

## 四、功能使用

### 4.1 启动Graph RAG检索引擎

创建 `start_rag_engine.py`：

```python
import asyncio
from neo4j import GraphDatabase
from pymilvus import connections, Collection
from openai import AsyncOpenAI
from graph_rag_engine import GraphRAGEngine

async def main():
    # 初始化连接
    neo4j_driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    connections.connect("default", host="localhost", port="19530")
    milvus_collection = Collection("tourism_knowledge")
    
    llm_client = AsyncOpenAI()
    
    # 创建Graph RAG引擎
    rag_engine = GraphRAGEngine(
        neo4j_driver=neo4j_driver,
        milvus_collection=milvus_collection,
        embedding_model=None,  # 使用OpenAI embedding
        llm_client=llm_client
    )
    
    # 执行检索
    query = "推荐一些北京的历史文化景点"
    results = await rag_engine.retrieve(query, top_k=5)
    
    # 生成答案
    answer = await rag_engine.generate_answer(query, results)
    
    print(f"查询: {query}")
    print(f"答案: {answer}")
    
    # 关闭连接
    neo4j_driver.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.2 启动数字人对话系统

创建 `start_digital_human.py`：

```python
import asyncio
from neo4j import GraphDatabase
from pymilvus import connections, Collection
from openai import AsyncOpenAI
from graph_rag_engine import GraphRAGEngine
from digital_human_integration import DigitalHumanDialogueSystem, DigitalHumanAPI

async def main():
    # 初始化组件
    neo4j_driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    connections.connect("default", host="localhost", port="19530")
    milvus_collection = Collection("tourism_knowledge")
    llm_client = AsyncOpenAI()
    
    # 创建Graph RAG引擎
    rag_engine = GraphRAGEngine(
        neo4j_driver=neo4j_driver,
        milvus_collection=milvus_collection,
        embedding_model=None,
        llm_client=llm_client
    )
    
    # 创建对话系统
    dialogue_system = DigitalHumanDialogueSystem(
        graph_rag_engine=rag_engine,
        neo4j_driver=neo4j_driver,
        llm_client=llm_client
    )
    
    # 创建API
    api = DigitalHumanAPI(dialogue_system)
    
    # 开始对话
    session_id = "session_001"
    user_id = "user_001"
    
    # 1. 开始会话
    start_response = await api.handle_request({
        "action": "start",
        "session_id": session_id,
        "user_id": user_id
    })
    print(f"数字人: {start_response['text']}")
    
    # 2. 发送消息
    message_response = await api.handle_request({
        "action": "message",
        "session_id": session_id,
        "message": "我想了解故宫的历史"
    })
    print(f"数字人: {message_response['text']}")
    
    # 3. 继续对话
    message_response = await api.handle_request({
        "action": "message",
        "session_id": session_id,
        "message": "故宫门票多少钱？"
    })
    print(f"数字人: {message_response['text']}")
    
    # 4. 结束会话
    end_response = await api.handle_request({
        "action": "end",
        "session_id": session_id
    })
    print(f"数字人: {end_response['text']}")
    
    neo4j_driver.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 启动Web API服务

创建 `api_server.py`：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
from neo4j import GraphDatabase
from pymilvus import connections, Collection
from openai import AsyncOpenAI
from graph_rag_engine import GraphRAGEngine
from digital_human_integration import DigitalHumanDialogueSystem, DigitalHumanAPI

# 初始化FastAPI
app = FastAPI(title="文旅数字人API")

# 全局变量
api_handler = None

class ChatRequest(BaseModel):
    action: str
    session_id: str
    user_id: str = "anonymous"
    message: str = None

@app.on_event("startup")
async def startup():
    global api_handler
    
    # 初始化连接
    neo4j_driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "your_password")
    )
    
    connections.connect("default", host="localhost", port="19530")
    milvus_collection = Collection("tourism_knowledge")
    llm_client = AsyncOpenAI()
    
    # 创建引擎和API
    rag_engine = GraphRAGEngine(
        neo4j_driver=neo4j_driver,
        milvus_collection=milvus_collection,
        embedding_model=None,
        llm_client=llm_client
    )
    
    dialogue_system = DigitalHumanDialogueSystem(
        graph_rag_engine=rag_engine,
        neo4j_driver=neo4j_driver,
        llm_client=llm_client
    )
    
    api_handler = DigitalHumanAPI(dialogue_system)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = await api_handler.handle_request(request.dict())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

启动服务：
```bash
python api_server.py
```

访问API文档：http://localhost:8000/docs

---

## 五、集成到现有系统

### 5.1 前端集成示例

```javascript
// 前端调用示例
class DigitalHumanClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
        this.sessionId = null;
    }
    
    async startSession(userId) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                action: 'start',
                session_id: this.generateSessionId(),
                user_id: userId
            })
        });
        
        const data = await response.json();
        this.sessionId = data.session_id;
        return data;
    }
    
    async sendMessage(message) {
        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                action: 'message',
                session_id: this.sessionId,
                message: message
            })
        });
        
        return await response.json();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}

// 使用示例
const client = new DigitalHumanClient('http://localhost:8000');

// 开始对话
const startResponse = await client.startSession('user_001');
console.log(startResponse.text);

// 发送消息
const messageResponse = await client.sendMessage('推荐一些北京的历史景点');
console.log(messageResponse.text);
```

### 5.2 移动端集成（Android示例）

```kotlin
// Android Kotlin示例
class DigitalHumanService {
    private val client = OkHttpClient()
    private val baseUrl = "http://your-server:8000"
    private var sessionId: String? = null
    
    suspend fun startSession(userId: String): JSONObject {
        val request = JSONObject()
        request.put("action", "start")
        request.put("session_id", generateSessionId())
        request.put("user_id", userId)
        
        val response = postRequest("$baseUrl/chat", request)
        sessionId = response.getString("session_id")
        return response
    }
    
    suspend fun sendMessage(message: String): JSONObject {
        val request = JSONObject()
        request.put("action", "message")
        request.put("session_id", sessionId)
        request.put("message", message)
        
        return postRequest("$baseUrl/chat", request)
    }
    
    private fun generateSessionId(): String {
        return "session_${System.currentTimeMillis()}_${UUID.randomUUID()}"
    }
    
    private suspend fun postRequest(url: String, data: JSONObject): JSONObject {
        // 实现HTTP POST请求
        // ...
    }
}
```

---

## 六、常见问题

### Q1: Neo4j连接失败
```bash
# 检查Neo4j是否运行
docker ps | grep neo4j

# 检查端口是否开放
telnet localhost 7687

# 查看Neo4j日志
docker logs tourism_neo4j
```

### Q2: Milvus检索速度慢
```python
# 优化索引参数
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",  # 改用IVF_FLAT
    "params": {"nlist": 1024}
}

# 调整搜索参数
search_params = {
    "metric_type": "COSINE",
    "params": {"nprobe": 10}  # 减少nprobe提升速度
}
```

### Q3: OpenAI API调用失败
```python
# 使用代理或国内API
from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="your_key",
    base_url="https://your-proxy.com/v1"  # 设置代理
)
```

### Q4: 内存不足
```bash
# 调整Neo4j内存
docker update tourism_neo4j --memory=4g --memory-swap=4g

# 调整Milvus内存
# 修改docker-compose-milvus.yml中的内存限制
```

### Q5: 数据更新后检索结果未变化
```python
# 重新生成embedding并更新
async def update_embedding(entity_id, new_text):
    # 生成新embedding
    embedding = await generate_embedding(new_text)
    
    # 更新Milvus
    collection.upsert([{
        "id": entity_id,
        "embedding": embedding,
        "text": new_text
    }])
    
    # 更新Neo4j
    query = """
        MATCH (n {id: $id})
        SET n.embedding = $embedding
    """
    session.run(query, id=entity_id, embedding=embedding)
```

---

## 七、性能优化建议

### 7.1 数据库优化
```cypher
// Neo4j优化
// 1. 定期执行数据统计
CALL db.stats.retrieve('GRAPH COUNTS')

// 2. 重建索引
CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()

// 3. 清理孤立节点
MATCH (n) WHERE NOT (n)--() DELETE n
```

### 7.2 缓存策略
```python
# 使用Redis缓存热门查询
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

async def cached_search(query):
    # 检查缓存
    cache_key = f"search:{query}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # 执行检索
    results = await rag_engine.retrieve(query)
    
    # 缓存结果（5分钟）
    redis_client.setex(cache_key, 300, json.dumps(results))
    
    return results
```

### 7.3 批量处理
```python
# 批量导入数据
async def batch_import(entities, batch_size=100):
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i+batch_size]
        await import_batch(batch)
        print(f"已导入 {i+len(batch)}/{len(entities)}")
```

---

## 八、监控与运维

### 8.1 健康检查脚本
```python
import asyncio
import requests
from neo4j import GraphDatabase

async def health_check():
    # 检查Neo4j
    try:
        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Neo4j: 正常")
    except:
        print("❌ Neo4j: 异常")
    
    # 检查Milvus
    try:
        response = requests.get("http://localhost:9091/healthz")
        if response.status_code == 200:
            print("✅ Milvus: 正常")
    except:
        print("❌ Milvus: 异常")
    
    # 检查Redis
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("✅ Redis: 正常")
    except:
        print("❌ Redis: 异常")

if __name__ == "__main__":
    asyncio.run(health_check())
```

### 8.2 日志配置
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tourism_rag.log'),
        logging.StreamHandler()
    ]
)
```

---

## 九、下一步建议

1. **数据准备**：收集文旅领域数据（景区信息、文化遗产、旅游攻略等）
2. **模型微调**：针对文旅场景微调embedding模型
3. **性能测试**：进行压力测试，优化系统性能
4. **安全加固**：添加认证、授权、数据加密
5. **监控部署**：部署Prometheus、Grafana监控系统

---

**技术支持**：如有问题，请参考README.md或提交Issue。
