"""
Graph RAG检索与推理引擎
实现向量检索、图结构检索、知识推理的融合检索系统
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

# 假设的依赖库
from neo4j import GraphDatabase
from pymilvus import Collection
import openai


@dataclass
class QueryIntent:
    """查询意图"""
    original_query: str
    intent_type: str  # recommendation, planning, information, comparison
    entities: List[str]
    constraints: Dict
    rewritten_query: str
    query_vector: List[float] = None


@dataclass
class RetrievalResult:
    """检索结果"""
    entity_id: str
    entity_type: str
    name: str
    properties: Dict
    score: float
    source: str  # vector, graph, reasoning
    metadata: Dict = field(default_factory=dict)


@dataclass
class GraphContext:
    """图上下文"""
    nodes: List[Dict]
    relations: List[Dict]
    paths: List[Dict]
    subgraph: Dict


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.intent_patterns = {
            "recommendation": [
                "推荐", "建议", "适合", "值得去", "好玩",
                "recommend", "suggest", "worth visiting"
            ],
            "planning": [
                "路线", "行程", "规划", "怎么走", "游玩顺序",
                "route", "itinerary", "plan", "how to go"
            ],
            "information": [
                "介绍", "是什么", "有什么", "特色", "历史",
                "introduce", "what is", "feature", "history"
            ],
            "comparison": [
                "对比", "比较", "哪个好", "区别", "不同",
                "compare", "difference", "which is better"
            ]
        }
    
    def classify(self, query: str) -> str:
        """分类查询意图"""
        query_lower = query.lower()
        
        scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            scores[intent] = score
        
        # 返回得分最高的意图
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        return "information"  # 默认意图


class EntityRecognizer:
    """实体识别器"""
    
    def __init__(self):
        self.entity_patterns = {
            "ScenicSpot": [
                r'([\u4e00-\u9fa5]{2,}(景区|景点|公园|博物馆|山|湖|河|瀑布|峡谷|古镇|古城))',
                r'([\u4e00-\u9fa5]{2,}(博物院|纪念馆|故居|遗址|名胜))'
            ],
            "CulturalHeritage": [
                r'([\u4e00-\u9fa5]{2,}(文化遗产|非遗|传统技艺))'
            ],
            "Event": [
                r'([\u4e00-\u9fa5]{2,}(节|展|会|赛|活动))'
            ]
        }
    
    def recognize(self, query: str) -> List[Tuple[str, str]]:
        """识别查询中的实体"""
        import re
        
        entities = []
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, query)
                for match in matches:
                    entities.append((match.group(1), entity_type))
        
        return entities


class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def rewrite(self, query: str, context: Dict) -> str:
        """改写查询以提升检索效果"""
        # 使用LLM改写查询
        prompt = f"""
        请改写以下旅游相关查询，使其更适合检索：
        原查询：{query}
        上下文：{context}
        
        要求：
        1. 补充缺失的关键信息
        2. 扩展同义词和相关概念
        3. 保持原意不变
        
        改写后的查询：
        """
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # 如果LLM调用失败，返回原查询
            return query


class VectorRetriever:
    """向量检索器"""
    
    def __init__(self, milvus_collection: Collection, embedding_model):
        self.collection = milvus_collection
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(__name__)
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 20,
        filters: Dict = None
    ) -> List[RetrievalResult]:
        """向量相似度检索"""
        try:
            # 构建过滤表达式
            filter_expr = self._build_filter_expr(filters) if filters else None
            
            # Milvus检索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 执行检索
            results = self.collection.search(
                data=[query_vector],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["name", "entity_type", "properties"]
            )
            
            # 格式化结果
            retrieval_results = []
            for hits in results:
                for hit in hits:
                    result = RetrievalResult(
                        entity_id=str(hit.id),
                        entity_type=hit.entity.get("entity_type", "Unknown"),
                        name=hit.entity.get("name", ""),
                        properties=hit.entity.get("properties", {}),
                        score=hit.score,
                        source="vector"
                    )
                    retrieval_results.append(result)
            
            return retrieval_results
        
        except Exception as e:
            self.logger.error(f"向量检索失败: {e}")
            return []
    
    def _build_filter_expr(self, filters: Dict) -> str:
        """构建Milvus过滤表达式"""
        expr_parts = []
        for key, value in filters.items():
            if isinstance(value, list):
                expr_parts.append(f"{key} in {value}")
            elif isinstance(value, str):
                expr_parts.append(f'{key} == "{value}"')
            else:
                expr_parts.append(f"{key} == {value}")
        return " and ".join(expr_parts)


class GraphRetriever:
    """图结构检索器"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger(__name__)
    
    async def search(
        self,
        entities: List[Tuple[str, str]],
        intent: str,
        max_depth: int = 2
    ) -> List[RetrievalResult]:
        """图结构检索"""
        try:
            with self.driver.session() as session:
                # 根据意图选择查询策略
                if intent == "recommendation":
                    return await self._search_recommendation(session, entities)
                elif intent == "planning":
                    return await self._search_planning(session, entities)
                elif intent == "information":
                    return await self._search_information(session, entities, max_depth)
                else:
                    return await self._search_general(session, entities)
        
        except Exception as e:
            self.logger.error(f"图检索失败: {e}")
            return []
    
    async def _search_recommendation(
        self,
        session,
        entities: List[Tuple[str, str]]
    ) -> List[RetrievalResult]:
        """推荐查询"""
        query = """
            MATCH (s:ScenicSpot)
            WHERE s.name IN $entity_names
            OPTIONAL MATCH (s)-[:SIMILAR_TO]->(similar:ScenicSpot)
            OPTIONAL MATCH (s)-[:HAS_THEME]->(theme:Theme)<-[:HAS_THEME]-(themed:ScenicSpot)
            WITH s, collect(DISTINCT similar) as similar_spots, 
                 collect(DISTINCT themed) as themed_spots
            UNWIND similar_spots + themed_spots as related
            WITH DISTINCT related as spot, 
                 count(s) as relevance_score
            RETURN spot.name as name,
                   labels(spot)[0] as entity_type,
                   spot as properties,
                   relevance_score * 0.3 as score
            ORDER BY score DESC
            LIMIT 10
        """
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="graph"
            )
            for record in result
        ]
    
    async def _search_planning(
        self,
        session,
        entities: List[Tuple[str, str]]
    ) -> List[RetrievalResult]:
        """路线规划查询"""
        query = """
            MATCH (r:TravelRoute)-[:PASSES_THROUGH]->(s:ScenicSpot)
            WHERE s.name IN $entity_names
            WITH r, count(DISTINCT s) as matched_count,
                 collect(DISTINCT s.name) as matched_spots
            ORDER BY matched_count DESC
            LIMIT 5
            MATCH (r)-[p:PASSES_THROUGH]->(all_s:ScenicSpot)
            RETURN r.name as name,
                   'TravelRoute' as entity_type,
                   r as properties,
                   matched_count as score,
                   collect({spot: all_s.name, day: p.day, order: p.order}) as itinerary
        """
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="graph",
                metadata={"itinerary": record["itinerary"]}
            )
            for record in result
        ]
    
    async def _search_information(
        self,
        session,
        entities: List[Tuple[str, str]],
        max_depth: int
    ) -> List[RetrievalResult]:
        """信息查询"""
        query = """
            MATCH path = (n)-[r*1..{}]-(m)
            WHERE n.name IN $entity_names
            AND labels(m) IN [['ScenicSpot'], ['CulturalHeritage'], ['Event']]
            WITH m, length(path) as depth, 
                 collect(DISTINCT nodes(path)) as path_nodes
            RETURN m.name as name,
                   labels(m)[0] as entity_type,
                   m as properties,
                   1.0 / depth as score,
                   path_nodes as context_paths
            ORDER BY score DESC
            LIMIT 10
        """.format(max_depth)
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="graph",
                metadata={"context_paths": record["context_paths"]}
            )
            for record in result
        ]
    
    async def _search_general(
        self,
        session,
        entities: List[Tuple[str, str]]
    ) -> List[RetrievalResult]:
        """通用查询"""
        query = """
            MATCH (n)
            WHERE n.name IN $entity_names
            RETURN n.name as name,
                   labels(n)[0] as entity_type,
                   n as properties,
                   1.0 as score
        """
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="graph"
            )
            for record in result
        ]
    
    async def get_graph_context(
        self,
        entity_ids: List[str],
        max_depth: int = 2
    ) -> GraphContext:
        """获取图上下文"""
        query = """
            MATCH path = (n)-[r*1..{}]-(m)
            WHERE n.id IN $entity_ids
            WITH nodes(path) as path_nodes, 
                 relationships(path) as path_rels,
                 path
            WITH collect(DISTINCT path_nodes) as all_nodes,
                 collect(DISTINCT path_rels) as all_rels,
                 collect(DISTINCT path) as all_paths
            RETURN all_nodes, all_rels, all_paths
        """.format(max_depth)
        
        with self.driver.session() as session:
            result = session.run(query, entity_ids=entity_ids)
            record = result.single()
            
            if record:
                return GraphContext(
                    nodes=record["all_nodes"],
                    relations=record["all_rels"],
                    paths=record["all_paths"],
                    subgraph={}
                )
            
            return GraphContext(nodes=[], relations=[], paths=[], subgraph={})


class KnowledgeReasoner:
    """知识推理引擎"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger(__name__)
    
    async def infer(
        self,
        entities: List[Tuple[str, str]],
        intent: str
    ) -> List[RetrievalResult]:
        """知识推理"""
        try:
            with self.driver.session() as session:
                # 执行推理规则
                if intent == "recommendation":
                    return await self._infer_recommendations(session, entities)
                else:
                    return await self._infer_associations(session, entities)
        
        except Exception as e:
            self.logger.error(f"知识推理失败: {e}")
            return []
    
    async def _infer_recommendations(
        self,
        session,
        entities: List[Tuple[str, str]]
    ) -> List[RetrievalResult]:
        """推理推荐"""
        # 基于用户历史行为和相似度推理
        query = """
            // 假设用户喜欢某些景点，推理相似景点
            MATCH (liked:ScenicSpot)
            WHERE liked.name IN $entity_names
            
            // 找到相似的景点
            MATCH (liked)-[s:SIMILAR_TO]->(similar:ScenicSpot)
            WHERE s.similarity > 0.6
            
            // 找到相同主题的景点
            OPTIONAL MATCH (liked)-[:HAS_THEME]->(theme:Theme)<-[:HAS_THEME]-(themed:ScenicSpot)
            WHERE themed <> liked
            
            WITH similar, themed, s.similarity as sim_score
            WITH COALESCE(similar, themed) as recommended,
                 COALESCE(sim_score, 0.5) as score
            
            RETURN DISTINCT recommended.name as name,
                   'ScenicSpot' as entity_type,
                   recommended as properties,
                   score as score
            ORDER BY score DESC
            LIMIT 10
        """
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="reasoning"
            )
            for record in result
        ]
    
    async def _infer_associations(
        self,
        session,
        entities: List[Tuple[str, str]]
    ) -> List[RetrievalResult]:
        """推理关联"""
        # 推理实体间的隐含关联
        query = """
            MATCH (e1)-[r1]->(common)<-[r2]-(e2)
            WHERE e1.name IN $entity_names
            AND e2.name IN $entity_names
            AND e1 <> e2
            WITH common, count(DISTINCT e1) as association_strength
            WHERE association_strength > 0
            RETURN common.name as name,
                   labels(common)[0] as entity_type,
                   common as properties,
                   association_strength * 0.2 as score
            ORDER BY score DESC
            LIMIT 5
        """
        
        entity_names = [name for name, _ in entities]
        result = session.run(query, entity_names=entity_names)
        
        return [
            RetrievalResult(
                entity_id=record["properties"].get("id", ""),
                entity_type=record["entity_type"],
                name=record["name"],
                properties=dict(record["properties"]),
                score=record["score"],
                source="reasoning"
            )
            for record in result
        ]


class ResultFusion:
    """结果融合器"""
    
    def __init__(self):
        self.weights = {
            "vector": 0.4,
            "graph": 0.4,
            "reasoning": 0.2
        }
    
    def merge(
        self,
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        reasoning_results: List[RetrievalResult],
        top_k: int = 10
    ) -> List[RetrievalResult]:
        """融合多路检索结果"""
        # 按实体ID聚合分数
        entity_scores = defaultdict(lambda: {"scores": {}, "result": None})
        
        # 向量检索结果
        for result in vector_results:
            entity_scores[result.entity_id]["scores"]["vector"] = result.score
            entity_scores[result.entity_id]["result"] = result
        
        # 图检索结果
        for result in graph_results:
            entity_scores[result.entity_id]["scores"]["graph"] = result.score
            if not entity_scores[result.entity_id]["result"]:
                entity_scores[result.entity_id]["result"] = result
        
        # 推理结果
        for result in reasoning_results:
            entity_scores[result.entity_id]["scores"]["reasoning"] = result.score
            if not entity_scores[result.entity_id]["result"]:
                entity_scores[result.entity_id]["result"] = result
        
        # 计算融合分数
        merged_results = []
        for entity_id, data in entity_scores.items():
            scores = data["scores"]
            result = data["result"]
            
            # 加权融合
            fused_score = (
                scores.get("vector", 0) * self.weights["vector"] +
                scores.get("graph", 0) * self.weights["graph"] +
                scores.get("reasoning", 0) * self.weights["reasoning"]
            )
            
            # 更新分数
            result.score = fused_score
            result.source = "fusion"
            merged_results.append(result)
        
        # 排序并返回Top-K
        merged_results.sort(key=lambda x: x.score, reverse=True)
        return merged_results[:top_k]


class GraphRAGEngine:
    """Graph RAG检索引擎"""
    
    def __init__(
        self,
        neo4j_driver,
        milvus_collection: Collection,
        embedding_model,
        llm_client
    ):
        # 初始化各组件
        self.intent_classifier = IntentClassifier()
        self.entity_recognizer = EntityRecognizer()
        self.query_rewriter = QueryRewriter(llm_client)
        
        self.vector_retriever = VectorRetriever(milvus_collection, embedding_model)
        self.graph_retriever = GraphRetriever(neo4j_driver)
        self.knowledge_reasoner = KnowledgeReasoner(neo4j_driver)
        
        self.result_fusion = ResultFusion()
        
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        
        self.logger = logging.getLogger(__name__)
    
    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        context: Dict = None
    ) -> List[RetrievalResult]:
        """执行Graph RAG检索"""
        self.logger.info(f"开始检索: {query}")
        
        # 1. 查询理解
        query_intent = await self._understand_query(query, context or {})
        
        # 2. 并行执行多路检索
        vector_task = self.vector_retriever.search(
            query_intent.query_vector,
            top_k=top_k * 2
        )
        
        graph_task = self.graph_retriever.search(
            query_intent.entities,
            query_intent.intent_type
        )
        
        reasoning_task = self.knowledge_reasoner.infer(
            query_intent.entities,
            query_intent.intent_type
        )
        
        # 等待所有检索完成
        vector_results, graph_results, reasoning_results = await asyncio.gather(
            vector_task, graph_task, reasoning_task
        )
        
        # 3. 结果融合
        merged_results = self.result_fusion.merge(
            vector_results,
            graph_results,
            reasoning_results,
            top_k=top_k
        )
        
        self.logger.info(f"检索完成: {len(merged_results)}个结果")
        return merged_results
    
    async def _understand_query(self, query: str, context: Dict) -> QueryIntent:
        """理解查询意图"""
        # 意图分类
        intent = self.intent_classifier.classify(query)
        
        # 实体识别
        entities = self.entity_recognizer.recognize(query)
        
        # 查询改写
        rewritten_query = await self.query_rewriter.rewrite(query, context)
        
        # 生成查询向量
        query_vector = await self._generate_embedding(rewritten_query)
        
        return QueryIntent(
            original_query=query,
            intent_type=intent,
            entities=entities,
            constraints={},
            rewritten_query=rewritten_query,
            query_vector=query_vector
        )
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """生成文本embedding"""
        try:
            # 使用OpenAI embedding
            response = await self.llm_client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"生成embedding失败: {e}")
            # 返回零向量作为fallback
            return [0.0] * 1536
    
    async def generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult]
    ) -> str:
        """生成答案"""
        # 构建上下文
        context = self._build_context(retrieval_results)
        
        # 使用LLM生成答案
        prompt = f"""
        基于以下知识图谱检索结果回答用户问题：
        
        用户问题：{query}
        
        检索到的相关信息：
        {context}
        
        请提供准确、详细的回答，并引用相关信息来源：
        """
        
        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"生成答案失败: {e}")
            return "抱歉，无法生成答案。"
    
    def _build_context(self, results: List[RetrievalResult]) -> str:
        """构建上下文"""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"{i}. {result.name} ({result.entity_type})\n"
                f"   相关度: {result.score:.3f}\n"
                f"   属性: {result.properties}\n"
            )
        
        return "\n".join(context_parts)


# 使用示例
async def main():
    """示例：使用Graph RAG引擎"""
    
    # 初始化引擎（需要实际的连接）
    # engine = GraphRAGEngine(neo4j_driver, milvus_collection, embedding_model, llm_client)
    
    # 执行检索
    # results = await engine.retrieve("推荐一些北京的历史文化景点")
    
    # 生成答案
    # answer = await engine.generate_answer("推荐一些北京的历史文化景点", results)
    
    print("Graph RAG引擎示例")


if __name__ == "__main__":
    asyncio.run(main())
