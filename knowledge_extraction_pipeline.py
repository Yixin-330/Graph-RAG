"""
文旅知识图谱数据抽取与知识融合Pipeline
实现从多源数据到知识图谱的自动化构建流程
"""

import asyncio
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# 假设的NLP模型接口
from transformers import AutoTokenizer, AutoModel
import jieba
import jieba.posseg as pseg


@dataclass
class DataSource:
    """数据源定义"""
    source_type: str  # web, api, file, database
    source_url: str
    data_format: str  # html, json, csv, xml
    update_frequency: str  # daily, weekly, monthly
    priority: int  # 数据优先级


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    entity_type: str
    name: str
    properties: Dict
    confidence: float
    source: str
    extraction_time: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    relation_type: str
    source_entity: str
    target_entity: str
    properties: Dict
    confidence: float
    source: str


class DataCollector:
    """数据采集器"""
    
    def __init__(self):
        self.sources = []
        self.logger = logging.getLogger(__name__)
    
    def add_source(self, source: DataSource):
        """添加数据源"""
        self.sources.append(source)
    
    async def collect_from_web(self, url: str) -> str:
        """从网页采集数据"""
        # 实际实现中会使用requests/aiohttp
        self.logger.info(f"从网页采集数据: {url}")
        # 模拟返回数据
        return "<html>...</html>"
    
    async def collect_from_api(self, api_url: str, params: Dict) -> Dict:
        """从API采集数据"""
        self.logger.info(f"从API采集数据: {api_url}")
        # 模拟返回数据
        return {"data": []}
    
    async def collect_from_file(self, file_path: str) -> str:
        """从文件采集数据"""
        self.logger.info(f"从文件采集数据: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    async def collect_all(self) -> Dict[str, any]:
        """采集所有数据源"""
        collected_data = {}
        
        for source in self.sources:
            try:
                if source.source_type == "web":
                    data = await self.collect_from_web(source.source_url)
                elif source.source_type == "api":
                    data = await self.collect_from_api(source.source_url, {})
                elif source.source_type == "file":
                    data = await self.collect_from_file(source.source_url)
                
                collected_data[source.source_url] = data
            except Exception as e:
                self.logger.error(f"采集数据失败 {source.source_url}: {e}")
        
        return collected_data


class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self, model_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # 加载NLP模型
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
        
        # 加载自定义词典
        self._load_custom_dict()
    
    def _load_custom_dict(self):
        """加载文旅领域自定义词典"""
        # 景区景点词典
        jieba.add_word("故宫博物院", tag="ns")
        jieba.add_word("长城", tag="ns")
        jieba.add_word("颐和园", tag="ns")
        
        # 文化遗产词典
        jieba.add_word("非物质文化遗产", tag="n")
        jieba.add_word("世界文化遗产", tag="n")
        
        # 旅游相关词典
        jieba.add_word("5A级景区", tag="n")
        jieba.add_word("门票价格", tag="n")
    
    def extract_scenic_spots(self, text: str) -> List[ExtractedEntity]:
        """抽取景区景点实体"""
        entities = []
        
        # 使用jieba进行命名实体识别
        words = pseg.cut(text)
        
        # 景区名称模式匹配
        patterns = [
            r'([\u4e00-\u9fa5]{2,}(景区|景点|公园|博物馆|寺庙|古镇|古城|山|湖|河|瀑布|峡谷))',
            r'([\u4e00-\u9fa5]{2,}(博物院|纪念馆|故居|遗址|名胜))'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity_name = match.group(1)
                
                # 提取相关属性
                properties = self._extract_spot_properties(text, entity_name)
                
                entity = ExtractedEntity(
                    entity_type="ScenicSpot",
                    name=entity_name,
                    properties=properties,
                    confidence=0.8,
                    source="text_extraction"
                )
                entities.append(entity)
        
        return entities
    
    def _extract_spot_properties(self, text: str, spot_name: str) -> Dict:
        """提取景点属性"""
        properties = {}
        
        # 提取景区等级
        level_pattern = r'(\d+A)[级]?景区'
        level_match = re.search(level_pattern, text)
        if level_match:
            properties['level'] = level_match.group(1)
        
        # 提取门票价格
        price_pattern = r'门票[价格]?[:：]?\s*(\d+)[元]?'
        price_match = re.search(price_pattern, text)
        if price_match:
            properties['ticket'] = {'price': float(price_match.group(1))}
        
        # 提取开放时间
        time_pattern = r'开放时间[:：]?\s*([\d:~-]+)'
        time_match = re.search(time_pattern, text)
        if time_match:
            properties['open_time'] = time_match.group(1)
        
        # 提取地理位置
        location_pattern = r'位于[:：]?\s*([\u4e00-\u9fa5省市区县]+)'
        location_match = re.search(location_pattern, text)
        if location_match:
            properties['location'] = {'address': location_match.group(1)}
        
        return properties
    
    def extract_cultural_heritage(self, text: str) -> List[ExtractedEntity]:
        """抽取文化遗产实体"""
        entities = []
        
        # 文化遗产模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,})被列入(世界文化遗产|国家级非物质文化遗产|省级文物保护单位)',
            r'(世界文化遗产|国家级非物质文化遗产)[:：]?\s*([\u4e00-\u9fa5]{2,})'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if len(match.groups()) >= 2:
                    entity_name = match.group(1)
                    heritage_type = match.group(2)
                    
                    properties = {
                        'heritage_type': '物质文化遗产' if '物质' not in heritage_type else '非物质文化遗产',
                        'protection_level': self._parse_protection_level(heritage_type)
                    }
                    
                    entity = ExtractedEntity(
                        entity_type="CulturalHeritage",
                        name=entity_name,
                        properties=properties,
                        confidence=0.85,
                        source="text_extraction"
                    )
                    entities.append(entity)
        
        return entities
    
    def _parse_protection_level(self, text: str) -> str:
        """解析保护级别"""
        if '世界' in text:
            return '世界级'
        elif '国家' in text:
            return '国家级'
        elif '省' in text:
            return '省级'
        elif '市' in text:
            return '市级'
        return '未知'
    
    def extract_events(self, text: str) -> List[ExtractedEntity]:
        """抽取活动节庆实体"""
        entities = []
        
        # 活动模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,}(节|展|会|赛|活动))',
            r'(\d{4}年[\u4e00-\u9fa5]{2,}(节|展|会))'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                event_name = match.group(1)
                
                # 提取时间信息
                properties = self._extract_event_properties(text, event_name)
                
                entity = ExtractedEntity(
                    entity_type="Event",
                    name=event_name,
                    properties=properties,
                    confidence=0.75,
                    source="text_extraction"
                )
                entities.append(entity)
        
        return entities
    
    def _extract_event_properties(self, text: str, event_name: str) -> Dict:
        """提取活动属性"""
        properties = {}
        
        # 提取时间
        date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日)[至到-](\d{4}年\d{1,2}月\d{1,2}日)'
        date_match = re.search(date_pattern, text)
        if date_match:
            properties['start_date'] = date_match.group(1)
            properties['end_date'] = date_match.group(2)
        
        # 提取地点
        location_pattern = r'在[:：]?\s*([\u4e00-\u9fa5]{2,})举办'
        location_match = re.search(location_pattern, text)
        if location_match:
            properties['location'] = {'venue': location_match.group(1)}
        
        return properties


class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """抽取实体间关系"""
        relations = []
        
        # 提取空间关系
        relations.extend(self._extract_spatial_relations(text, entities))
        
        # 提取包含关系
        relations.extend(self._extract_contains_relations(text, entities))
        
        # 提取文化关联关系
        relations.extend(self._extract_cultural_relations(text, entities))
        
        return relations
    
    def _extract_spatial_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """提取空间关系"""
        relations = []
        
        # 邻近关系模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,})位于([\u4e00-\u9fa5]{2,})(附近|旁边|东侧|西侧|南侧|北侧)',
            r'([\u4e00-\u9fa5]{2,})距离([\u4e00-\u9fa5]{2,})(\d+)(公里|米)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                source_name = match.group(1)
                target_name = match.group(2)
                
                # 查找对应实体
                source_entity = self._find_entity(entities, source_name)
                target_entity = self._find_entity(entities, target_name)
                
                if source_entity and target_entity:
                    properties = {}
                    if len(match.groups()) >= 4:
                        distance = match.group(3)
                        unit = match.group(4)
                        properties['distance'] = float(distance) * (1000 if unit == '公里' else 1)
                    
                    relation = ExtractedRelation(
                        relation_type="NEARBY",
                        source_entity=source_entity.name,
                        target_entity=target_entity.name,
                        properties=properties,
                        confidence=0.7,
                        source="text_extraction"
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_contains_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """提取包含关系"""
        relations = []
        
        # 包含关系模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,})包括([\u4e00-\u9fa5]{2,})',
            r'([\u4e00-\u9fa5]{2,})包含([\u4e00-\u9fa5]{2,})',
            r'([\u4e00-\u9fa5]{2,})由([\u4e00-\u9fa5]{2,})组成'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                source_name = match.group(1)
                target_name = match.group(2)
                
                source_entity = self._find_entity(entities, source_name)
                target_entity = self._find_entity(entities, target_name)
                
                if source_entity and target_entity:
                    relation = ExtractedRelation(
                        relation_type="CONTAINS",
                        source_entity=source_entity.name,
                        target_entity=target_entity.name,
                        properties={},
                        confidence=0.75,
                        source="text_extraction"
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_cultural_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """提取文化关联关系"""
        relations = []
        
        # 文化关联模式
        patterns = [
            r'([\u4e00-\u9fa5]{2,})是([\u4e00-\u9fa5]{2,})的重要组成部分',
            r'([\u4e00-\u9fa5]{2,})与([\u4e00-\u9fa5]{2,})密切相关'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                source_name = match.group(1)
                target_name = match.group(2)
                
                source_entity = self._find_entity(entities, source_name)
                target_entity = self._find_entity(entities, target_name)
                
                if source_entity and target_entity:
                    relation = ExtractedRelation(
                        relation_type="RELATED_TO",
                        source_entity=source_entity.name,
                        target_entity=target_entity.name,
                        properties={'relation_type': '文化关联'},
                        confidence=0.7,
                        source="text_extraction"
                    )
                    relations.append(relation)
        
        return relations
    
    def _find_entity(self, entities: List[ExtractedEntity], name: str) -> Optional[ExtractedEntity]:
        """根据名称查找实体"""
        for entity in entities:
            if entity.name == name or name in entity.name or entity.name in name:
                return entity
        return None


class EntityFusion:
    """实体融合器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fuse_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """融合重复实体"""
        # 按名称分组
        entity_groups = {}
        for entity in entities:
            key = self._generate_entity_key(entity)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)
        
        # 融合每组实体
        fused_entities = []
        for key, group in entity_groups.items():
            if len(group) == 1:
                fused_entities.append(group[0])
            else:
                fused_entity = self._merge_entity_group(group)
                fused_entities.append(fused_entity)
        
        return fused_entities
    
    def _generate_entity_key(self, entity: ExtractedEntity) -> str:
        """生成实体唯一键"""
        # 简化名称
        name = entity.name.replace("景区", "").replace("景点", "").replace("公园", "")
        return f"{entity.entity_type}_{name}"
    
    def _merge_entity_group(self, group: List[ExtractedEntity]) -> ExtractedEntity:
        """合并一组实体"""
        # 选择置信度最高的作为基础
        base_entity = max(group, key=lambda e: e.confidence)
        
        # 合并属性
        merged_properties = {}
        for entity in group:
            for key, value in entity.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                elif isinstance(value, dict):
                    if key not in merged_properties:
                        merged_properties[key] = {}
                    merged_properties[key].update(value)
        
        # 计算平均置信度
        avg_confidence = sum(e.confidence for e in group) / len(group)
        
        return ExtractedEntity(
            entity_type=base_entity.entity_type,
            name=base_entity.name,
            properties=merged_properties,
            confidence=avg_confidence,
            source="fusion",
            extraction_time=datetime.now()
        )


class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver
        self.logger = logging.getLogger(__name__)
    
    async def build_from_entities(self, entities: List[ExtractedEntity]) -> Dict[str, int]:
        """从实体构建知识图谱"""
        stats = {"nodes_created": 0, "errors": 0}
        
        for entity in entities:
            try:
                node_id = await self._create_node(entity)
                stats["nodes_created"] += 1
            except Exception as e:
                self.logger.error(f"创建节点失败 {entity.name}: {e}")
                stats["errors"] += 1
        
        return stats
    
    async def build_from_relations(self, relations: List[ExtractedRelation]) -> Dict[str, int]:
        """从关系构建知识图谱"""
        stats = {"relations_created": 0, "errors": 0}
        
        for relation in relations:
            try:
                await self._create_relation(relation)
                stats["relations_created"] += 1
            except Exception as e:
                self.logger.error(f"创建关系失败 {relation.source_entity}-{relation.target_entity}: {e}")
                stats["errors"] += 1
        
        return stats
    
    async def _create_node(self, entity: ExtractedEntity) -> str:
        """创建节点"""
        query = f"""
            MERGE (n:{entity.entity_type} {{name: $name}})
            SET n += $properties,
                n.confidence = $confidence,
                n.source = $source,
                n.updated_at = datetime()
            RETURN id(n) as node_id
        """
        
        with self.driver.session() as session:
            result = session.run(
                query,
                name=entity.name,
                properties=entity.properties,
                confidence=entity.confidence,
                source=entity.source
            )
            return result.single()["node_id"]
    
    async def _create_relation(self, relation: ExtractedRelation):
        """创建关系"""
        query = f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            MERGE (source)-[r:{relation.relation_type}]->(target)
            SET r += $properties,
                r.confidence = $confidence
        """
        
        with self.driver.session() as session:
            session.run(
                query,
                source_name=relation.source_entity,
                target_name=relation.target_entity,
                properties=relation.properties,
                confidence=relation.confidence
            )


class KnowledgeExtractionPipeline:
    """知识抽取Pipeline"""
    
    def __init__(self, neo4j_driver=None):
        self.collector = DataCollector()
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor()
        self.entity_fusion = EntityFusion()
        self.graph_builder = KnowledgeGraphBuilder(neo4j_driver) if neo4j_driver else None
        
        self.logger = logging.getLogger(__name__)
    
    async def run(self, data_sources: List[DataSource]) -> Dict:
        """运行完整Pipeline"""
        self.logger.info("开始知识抽取Pipeline")
        
        # 1. 数据采集
        self.logger.info("阶段1: 数据采集")
        collected_data = await self.collector.collect_all()
        
        # 2. 实体抽取
        self.logger.info("阶段2: 实体抽取")
        all_entities = []
        for source_url, data in collected_data.items():
            if isinstance(data, str):
                # 抽取景区景点
                entities = self.entity_extractor.extract_scenic_spots(data)
                all_entities.extend(entities)
                
                # 抽取文化遗产
                entities = self.entity_extractor.extract_cultural_heritage(data)
                all_entities.extend(entities)
                
                # 抽取活动节庆
                entities = self.entity_extractor.extract_events(data)
                all_entities.extend(entities)
        
        # 3. 关系抽取
        self.logger.info("阶段3: 关系抽取")
        all_relations = []
        for source_url, data in collected_data.items():
            if isinstance(data, str):
                relations = self.relation_extractor.extract_relations(data, all_entities)
                all_relations.extend(relations)
        
        # 4. 实体融合
        self.logger.info("阶段4: 实体融合")
        fused_entities = self.entity_fusion.fuse_entities(all_entities)
        
        # 5. 构建知识图谱
        self.logger.info("阶段5: 构建知识图谱")
        stats = {}
        if self.graph_builder:
            entity_stats = await self.graph_builder.build_from_entities(fused_entities)
            relation_stats = await self.graph_builder.build_from_relations(all_relations)
            stats = {**entity_stats, **relation_stats}
        
        result = {
            "entities_count": len(fused_entities),
            "relations_count": len(all_relations),
            "stats": stats,
            "entities": fused_entities,
            "relations": all_relations
        }
        
        self.logger.info(f"Pipeline完成: {result['entities_count']}个实体, {result['relations_count']}个关系")
        return result


# 使用示例
async def main():
    """示例：运行知识抽取Pipeline"""
    
    # 创建Pipeline
    pipeline = KnowledgeExtractionPipeline()
    
    # 添加数据源
    data_sources = [
        DataSource(
            source_type="web",
            source_url="https://example.com/tourism/spots",
            data_format="html",
            update_frequency="weekly",
            priority=1
        ),
        DataSource(
            source_type="file",
            source_url="data/tourism_info.txt",
            data_format="text",
            update_frequency="monthly",
            priority=2
        )
    ]
    
    for source in data_sources:
        pipeline.collector.add_source(source)
    
    # 运行Pipeline
    result = await pipeline.run(data_sources)
    
    print(f"抽取结果:")
    print(f"  实体数量: {result['entities_count']}")
    print(f"  关系数量: {result['relations_count']}")
    print(f"  统计信息: {result['stats']}")


if __name__ == "__main__":
    asyncio.run(main())
