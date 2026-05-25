"""
文旅知识图谱本体层构建模块
定义知识图谱的概念层、约束规则和推理规则
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import json


class EntityType(Enum):
    """实体类型枚举"""
    SCENIC_SPOT = "ScenicSpot"
    CULTURAL_HERITAGE = "CulturalHeritage"
    TRAVEL_ROUTE = "TravelRoute"
    SERVICE_FACILITY = "ServiceFacility"
    EVENT = "Event"
    TOURIST_PROFILE = "TouristProfile"
    THEME = "Theme"
    ADMINISTRATIVE_AREA = "AdministrativeArea"


class RelationType(Enum):
    """关系类型枚举"""
    # 空间关系
    CONTAINS = "CONTAINS"
    NEARBY = "NEARBY"
    LOCATED_IN = "LOCATED_IN"
    
    # 旅游关系
    PASSES_THROUGH = "PASSES_THROUGH"
    VISITED = "VISITED"
    RECOMMENDED_TO = "RECOMMENDED_TO"
    
    # 文化关系
    RELATED_TO = "RELATED_TO"
    HELD_AT = "HELD_AT"
    HAS_THEME = "HAS_THEME"
    
    # 服务关系
    SERVES = "SERVES"
    INCLUDES_ACCOMMODATION = "INCLUDES_ACCOMMODATION"
    
    # 相似关系
    SIMILAR_TO = "SIMILAR_TO"


@dataclass
class PropertyDefinition:
    """属性定义"""
    name: str
    data_type: str  # String, Integer, Float, Date, List, Object
    required: bool = False
    default_value: Optional[any] = None
    constraints: Dict = field(default_factory=dict)
    description: str = ""


@dataclass
class EntitySchema:
    """实体模式定义"""
    entity_type: EntityType
    properties: List[PropertyDefinition]
    indexes: List[List[str]] = field(default_factory=list)
    fulltext_indexes: List[List[str]] = field(default_factory=list)
    vector_indexes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class RelationSchema:
    """关系模式定义"""
    relation_type: RelationType
    source_types: List[EntityType]
    target_types: List[EntityType]
    properties: List[PropertyDefinition] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class TourismOntology:
    """文旅知识图谱本体"""
    
    def __init__(self):
        self.entity_schemas: Dict[EntityType, EntitySchema] = {}
        self.relation_schemas: Dict[RelationType, RelationSchema] = {}
        self.inference_rules: List[Dict] = []
        
        self._initialize_entity_schemas()
        self._initialize_relation_schemas()
        self._initialize_inference_rules()
    
    def _initialize_entity_schemas(self):
        """初始化实体模式"""
        
        # 景区景点实体
        self.entity_schemas[EntityType.SCENIC_SPOT] = EntitySchema(
            entity_type=EntityType.SCENIC_SPOT,
            properties=[
                PropertyDefinition("id", "String", required=True, description="唯一标识"),
                PropertyDefinition("name", "String", required=True, description="景点名称"),
                PropertyDefinition("full_name", "String", description="完整名称"),
                PropertyDefinition("level", "String", constraints={"enum": ["5A", "4A", "3A", "2A", "A"]}, description="景区等级"),
                PropertyDefinition("category", "List", description="类型标签"),
                PropertyDefinition("description", "String", description="详细描述"),
                PropertyDefinition("location", "Object", required=True, description="地理位置"),
                PropertyDefinition("ticket", "Object", description="门票信息"),
                PropertyDefinition("open_time", "String", description="开放时间"),
                PropertyDefinition("best_season", "List", description="最佳游览季节"),
                PropertyDefinition("duration", "String", description="建议游玩时长"),
                PropertyDefinition("rating", "Float", constraints={"min": 0, "max": 5}, description="评分"),
                PropertyDefinition("popularity", "Integer", constraints={"min": 0}, description="热度指数"),
                PropertyDefinition("embedding", "List", description="文本向量表示"),
                PropertyDefinition("created_at", "Date", description="创建时间"),
                PropertyDefinition("updated_at", "Date", description="更新时间")
            ],
            indexes=[
                ["name"],
                ["level"],
                ["location.province", "location.city"]
            ],
            fulltext_indexes=[
                ["name", "description"]
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(location)"
            ]
        )
        
        # 文化遗产实体
        self.entity_schemas[EntityType.CULTURAL_HERITAGE] = EntitySchema(
            entity_type=EntityType.CULTURAL_HERITAGE,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True),
                PropertyDefinition("heritage_type", "String", required=True, 
                                 constraints={"enum": ["物质文化遗产", "非物质文化遗产"]}),
                PropertyDefinition("protection_level", "String", 
                                 constraints={"enum": ["世界级", "国家级", "省级", "市级"]}),
                PropertyDefinition("era", "String", description="所属年代"),
                PropertyDefinition("history", "String", description="历史背景"),
                PropertyDefinition("cultural_value", "String", description="文化价值"),
                PropertyDefinition("location", "Object"),
                PropertyDefinition("inheritors", "List", description="传承人"),
                PropertyDefinition("status", "String", description="保护状态"),
                PropertyDefinition("embedding", "List")
            ],
            fulltext_indexes=[["name", "history", "cultural_value"]],
            vector_indexes=["embedding"]
        )
        
        # 旅游路线实体
        self.entity_schemas[EntityType.TRAVEL_ROUTE] = EntitySchema(
            entity_type=EntityType.TRAVEL_ROUTE,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True),
                PropertyDefinition("theme", "String", description="主题"),
                PropertyDefinition("duration", "Integer", required=True, description="行程天数"),
                PropertyDefinition("difficulty", "String", constraints={"enum": ["简单", "中等", "困难"]}),
                PropertyDefinition("budget", "Object", description="预算范围"),
                PropertyDefinition("transportation", "List", description="交通方式"),
                PropertyDefinition("highlights", "List", description="行程亮点"),
                PropertyDefinition("itinerary", "List", description="详细行程"),
                PropertyDefinition("suitable_crowd", "List", description="适用人群"),
                PropertyDefinition("embedding", "List")
            ],
            indexes=[["theme"], ["duration"]],
            vector_indexes=["embedding"]
        )
        
        # 服务设施实体
        self.entity_schemas[EntityType.SERVICE_FACILITY] = EntitySchema(
            entity_type=EntityType.SERVICE_FACILITY,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True),
                PropertyDefinition("facility_type", "String", required=True,
                                 constraints={"enum": ["餐饮", "住宿", "交通", "购物", "娱乐"]}),
                PropertyDefinition("category", "String", description="细分类别"),
                PropertyDefinition("location", "Object", required=True),
                PropertyDefinition("price_level", "String", constraints={"enum": ["低", "中", "高", "豪华"]}),
                PropertyDefinition("rating", "Float", constraints={"min": 0, "max": 5}),
                PropertyDefinition("features", "List", description="特色标签"),
                PropertyDefinition("contact", "Object"),
                PropertyDefinition("opening_hours", "String")
            ],
            indexes=[["facility_type"], ["price_level"]]
        )
        
        # 活动节庆实体
        self.entity_schemas[EntityType.EVENT] = EntitySchema(
            entity_type=EntityType.EVENT,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True),
                PropertyDefinition("event_type", "String", required=True,
                                 constraints={"enum": ["节庆", "展览", "演出", "赛事"]}),
                PropertyDefinition("start_date", "Date", required=True),
                PropertyDefinition("end_date", "Date", required=True),
                PropertyDefinition("location", "Object", required=True),
                PropertyDefinition("description", "String"),
                PropertyDefinition("highlights", "List"),
                PropertyDefinition("ticket_info", "String"),
                PropertyDefinition("organizer", "String"),
                PropertyDefinition("scale", "String")
            ],
            indexes=[["event_type"], ["start_date", "end_date"]]
        )
        
        # 游客画像实体
        self.entity_schemas[EntityType.TOURIST_PROFILE] = EntitySchema(
            entity_type=EntityType.TOURIST_PROFILE,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("age_group", "String", 
                                 constraints={"enum": ["儿童", "青年", "中年", "老年"]}),
                PropertyDefinition("travel_preference", "List", description="旅游偏好"),
                PropertyDefinition("budget_range", "Object"),
                PropertyDefinition("travel_frequency", "String"),
                PropertyDefinition("interests", "List", description="兴趣标签"),
                PropertyDefinition("history", "List", description="历史行为"),
                PropertyDefinition("constraints", "Object", description="限制条件")
            ]
        )
    
    def _initialize_relation_schemas(self):
        """初始化关系模式"""
        
        # 包含关系
        self.relation_schemas[RelationType.CONTAINS] = RelationSchema(
            relation_type=RelationType.CONTAINS,
            source_types=[EntityType.SCENIC_SPOT],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("level", "Integer", description="层级")
            ],
            constraints=["NO_CYCLE"]  # 不能形成环
        )
        
        # 邻近关系
        self.relation_schemas[RelationType.NEARBY] = RelationSchema(
            relation_type=RelationType.NEARBY,
            source_types=[EntityType.SCENIC_SPOT],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("distance", "Float", description="距离(米)"),
                PropertyDefinition("walk_time", "Integer", description="步行时间(分钟)"),
                PropertyDefinition("transport_time", "Integer", description="交通时间(分钟)")
            ],
            constraints=["SYMMETRIC"]  # 对称关系
        )
        
        # 路线经过关系
        self.relation_schemas[RelationType.PASSES_THROUGH] = RelationSchema(
            relation_type=RelationType.PASSES_THROUGH,
            source_types=[EntityType.TRAVEL_ROUTE],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("day", "Integer", required=True, description="第几天"),
                PropertyDefinition("order", "Integer", required=True, description="当天顺序"),
                PropertyDefinition("duration", "String", description="游玩时长")
            ],
            constraints=["ORDERED_BY_DAY_AND_ORDER"]  # 按天和顺序有序
        )
        
        # 游览关系
        self.relation_schemas[RelationType.VISITED] = RelationSchema(
            relation_type=RelationType.VISITED,
            source_types=[EntityType.TOURIST_PROFILE],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("time", "Date", description="游览时间"),
                PropertyDefinition("rating", "Float", description="评分"),
                PropertyDefinition("duration", "String", description="游玩时长")
            ]
        )
        
        # 推荐关系
        self.relation_schemas[RelationType.RECOMMENDED_TO] = RelationSchema(
            relation_type=RelationType.RECOMMENDED_TO,
            source_types=[EntityType.SCENIC_SPOT],
            target_types=[EntityType.TOURIST_PROFILE],
            properties=[
                PropertyDefinition("score", "Float", description="推荐分数"),
                PropertyDefinition("reason", "String", description="推荐理由")
            ]
        )
        
        # 文化关联关系
        self.relation_schemas[RelationType.RELATED_TO] = RelationSchema(
            relation_type=RelationType.RELATED_TO,
            source_types=[EntityType.CULTURAL_HERITAGE],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("relation_type", "String", description="关联类型"),
                PropertyDefinition("description", "String", description="关联描述")
            ]
        )
        
        # 活动举办关系
        self.relation_schemas[RelationType.HELD_AT] = RelationSchema(
            relation_type=RelationType.HELD_AT,
            source_types=[EntityType.EVENT],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[]
        )
        
        # 主题关系
        self.relation_schemas[RelationType.HAS_THEME] = RelationSchema(
            relation_type=RelationType.HAS_THEME,
            source_types=[EntityType.SCENIC_SPOT],
            target_types=[EntityType.THEME],
            properties=[]
        )
        
        # 服务关系
        self.relation_schemas[RelationType.SERVES] = RelationSchema(
            relation_type=RelationType.SERVES,
            source_types=[EntityType.SERVICE_FACILITY],
            target_types=[EntityType.SCENIC_SPOT],
            properties=[
                PropertyDefinition("distance", "Float", description="距离"),
                PropertyDefinition("service_type", "String", description="服务类型")
            ]
        )
        
        # 相似关系
        self.relation_schemas[RelationType.SIMILAR_TO] = RelationSchema(
            relation_type=RelationType.SIMILAR_TO,
            source_types=[EntityType.SCENIC_SPOT, EntityType.TRAVEL_ROUTE],
            target_types=[EntityType.SCENIC_SPOT, EntityType.TRAVEL_ROUTE],
            properties=[
                PropertyDefinition("similarity", "Float", constraints={"min": 0, "max": 1}),
                PropertyDefinition("common_features", "List")
            ],
            constraints=["SYMMETRIC"]
        )
    
    def _initialize_inference_rules(self):
        """初始化推理规则"""
        
        self.inference_rules = [
            {
                "name": "传递邻近关系",
                "description": "如果A邻近B，B邻近C，且距离小于阈值，则A邻近C",
                "rule": """
                    MATCH (a:ScenicSpot)-[r1:NEARBY]->(b:ScenicSpot)-[r2:NEARBY]->(c:ScenicSpot)
                    WHERE a <> c AND (r1.distance + r2.distance) < 5000
                    AND NOT (a)-[:NEARBY]->(c)
                    MERGE (a)-[:NEARBY {
                        distance: r1.distance + r2.distance,
                        inferred: true
                    }]->(c)
                """
            },
            {
                "name": "相似景点推荐",
                "description": "如果用户喜欢A，A相似B，则推荐B给用户",
                "rule": """
                    MATCH (u:TouristProfile)-[v:VISITED]->(a:ScenicSpot)
                    WHERE v.rating > 3.5
                    MATCH (a)-[s:SIMILAR_TO]->(b:ScenicSpot)
                    WHERE s.similarity > 0.7
                    AND NOT (u)-[:VISITED]->(b)
                    MERGE (b)-[:RECOMMENDED_TO {
                        score: v.rating * s.similarity,
                        reason: '基于相似景点推荐',
                        inferred: true
                    }]->(u)
                """
            },
            {
                "name": "主题关联推理",
                "description": "相同主题的景点建立相似关系",
                "rule": """
                    MATCH (a:ScenicSpot)-[:HAS_THEME]->(t:Theme)<-[:HAS_THEME]-(b:ScenicSpot)
                    WHERE a <> b AND NOT (a)-[:SIMILAR_TO]->(b)
                    MERGE (a)-[:SIMILAR_TO {
                        similarity: 0.6,
                        common_features: ['相同主题:' + t.name],
                        inferred: true
                    }]->(b)
                """
            },
            {
                "name": "路线热度计算",
                "description": "根据路线包含景点的热度计算路线热度",
                "rule": """
                    MATCH (r:TravelRoute)-[:PASSES_THROUGH]->(s:ScenicSpot)
                    WITH r, avg(s.popularity) as avg_popularity, 
                         avg(s.rating) as avg_rating
                    SET r.popularity = avg_popularity,
                        r.rating = avg_rating
                """
            }
        ]
    
    def generate_neo4j_schema(self) -> str:
        """生成Neo4j Schema创建语句"""
        cypher_statements = []
        
        # 创建约束和索引
        for entity_type, schema in self.entity_schemas.items():
            # 唯一性约束
            cypher_statements.append(
                f"CREATE CONSTRAINT {entity_type.value}_id_unique IF NOT EXISTS "
                f"FOR (n:{entity_type.value}) REQUIRE n.id IS UNIQUE;"
            )
            
            # 索引
            for index in schema.indexes:
                if len(index) == 1:
                    cypher_statements.append(
                        f"CREATE INDEX {entity_type.value}_{index[0]}_idx IF NOT EXISTS "
                        f"FOR (n:{entity_type.value}) ON (n.{index[0]});"
                    )
                else:
                    idx_name = "_".join(index)
                    idx_fields = ", ".join([f"n.{field}" for field in index])
                    cypher_statements.append(
                        f"CREATE INDEX {entity_type.value}_{idx_name}_idx IF NOT EXISTS "
                        f"FOR (n:{entity_type.value}) ON ({idx_fields});"
                    )
            
            # 全文索引
            for fulltext_idx in schema.fulltext_indexes:
                idx_name = "_".join(fulltext_idx)
                idx_fields = ", ".join([f"n.{field}" for field in fulltext_idx])
                cypher_statements.append(
                    f"CREATE FULLTEXT INDEX {entity_type.value}_{idx_name}_fulltext IF NOT EXISTS "
                    f"FOR (n:{entity_type.value}) ON EACH [{idx_fields}];"
                )
            
            # 向量索引
            for vector_idx in schema.vector_indexes:
                cypher_statements.append(
                    f"CREATE VECTOR INDEX {entity_type.value}_{vector_idx}_vector IF NOT EXISTS "
                    f"FOR (n:{entity_type.value}) ON n.{vector_idx} "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}}};"
                )
        
        return "\n".join(cypher_statements)
    
    def validate_entity(self, entity_type: EntityType, properties: Dict) -> List[str]:
        """验证实体属性"""
        errors = []
        schema = self.entity_schemas.get(entity_type)
        
        if not schema:
            errors.append(f"未知的实体类型: {entity_type}")
            return errors
        
        for prop_def in schema.properties:
            # 检查必填属性
            if prop_def.required and prop_def.name not in properties:
                errors.append(f"缺少必填属性: {prop_def.name}")
                continue
            
            if prop_def.name in properties:
                value = properties[prop_def.name]
                
                # 检查枚举约束
                if "enum" in prop_def.constraints:
                    if value not in prop_def.constraints["enum"]:
                        errors.append(
                            f"属性 {prop_def.name} 的值 {value} 不在允许的枚举值中: "
                            f"{prop_def.constraints['enum']}"
                        )
                
                # 检查范围约束
                if "min" in prop_def.constraints and value < prop_def.constraints["min"]:
                    errors.append(
                        f"属性 {prop_def.name} 的值 {value} 小于最小值 {prop_def.constraints['min']}"
                    )
                
                if "max" in prop_def.constraints and value > prop_def.constraints["max"]:
                    errors.append(
                        f"属性 {prop_def.name} 的值 {value} 大于最大值 {prop_def.constraints['max']}"
                    )
        
        return errors
    
    def to_json(self) -> str:
        """导出本体为JSON格式"""
        ontology_dict = {
            "entity_schemas": {},
            "relation_schemas": {},
            "inference_rules": self.inference_rules
        }
        
        for entity_type, schema in self.entity_schemas.items():
            ontology_dict["entity_schemas"][entity_type.value] = {
                "properties": [
                    {
                        "name": p.name,
                        "data_type": p.data_type,
                        "required": p.required,
                        "constraints": p.constraints,
                        "description": p.description
                    }
                    for p in schema.properties
                ],
                "indexes": schema.indexes,
                "fulltext_indexes": schema.fulltext_indexes,
                "vector_indexes": schema.vector_indexes,
                "constraints": schema.constraints
            }
        
        for relation_type, schema in self.relation_schemas.items():
            ontology_dict["relation_schemas"][relation_type.value] = {
                "source_types": [t.value for t in schema.source_types],
                "target_types": [t.value for t in schema.target_types],
                "properties": [
                    {
                        "name": p.name,
                        "data_type": p.data_type,
                        "required": p.required,
                        "description": p.description
                    }
                    for p in schema.properties
                ],
                "constraints": schema.constraints
            }
        
        return json.dumps(ontology_dict, ensure_ascii=False, indent=2)


# 使用示例
if __name__ == "__main__":
    # 创建本体实例
    ontology = TourismOntology()
    
    # 生成Neo4j Schema
    neo4j_schema = ontology.generate_neo4j_schema()
    print("Neo4j Schema创建语句:")
    print(neo4j_schema)
    
    # 验证实体
    test_entity = {
        "id": "spot_001",
        "name": "故宫博物院",
        "level": "5A",
        "rating": 4.8,
        "location": {
            "province": "北京市",
            "city": "北京市"
        }
    }
    
    validation_errors = ontology.validate_entity(EntityType.SCENIC_SPOT, test_entity)
    if validation_errors:
        print("\n验证错误:")
        for error in validation_errors:
            print(f"  - {error}")
    else:
        print("\n实体验证通过")
    
    # 导出JSON
    ontology_json = ontology.to_json()
    print("\n本体JSON导出:")
    print(ontology_json[:500] + "...")
