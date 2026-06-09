"""
非遗剪纸知识图谱本体层构建模块
定义剪纸领域的实体类型、关系类型、属性约束和推理规则
与文旅本体并行扩展，通过 CulturalHeritage 实体桥接
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
import json


class PaperCutEntityType(Enum):
    """剪纸领域实体类型枚举"""
    PAPER_CUT_PATTERN = "PaperCutPattern"          # 剪纸纹样（核心实体）
    PAPER_CUT_MOTIF = "PaperCutMotif"               # 纹样母题（云纹/龙纹/花鸟等）
    TECHNIQUE = "Technique"                         # 技法（阴刻/阳刻/套色等）
    REGIONAL_STYLE = "RegionalStyle"                # 地域流派
    CULTURAL_SYMBOL = "CulturalSymbol"              # 文化象征/吉祥寓意
    PAPER_CUT_INHERITOR = "PaperCutInheritor"       # 传承人（审核者）
    MATERIAL = "Material"                           # 材料工具
    GENERATION_RECORD = "GenerationRecord"          # AI生成记录（用于反馈+训练）


class PaperCutRelationType(Enum):
    """剪纸领域关系类型枚举"""
    # 纹样关系
    HAS_MOTIF = "HAS_MOTIF"                         # 纹样包含母题
    HAS_VARIANT = "HAS_VARIANT"                     # 纹样变体关系
    COMPOSED_OF = "COMPOSED_OF"                     # 纹样由多个子纹样组成

    # 技法关系
    USES_TECHNIQUE = "USES_TECHNIQUE"               # 纹样使用技法
    SPECIALIZES_IN = "SPECIALIZES_IN"               # 传承人擅长技法

    # 流派关系
    BELONGS_TO_STYLE = "BELONGS_TO_STYLE"           # 纹样属于流派
    INFLUENCED_BY = "INFLUENCED_BY"                 # 流派间影响

    # 文化象征关系
    HAS_SYMBOLISM = "HAS_SYMBOLISM"                 # 纹样具有象征意义
    REPRESENTS = "REPRESENTS"                       # 母题代表文化象征

    # 人物关系
    CREATED_BY = "CREATED_BY"                       # 纹样由传承人创作
    REVIEWED_BY = "REVIEWED_BY"                     # 生成记录由传承人审核

    # 材料关系
    REQUIRES_MATERIAL = "REQUIRES_MATERIAL"         # 纹样需要材料

    # 生成记录关系
    GENERATED_FROM = "GENERATED_FROM"               # 生成记录基于纹样

    # 跨领域桥接关系（连接文旅体系）
    HAS_CULTURAL_BACKGROUND = "HAS_CULTURAL_BACKGROUND"  # 纹样具有文化遗产背景

    # 母题间关系
    RELATED_MOTIF = "RELATED_MOTIF"                 # 母题演化/组合/派生关系


@dataclass
class PropertyDefinition:
    """属性定义"""
    name: str
    data_type: str  # String, Integer, Float, Date, Boolean, List, Object
    required: bool = False
    default_value: Optional[any] = None
    constraints: Dict = field(default_factory=dict)
    description: str = ""


@dataclass
class EntitySchema:
    """实体模式定义"""
    entity_type: PaperCutEntityType
    properties: List[PropertyDefinition]
    indexes: List[List[str]] = field(default_factory=list)
    fulltext_indexes: List[List[str]] = field(default_factory=list)
    vector_indexes: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class RelationSchema:
    """关系模式定义"""
    relation_type: PaperCutRelationType
    source_types: List[PaperCutEntityType]
    target_types: List[PaperCutEntityType]
    properties: List[PropertyDefinition] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


class PaperCutOntology:
    """非遗剪纸知识图谱本体"""

    def __init__(self):
        self.entity_schemas: Dict[PaperCutEntityType, EntitySchema] = {}
        self.relation_schemas: Dict[PaperCutRelationType, RelationSchema] = {}
        self.inference_rules: List[Dict] = []

        self._initialize_entity_schemas()
        self._initialize_relation_schemas()
        self._initialize_inference_rules()

    # ========== 实体模式定义 ==========

    def _initialize_entity_schemas(self):
        """初始化所有剪纸领域实体模式"""

        # 1. 剪纸纹样（核心实体）
        self.entity_schemas[PaperCutEntityType.PAPER_CUT_PATTERN] = EntitySchema(
            entity_type=PaperCutEntityType.PAPER_CUT_PATTERN,
            properties=[
                PropertyDefinition("id", "String", required=True, description="唯一标识"),
                PropertyDefinition("name", "String", required=True, description="纹样名称"),
                PropertyDefinition("full_name", "String", description="完整名称/俗称"),
                PropertyDefinition("category", "String", required=True,
                                   constraints={"enum": ["传统纹样", "创新纹样", "民俗纹样", "宗教纹样", "宫廷纹样"]},
                                   description="纹样大类"),
                PropertyDefinition("sub_category", "List", description="细分类别标签"),
                PropertyDefinition("description", "String", description="纹样详细描述"),
                PropertyDefinition("complexity", "String",
                                   constraints={"enum": ["简单", "中等", "复杂", "极复杂"]},
                                   description="复杂度"),
                PropertyDefinition("typical_dimensions", "Object", description="典型尺寸 {width, height, unit}"),
                PropertyDefinition("recommended_colors", "List", description="推荐配色方案"),
                PropertyDefinition("color_count", "Integer", constraints={"min": 1, "max": 20},
                                   description="色彩数量（单色/多色）"),
                PropertyDefinition("meaning", "String", description="吉祥寓意简述"),
                PropertyDefinition("cultural_context", "String", description="文化背景与历史渊源"),
                PropertyDefinition("era", "String", description="起源/流行年代"),
                PropertyDefinition("region", "String", description="主要分布区域"),
                PropertyDefinition("symmetry_type", "String",
                                   constraints={"enum": ["中心对称", "轴对称", "旋转对称", "不对称"]},
                                   description="对称类型"),
                PropertyDefinition("applicable_scenes", "List", description="适用场景（春节/婚庆/装饰等）"),
                PropertyDefinition("difficulty_level", "Integer", constraints={"min": 1, "max": 5},
                                   description="制作难度 1-5"),
                PropertyDefinition("reference_image_url", "String", description="参考图链接"),
                PropertyDefinition("reference_source", "String", description="图片来源/出处"),
                PropertyDefinition("status", "String", required=True,
                                   constraints={"enum": ["草稿", "待审核", "已审核", "已发布", "已归档"]},
                                   description="数据状态（审核流）"),
                PropertyDefinition("is_public", "Boolean", default_value=True, description="是否公开可见"),
                PropertyDefinition("popularity", "Integer", constraints={"min": 0}, description="热度/使用频率"),
                PropertyDefinition("embedding", "List", description="纹样描述向量"),
                PropertyDefinition("tags", "List", description="自定义标签"),
                PropertyDefinition("created_by", "String", description="创建人"),
                PropertyDefinition("reviewed_by", "String", description="审核人"),
                PropertyDefinition("review_comment", "String", description="审核意见"),
                PropertyDefinition("version", "Integer", default_value=1, description="版本号"),
                PropertyDefinition("created_at", "Date", description="创建时间"),
                PropertyDefinition("updated_at", "Date", description="更新时间"),
            ],
            indexes=[
                ["name"],
                ["category"],
                ["region"],
                ["status"],
                ["applicable_scenes"],
                ["symmetry_type"],
                ["complexity"],
            ],
            fulltext_indexes=[
                ["name", "description", "meaning"],
                ["cultural_context", "tags"],
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(category)",
            ]
        )

        # 2. 纹样母题
        self.entity_schemas[PaperCutEntityType.PAPER_CUT_MOTIF] = EntitySchema(
            entity_type=PaperCutEntityType.PAPER_CUT_MOTIF,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="母题名称（云纹/龙纹/牡丹等）"),
                PropertyDefinition("category", "String", required=True,
                                   constraints={"enum": ["动物", "植物", "人物", "几何",
                                                         "文字", "器物", "自然景观", "神话传说"]},
                                   description="母题大类"),
                PropertyDefinition("sub_category", "String", description="细分类别"),
                PropertyDefinition("description", "String", description="母题描述"),
                PropertyDefinition("symbolic_meaning", "String", description="象征意义"),
                PropertyDefinition("cultural_background", "String", description="文化背景"),
                PropertyDefinition("typical_techniques", "List", description="常用技法"),
                PropertyDefinition("common_regions", "List", description="常见地域"),
                PropertyDefinition("style_variants", "List", description="风格变体描述"),
                PropertyDefinition("reference_image_url", "String", description="参考图"),
                PropertyDefinition("embedding", "List"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["category"],
            ],
            fulltext_indexes=[
                ["name", "description", "symbolic_meaning"],
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(category)",
            ]
        )

        # 3. 技法
        self.entity_schemas[PaperCutEntityType.TECHNIQUE] = EntitySchema(
            entity_type=PaperCutEntityType.TECHNIQUE,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="技法名称"),
                PropertyDefinition("category", "String", required=True,
                                   constraints={"enum": ["阴刻", "阳刻", "阴阳刻", "套色剪纸",
                                                         "剪纸", "刻纸", "撕纸", "火烫", "染纸",
                                                         "拼色", "立体剪纸"]},
                                   description="技法大类"),
                PropertyDefinition("description", "String", description="技法描述"),
                PropertyDefinition("difficulty_level", "Integer", constraints={"min": 1, "max": 5},
                                   description="难度等级"),
                PropertyDefinition("tools_required", "List", description="所需工具"),
                PropertyDefinition("skill_points", "List", description="技巧要点"),
                PropertyDefinition("typical_effects", "String", description="典型效果"),
                PropertyDefinition("origin_region", "String", description="发源地"),
                PropertyDefinition("is_traditional", "Boolean", default_value=True, description="是否为传统技法"),
                PropertyDefinition("video_tutorial_url", "String", description="教学视频链接"),
                PropertyDefinition("history", "String", description="技法历史"),
                PropertyDefinition("embedding", "List"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["category"],
                ["difficulty_level"],
            ],
            fulltext_indexes=[
                ["name", "description", "history"],
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(category)",
            ]
        )

        # 4. 地域流派
        self.entity_schemas[PaperCutEntityType.REGIONAL_STYLE] = EntitySchema(
            entity_type=PaperCutEntityType.REGIONAL_STYLE,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="流派名称"),
                PropertyDefinition("region", "String", required=True, description="所属地区"),
                PropertyDefinition("province", "String", description="所在省份"),
                PropertyDefinition("history", "String", description="流派历史"),
                PropertyDefinition("characteristics", "String", required=True, description="艺术特色"),
                PropertyDefinition("typical_motifs", "List", description="典型母题"),
                PropertyDefinition("representative_works", "List", description="代表作品"),
                PropertyDefinition("representative_inheritors", "List", description="代表性传承人"),
                PropertyDefinition("style_tags", "List",
                                   description="风格标签（粗犷/细腻/写实/写意/传统/创新等）"),
                PropertyDefinition("influence_scope", "String", description="影响范围"),
                PropertyDefinition("status", "String",
                                   constraints={"enum": ["活跃", "待传承", "濒危", "已失传"]},
                                   description="传承状态"),
                PropertyDefinition("protection_level", "String",
                                   constraints={"enum": ["世界级非遗", "国家级非遗", "省级非遗",
                                                         "市级非遗", "未评级"]},
                                   description="非遗保护级别"),
                PropertyDefinition("embedding", "List"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["province"],
                ["status"],
                ["protection_level"],
            ],
            fulltext_indexes=[
                ["name", "characteristics", "history"],
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(region)",
                "EXISTS(characteristics)",
            ]
        )

        # 5. 文化象征/吉祥寓意
        self.entity_schemas[PaperCutEntityType.CULTURAL_SYMBOL] = EntitySchema(
            entity_type=PaperCutEntityType.CULTURAL_SYMBOL,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="象征名称（福/寿/喜/财等）"),
                PropertyDefinition("category", "String", required=True,
                                   constraints={"enum": ["吉祥", "避邪", "喜庆", "纪念",
                                                         "祈福", "崇拜", "教化", "装饰"]},
                                   description="象征类别"),
                PropertyDefinition("meaning", "String", required=True, description="寓意解释"),
                PropertyDefinition("typical_scenes", "List", description="典型应用场景"),
                PropertyDefinition("related_holidays", "List", description="关联节日"),
                PropertyDefinition("related_motifs", "List", description="常用母题组合"),
                PropertyDefinition("cultural_origin", "String", description="文化起源"),
                PropertyDefinition("positive_keywords", "List", description="正向关联关键词"),
                PropertyDefinition("embedding", "List"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["category"],
            ],
            fulltext_indexes=[
                ["name", "meaning", "cultural_origin"],
            ],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(category)",
                "EXISTS(meaning)",
            ]
        )

        # 6. 传承人（审核者）
        self.entity_schemas[PaperCutEntityType.PAPER_CUT_INHERITOR] = EntitySchema(
            entity_type=PaperCutEntityType.PAPER_CUT_INHERITOR,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="传承人姓名"),
                PropertyDefinition("title", "String", description="称号（国家级/省级传承人等）"),
                PropertyDefinition("gender", "String", constraints={"enum": ["男", "女"]}),
                PropertyDefinition("birth_year", "Integer", description="出生年份"),
                PropertyDefinition("region", "String", description="所在地区"),
                PropertyDefinition("style_affiliation", "List", description="所属流派"),
                PropertyDefinition("biography", "String", description="传承人简介"),
                PropertyDefinition("specialties", "List", description="专长领域"),
                PropertyDefinition("master_level", "String",
                                   constraints={"enum": ["初级", "中级", "高级", "大师级", "国宝级"]},
                                   description="技艺等级"),
                PropertyDefinition("years_of_experience", "Integer", constraints={"min": 0}, description="从业年限"),
                PropertyDefinition("representative_works", "List", description="代表作"),
                PropertyDefinition("awards", "List", description="荣誉奖项"),
                PropertyDefinition("contact", "Object", description="联系方式（脱敏）"),
                PropertyDefinition("status", "String",
                                   constraints={"enum": ["活跃", "半退休", "已故"]}),
                PropertyDefinition("is_reviewer", "Boolean", default_value=False,
                                   description="是否具有审核权限"),
                PropertyDefinition("reviewer_since", "Date", description="成为审核者时间"),
                PropertyDefinition("embedding", "List"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["region"],
                ["master_level"],
                ["title"],
                ["status"],
            ],
            fulltext_indexes=[
                ["name", "biography", "specialties"],
            ],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
            ]
        )

        # 7. 材料工具
        self.entity_schemas[PaperCutEntityType.MATERIAL] = EntitySchema(
            entity_type=PaperCutEntityType.MATERIAL,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("name", "String", required=True, description="名称"),
                PropertyDefinition("category", "String", required=True,
                                   constraints={"enum": ["纸", "刀", "刻板", "颜料",
                                                         "装裱", "辅助工具"]},
                                   description="材料类别"),
                PropertyDefinition("sub_type", "String", description="子类别"),
                PropertyDefinition("description", "String", description="描述"),
                PropertyDefinition("specifications", "List", description="规格参数"),
                PropertyDefinition("recommended_for", "List", description="适用技法"),
                PropertyDefinition("alternative_names", "List", description="别称"),
                PropertyDefinition("is_traditional", "Boolean", description="是否为传统材料"),
                PropertyDefinition("created_at", "Date"),
            ],
            indexes=[
                ["name"],
                ["category"],
                ["sub_type"],
            ],
            constraints=[
                "UNIQUE(id)",
                "EXISTS(name)",
                "EXISTS(category)",
            ]
        )

        # 8. AI生成记录
        self.entity_schemas[PaperCutEntityType.GENERATION_RECORD] = EntitySchema(
            entity_type=PaperCutEntityType.GENERATION_RECORD,
            properties=[
                PropertyDefinition("id", "String", required=True),
                PropertyDefinition("session_id", "String", description="会话ID"),
                PropertyDefinition("input_keywords", "String", required=True, description="用户输入关键词"),
                PropertyDefinition("rewritten_query", "String", description="改写后查询"),
                PropertyDefinition("constraint_parameters", "Object", description="约束参数（JSON，传给图像模型）"),
                PropertyDefinition("retrieval_context", "Object", description="RAG检索上下文摘要"),
                PropertyDefinition("output_image_url", "String", description="生成图像URL"),
                PropertyDefinition("output_thumbnail_url", "String", description="缩略图URL"),
                PropertyDefinition("ai_model_used", "String", description="使用的AI模型"),
                PropertyDefinition("ai_model_version", "String", description="模型版本"),
                PropertyDefinition("generation_params", "Object", description="生成参数快照"),
                PropertyDefinition("generation_duration_ms", "Integer", description="生成耗时(ms)"),
                PropertyDefinition("user_feedback", "Object", description="用户反馈 {rating, comment}"),
                PropertyDefinition("inheritor_review_status", "String",
                                   constraints={"enum": ["未审核", "审核中", "通过", "需修改", "驳回"]},
                                   description="传承人审核状态"),
                PropertyDefinition("inheritor_review_comment", "String", description="传承人审核意见"),
                PropertyDefinition("inheritor_id", "String", description="审核人ID"),
                PropertyDefinition("review_date", "Date", description="审核日期"),
                PropertyDefinition("is_used_for_training", "Boolean", default_value=False,
                                   description="是否已用于模型训练"),
                PropertyDefinition("copyright_notes", "String", description="版权说明"),
                PropertyDefinition("embedding", "List", description="输入语义向量（用于相似度分析）"),
                PropertyDefinition("created_at", "Date", description="创建时间"),
            ],
            indexes=[
                ["input_keywords"],
                ["inheritor_review_status"],
                ["ai_model_used"],
                ["is_used_for_training"],
                ["created_at"],
            ],
            fulltext_indexes=[
                ["input_keywords", "constraint_parameters"],
            ],
            vector_indexes=["embedding"],
            constraints=[
                "UNIQUE(id)",
            ]
        )

    # ========== 关系模式定义 ==========

    def _initialize_relation_schemas(self):
        """初始化所有剪纸领域关系模式"""

        # 1. 纹样包含母题
        self.relation_schemas[PaperCutRelationType.HAS_MOTIF] = RelationSchema(
            relation_type=PaperCutRelationType.HAS_MOTIF,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.PAPER_CUT_MOTIF],
            properties=[
                PropertyDefinition("prominence", "Float", constraints={"min": 0, "max": 1},
                                   description="显著程度"),
                PropertyDefinition("position", "String", description="在纹样中的位置"),
                PropertyDefinition("is_primary", "Boolean", default_value=False,
                                   description="是否为主母题"),
            ],
        )

        # 2. 纹样变体关系
        self.relation_schemas[PaperCutRelationType.HAS_VARIANT] = RelationSchema(
            relation_type=PaperCutRelationType.HAS_VARIANT,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            properties=[
                PropertyDefinition("variation_type", "String",
                                   constraints={"enum": ["简化", "繁化", "地域变体", "风格融合", "创新衍生"]},
                                   description="变体类型"),
                PropertyDefinition("description", "String", description="变体描述"),
                PropertyDefinition("similarity", "Float", constraints={"min": 0, "max": 1},
                                   description="相似度"),
            ],
            constraints=["NO_CYCLE"],
        )

        # 3. 纹样由子纹样组成
        self.relation_schemas[PaperCutRelationType.COMPOSED_OF] = RelationSchema(
            relation_type=PaperCutRelationType.COMPOSED_OF,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            properties=[
                PropertyDefinition("layer", "Integer", description="层次顺序"),
                PropertyDefinition("position", "Object", description="相对位置"),
                PropertyDefinition("scale", "Float", default_value=1.0, description="相对比例"),
            ],
            constraints=["NO_CYCLE", "HIERARCHICAL"],
        )

        # 4. 纹样使用技法
        self.relation_schemas[PaperCutRelationType.USES_TECHNIQUE] = RelationSchema(
            relation_type=PaperCutRelationType.USES_TECHNIQUE,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.TECHNIQUE],
            properties=[
                PropertyDefinition("is_primary", "Boolean", default_value=False,
                                   description="是否为主要技法"),
                PropertyDefinition("order", "Integer", description="技法应用顺序"),
                PropertyDefinition("details", "String", description="技法应用细节"),
            ],
        )

        # 5. 传承人擅长技法
        self.relation_schemas[PaperCutRelationType.SPECIALIZES_IN] = RelationSchema(
            relation_type=PaperCutRelationType.SPECIALIZES_IN,
            source_types=[PaperCutEntityType.PAPER_CUT_INHERITOR],
            target_types=[PaperCutEntityType.TECHNIQUE],
            properties=[
                PropertyDefinition("mastery_level", "String",
                                   constraints={"enum": ["入门", "熟练", "精通", "大师级"]},
                                   description="精熟程度"),
                PropertyDefinition("years", "Integer", description="研习年限"),
                PropertyDefinition("notable_works", "List", description="代表作"),
            ],
        )

        # 6. 纹样属于流派
        self.relation_schemas[PaperCutRelationType.BELONGS_TO_STYLE] = RelationSchema(
            relation_type=PaperCutRelationType.BELONGS_TO_STYLE,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.REGIONAL_STYLE],
            properties=[
                PropertyDefinition("influence_degree", "Float", constraints={"min": 0, "max": 1},
                                   description="流派影响程度"),
                PropertyDefinition("is_representative", "Boolean", default_value=False,
                                   description="是否为代表性作品"),
            ],
        )

        # 7. 流派影响关系
        self.relation_schemas[PaperCutRelationType.INFLUENCED_BY] = RelationSchema(
            relation_type=PaperCutRelationType.INFLUENCED_BY,
            source_types=[PaperCutEntityType.REGIONAL_STYLE],
            target_types=[PaperCutEntityType.REGIONAL_STYLE],
            properties=[
                PropertyDefinition("influence_type", "String",
                                   constraints={"enum": ["直接继承", "相互影响", "融合发展", "分支派生"]},
                                   description="影响类型"),
                PropertyDefinition("period", "String", description="影响时期"),
                PropertyDefinition("description", "String", description="影响描述"),
            ],
        )

        # 8. 纹样具有象征意义
        self.relation_schemas[PaperCutRelationType.HAS_SYMBOLISM] = RelationSchema(
            relation_type=PaperCutRelationType.HAS_SYMBOLISM,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN,
                          PaperCutEntityType.PAPER_CUT_MOTIF],
            target_types=[PaperCutEntityType.CULTURAL_SYMBOL],
            properties=[
                PropertyDefinition("significance", "Float", constraints={"min": 0, "max": 1},
                                   description="象征重要程度"),
                PropertyDefinition("cultural_note", "String", description="文化注释"),
            ],
        )

        # 9. 母题代表文化象征
        self.relation_schemas[PaperCutRelationType.REPRESENTS] = RelationSchema(
            relation_type=PaperCutRelationType.REPRESENTS,
            source_types=[PaperCutEntityType.PAPER_CUT_MOTIF],
            target_types=[PaperCutEntityType.CULTURAL_SYMBOL],
            properties=[
                PropertyDefinition("strength", "Float", constraints={"min": 0, "max": 1},
                                   description="关联强度"),
                PropertyDefinition("is_primary", "Boolean", default_value=True),
            ],
        )

        # 10. 传承人创作纹样
        self.relation_schemas[PaperCutRelationType.CREATED_BY] = RelationSchema(
            relation_type=PaperCutRelationType.CREATED_BY,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.PAPER_CUT_INHERITOR],
            properties=[
                PropertyDefinition("creation_date", "Date", description="创作日期"),
                PropertyDefinition("verification_status", "String",
                                   constraints={"enum": ["待确认", "已确认", "有争议"]},
                                   description="归属确认状态"),
                PropertyDefinition("notes", "String", description="备注"),
            ],
        )

        # 11. 传承人审核生成记录
        self.relation_schemas[PaperCutRelationType.REVIEWED_BY] = RelationSchema(
            relation_type=PaperCutRelationType.REVIEWED_BY,
            source_types=[PaperCutEntityType.GENERATION_RECORD],
            target_types=[PaperCutEntityType.PAPER_CUT_INHERITOR],
            properties=[
                PropertyDefinition("review_date", "Date", required=True, description="审核日期"),
                PropertyDefinition("review_status", "String", required=True,
                                   constraints={"enum": ["通过", "需修改", "驳回"]},
                                   description="审核结果"),
                PropertyDefinition("comments", "String", description="审核意见"),
                PropertyDefinition("suggested_improvements", "List", description="改进建议"),
            ],
        )

        # 12. 纹样需要材料
        self.relation_schemas[PaperCutRelationType.REQUIRES_MATERIAL] = RelationSchema(
            relation_type=PaperCutRelationType.REQUIRES_MATERIAL,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            target_types=[PaperCutEntityType.MATERIAL],
            properties=[
                PropertyDefinition("is_default", "Boolean", default_value=True),
                PropertyDefinition("specification_detail", "String", description="规格要求"),
            ],
        )

        # 13. 生成记录基于纹样
        self.relation_schemas[PaperCutRelationType.GENERATED_FROM] = RelationSchema(
            relation_type=PaperCutRelationType.GENERATED_FROM,
            source_types=[PaperCutEntityType.GENERATION_RECORD],
            target_types=[PaperCutEntityType.PAPER_CUT_PATTERN],
            properties=[
                PropertyDefinition("similarity_score", "Float", constraints={"min": 0, "max": 1},
                                   description="与参考纹样的相似度"),
                PropertyDefinition("modification_details", "String", description="修改说明"),
                PropertyDefinition("reference_weight", "Float", constraints={"min": 0, "max": 1},
                                   description="生成时参考该纹样的权重"),
            ],
        )

        # 14. 纹样具有文化遗产背景（连接文旅体系的桥梁）
        self.relation_schemas[PaperCutRelationType.HAS_CULTURAL_BACKGROUND] = RelationSchema(
            relation_type=PaperCutRelationType.HAS_CULTURAL_BACKGROUND,
            source_types=[PaperCutEntityType.PAPER_CUT_PATTERN,
                          PaperCutEntityType.PAPER_CUT_MOTIF,
                          PaperCutEntityType.REGIONAL_STYLE],
            target_types=[PaperCutEntityType.CULTURAL_SYMBOL],
            # 注意：这里的 CULTURAL_SYMBOL 是剪纸领域的文化象征
            # 与文旅体系的 CulturalHeritage 的桥接通过查询层实现
            properties=[
                PropertyDefinition("connection_type", "String",
                                   constraints={"enum": ["历史渊源", "文化同源", "节日关联", "地域关联"]},
                                   description="关联类型"),
                PropertyDefinition("description", "String"),
            ],
        )

        # 15. 母题间关联关系
        self.relation_schemas[PaperCutRelationType.RELATED_MOTIF] = RelationSchema(
            relation_type=PaperCutRelationType.RELATED_MOTIF,
            source_types=[PaperCutEntityType.PAPER_CUT_MOTIF],
            target_types=[PaperCutEntityType.PAPER_CUT_MOTIF],
            properties=[
                PropertyDefinition("relation_type", "String", required=True,
                                   constraints={"enum": ["演化", "组合", "派生", "相似", "对比"]},
                                   description="关联类型"),
                PropertyDefinition("description", "String", description="关联描述"),
                PropertyDefinition("strength", "Float", constraints={"min": 0, "max": 1},
                                   description="关联强度"),
            ],
        )

    # ========== 推理规则 ==========

    def _initialize_inference_rules(self):
        """初始化剪纸领域的推理规则"""

        self.inference_rules = [
            {
                "name": "母题象征传递",
                "description": "如果一个纹样包含母题A，且母题A代表文化象征B，则纹样具有文化象征B",
                "rule": """
                    MATCH (p:PaperCutPattern)-[:HAS_MOTIF]->(m:PaperCutMotif)-[:REPRESENTS]->(s:CulturalSymbol)
                    WHERE NOT (p)-[:HAS_SYMBOLISM]->(s)
                    MERGE (p)-[:HAS_SYMBOLISM {
                        significance: 0.5,
                        cultural_note: '通过母题' + m.name + '推理',
                        inferred: true
                    }]->(s)
                """
            },
            {
                "name": "流派技法推荐",
                "description": "同一流派的纹样倾向于使用相似的技法",
                "rule": """
                    MATCH (p1:PaperCutPattern)-[:BELONGS_TO_STYLE]->(style:RegionalStyle)
                    MATCH (p1)-[:USES_TECHNIQUE]->(t:Technique)
                    MATCH (p2:PaperCutPattern)-[:BELONGS_TO_STYLE]->(style)
                    WHERE p1 <> p2 AND NOT (p2)-[:USES_TECHNIQUE]->(t)
                    MERGE (p2)-[:USES_TECHNIQUE {
                        is_primary: false,
                        inferred: true,
                        details: '基于' + style.name + '流派推荐'
                    }]->(t)
                """
            },
            {
                "name": "节日象征关联",
                "description": "适用于特定场景的纹样与该场景的文化象征互为关联",
                "rule": """
                    MATCH (p:PaperCutPattern)
                    WHERE p.applicable_scenes IS NOT NULL
                    MATCH (s:CulturalSymbol)
                    WHERE s.related_holidays IS NOT NULL
                    WITH p, s
                    WHERE any(scene IN p.applicable_scenes WHERE scene IN s.related_holidays)
                    AND NOT (p)-[:HAS_SYMBOLISM]->(s)
                    MERGE (p)-[:HAS_SYMBOLISM {
                        significance: 0.4,
                        inferred: true,
                        cultural_note: '基于应用场景与节日关联推理'
                    }]->(s)
                """
            },
            {
                "name": "传承人流派隶属",
                "description": "传承人擅长的技法所属流派自动关联",
                "rule": """
                    MATCH (i:PaperCutInheritor)-[:SPECIALIZES_IN]->(t:Technique)
                    MATCH (style:RegionalStyle)
                    WHERE t.origin_region CONTAINS style.region
                    AND NOT (i)-[:BELONGS_TO_STYLE]->(style)
                    MERGE (i)-[:BELONGS_TO_STYLE {
                        inferred: true
                    }]->(style)
                """
            },
            {
                "name": "相似母题推荐",
                "description": "包含相同母题的纹样互为相关",
                "rule": """
                    MATCH (p1:PaperCutPattern)-[:HAS_MOTIF]->(m:PaperCutMotif)<-[:HAS_MOTIF]-(p2:PaperCutPattern)
                    WHERE p1 <> p2 AND NOT (p1)-[:HAS_VARIANT]->(p2)
                    MERGE (p1)-[:HAS_VARIANT {
                        variation_type: '相似',
                        similarity: 0.5,
                        inferred: true
                    }]->(p2)
                """
            },
        ]

    # ========== Schema 生成 ==========

    def generate_neo4j_schema(self) -> str:
        """生成Neo4j Schema创建语句"""
        cypher_statements = []

        for entity_type, schema in self.entity_schemas.items():
            # 唯一性约束
            cypher_statements.append(
                f"CREATE CONSTRAINT {entity_type.value}_id_unique IF NOT EXISTS "
                f"FOR (n:{entity_type.value}) REQUIRE n.id IS UNIQUE;"
            )

            # 属性索引
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

    # ========== 验证逻辑 ==========

    def validate_entity(self, entity_type: PaperCutEntityType, properties: Dict) -> List[str]:
        """验证实体的属性合规性"""
        errors = []
        schema = self.entity_schemas.get(entity_type)

        if not schema:
            errors.append(f"未知的实体类型: {entity_type}")
            return errors

        for prop_def in schema.properties:
            # 检查必填属性
            if prop_def.required and prop_def.name not in properties:
                errors.append(f"缺少必填属性 [{prop_def.name}]: {prop_def.description}")
                continue

            if prop_def.name in properties:
                value = properties[prop_def.name]

                # 检查枚举约束
                if "enum" in prop_def.constraints:
                    if isinstance(value, list):
                        for v in value:
                            if v not in prop_def.constraints["enum"]:
                                errors.append(
                                    f"属性 {prop_def.name} 的值 '{v}' 不在允许范围: "
                                    f"{prop_def.constraints['enum']}"
                                )
                    elif value not in prop_def.constraints["enum"]:
                        errors.append(
                            f"属性 {prop_def.name} 的值 '{value}' 不在允许范围: "
                            f"{prop_def.constraints['enum']}"
                        )

                # 检查范围约束
                if "min" in prop_def.constraints and isinstance(value, (int, float)):
                    if value < prop_def.constraints["min"]:
                        errors.append(
                            f"属性 {prop_def.name} 的值 {value} 小于最小值 {prop_def.constraints['min']}"
                        )

                if "max" in prop_def.constraints and isinstance(value, (int, float)):
                    if value > prop_def.constraints["max"]:
                        errors.append(
                            f"属性 {prop_def.name} 的值 {value} 大于最大值 {prop_def.constraints['max']}"
                        )

        return errors

    # ========== 导出 ==========

    def to_json(self) -> str:
        """导出本体为JSON格式（用于前端工具/文档生成）"""
        ontology_dict = {
            "domain": "非遗剪纸",
            "entity_schemas": {},
            "relation_schemas": {},
            "inference_rules": self.inference_rules,
        }

        for entity_type, schema in self.entity_schemas.items():
            ontology_dict["entity_schemas"][entity_type.value] = {
                "properties": [
                    {
                        "name": p.name,
                        "data_type": p.data_type,
                        "required": p.required,
                        "default_value": p.default_value,
                        "constraints": p.constraints,
                        "description": p.description,
                    }
                    for p in schema.properties
                ],
                "indexes": schema.indexes,
                "fulltext_indexes": schema.fulltext_indexes,
                "vector_indexes": schema.vector_indexes,
                "constraints": schema.constraints,
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
                        "description": p.description,
                    }
                    for p in schema.properties
                ],
                "constraints": schema.constraints,
            }

        return json.dumps(ontology_dict, ensure_ascii=False, indent=2)

    def summary(self) -> Dict:
        """输出本体概览统计"""
        entity_counts = {}
        for etype, schema in self.entity_schemas.items():
            entity_counts[etype.value] = {
                "properties": len(schema.properties),
                "indexes": len(schema.indexes),
                "fulltext_indexes": len(schema.fulltext_indexes),
                "vector_indexes": len(schema.vector_indexes),
                "required_fields": [p.name for p in schema.properties if p.required],
            }

        relation_counts = {}
        for rtype, schema in self.relation_schemas.items():
            relation_counts[rtype.value] = {
                "source_types": [t.value for t in schema.source_types],
                "target_types": [t.value for t in schema.target_types],
                "properties": len(schema.properties),
                "constraints": schema.constraints,
            }

        return {
            "domain": "非遗剪纸",
            "entity_types": len(self.entity_schemas),
            "relation_types": len(self.relation_schemas),
            "inference_rules": len(self.inference_rules),
            "entities": entity_counts,
            "relations": relation_counts,
        }

    # ========== 桥接查询生成 ==========

    def generate_cross_domain_query(self, keyword: str) -> str:
        """
        生成跨文旅-剪纸领域的查询Cypher
        这是连接文旅与剪纸两个知识图谱的桥梁
        """
        return f"""
        // 跨领域查询：从文旅文化遗产桥接到剪纸纹样
        MATCH (ch:CulturalHeritage)
        WHERE ch.name CONTAINS '{keyword}'
           OR ch.description CONTAINS '{keyword}'
           OR ch.cultural_value CONTAINS '{keyword}'

        // 通过纹样的文化背景桥接
        OPTIONAL MATCH (p:PaperCutPattern)-[:HAS_CULTURAL_BACKGROUND]->(cs:CulturalSymbol)
        WHERE cs.name CONTAINS '{keyword}'
           OR cs.meaning CONTAINS '{keyword}'

        // 通过文化遗产关联的纹样
        OPTIONAL MATCH (p2:PaperCutPattern)
        WHERE p2.cultural_context CONTAINS '{keyword}'
           OR p2.meaning CONTAINS '{keyword}'

        WITH ch, p, p2
        RETURN
            ch.name AS heritage_name,
            ch.heritage_type AS heritage_type,
            COALESCE(p.name, p2.name) AS related_pattern,
            p.meaning AS pattern_meaning
        LIMIT 20
        """


# ========== 使用示例 ==========

if __name__ == "__main__":
    # 创建本体实例
    ontology = PaperCutOntology()

    # 输出概览
    summary = ontology.summary()
    print("=" * 60)
    print("非遗剪纸知识图谱本体")
    print("=" * 60)
    print(f"实体类型: {summary['entity_types']} 种")
    print(f"关系类型: {summary['relation_types']} 种")
    print(f"推理规则: {summary['inference_rules']} 条")
    print()

    for etype, info in summary["entities"].items():
        print(f"  - {etype}")
        print(f"     属性: {info['properties']}个, "
              f"索引: {info['indexes']}个, "
              f"全文索引: {info['fulltext_indexes']}个, "
              f"向量索引: {info['vector_indexes']}个")
        print(f"     必填字段: {', '.join(info['required_fields'])}")
        print()

    for rtype, info in summary["relations"].items():
        print(f"  - {rtype}")
        print(f"     源: {', '.join(info['source_types'])} -> 目标: {', '.join(info['target_types'])}")
        print(f"     属性: {info['properties']}个, 约束: {info['constraints']}")
        print()

    # 生成Neo4j Schema
    neo4j_schema = ontology.generate_neo4j_schema()
    print(f"\nNeo4j Schema (共 {len(neo4j_schema.splitlines())} 条语句):")
    for line in neo4j_schema.split("\n")[:10]:
        print(f"  {line}")
    print(f"  ...")
    print()

    # 验证示例
    print("=" * 60)
    print("验证示例")
    print("=" * 60)
    test_pattern = {
        "id": "pc_001",
        "name": "龙凤呈祥",
        "category": "传统纹样",
        "description": "经典婚庆剪纸，龙凤环绕喜字",
        "symmetry_type": "轴对称",
        "complexity": "复杂",
        "status": "草稿",
    }
    errors = ontology.validate_entity(PaperCutEntityType.PAPER_CUT_PATTERN, test_pattern)
    if errors:
        print("验证失败:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("纹样实体验证通过")

    # 导出一部分JSON
    json_out = ontology.to_json()
    print(f"\nJSON导出大小: {len(json_out)} 字符")

    # 跨领域查询示例
    print(f"\n跨领域查询示例 (关键词: '剪纸'):")
    print(ontology.generate_cross_domain_query("剪纸"))
