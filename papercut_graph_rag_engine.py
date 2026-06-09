"""
非遗剪纸 — Graph RAG 约束生成引擎
接收用户输入关键词，检索剪纸知识图谱，输出结构化约束参数传递给图像生成模型

核心链路:
  用户输入关键词
  → QueryAnalyzer: 意图分类 + 实体识别 + 查询扩展
  → KnowledgeBase: 加载知识（本地JSON 或 Neo4j）
  → Retriever: 多路检索（语义匹配 + 图遍历 + 推理）
  → ConstraintGenerator: 融合生成结构化参数
  → 输出: { motifs, style, technique, colors, symmetry, ... }
"""

import json
import re
import logging
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter


# ========== 数据结构 ==========

@dataclass
class QueryAnalysis:
    """查询分析结果"""
    raw_input: str
    intents: List[str]                       # 意图标签
    recognized_entities: Dict[str, List[str]] = field(default_factory=dict)  # {type: [names]}
    expanded_terms: List[str] = field(default_factory=list)     # 扩展关键词
    negative_terms: List[str] = field(default_factory=list)     # 排除关键词
    confidence: float = 1.0

    def to_dict(self):
        return asdict(self)


@dataclass
class RetrievalResult:
    """检索结果"""
    entity_id: str
    entity_type: str
    name: str
    score: float
    source: str               # exact_match, semantic, graph, inference
    matched_field: str = ""
    properties: Dict = field(default_factory=dict)


@dataclass
class ConstraintParameters:
    """传递给图像生成模型的结构化约束参数"""
    # ---- 核心纹样约束 ----
    motifs: List[Dict] = field(default_factory=list)          # [{"name": "龙纹", "prominence": 0.9, "position": "主图"}, ...]
    primary_motif: Optional[str] = None                       # 主母题
    secondary_motifs: List[str] = field(default_factory=list) # 辅助母题

    # ---- 风格约束 ----
    style: Optional[str] = None                               # 流派
    style_tags: List[str] = field(default_factory=list)       # 风格标签（粗犷/细腻/写实/写意）

    # ---- 技法约束 ----
    technique: Optional[str] = None                           # 主要技法
    alternative_techniques: List[str] = field(default_factory=list)

    # ---- 色彩约束 ----
    recommended_colors: List[str] = field(default_factory=list)
    color_count: Optional[int] = None
    color_scheme: Optional[str] = None                        # 单色/双色/套色

    # ---- 构图约束 ----
    symmetry_type: Optional[str] = None                       # 中心对称/轴对称/旋转对称/不对称
    complexity: Optional[str] = None                          # 简单/中等/复杂/极复杂

    # ---- 文化语义 ----
    meaning: Optional[str] = None                             # 吉祥寓意
    applicable_scenes: List[str] = field(default_factory=list)
    cultural_context: Optional[str] = None

    # ---- 参考图 ----
    reference_image_urls: List[str] = field(default_factory=list)

    # ---- 聚合 ----
    generation_prompt: Optional[str] = None                   # 汇总生成的提示词（传给图像模型）
    ranking_reason: Optional[str] = None                      # 为什么推荐这个方案

    def to_dict(self):
        return asdict(self)

    def to_generation_prompt(self) -> str:
        """生成自然语言提示词（可直接传给图像模型）"""
        parts = []
        parts.append("中国传统剪纸")

        if self.style:
            parts.append(f"{self.style}风格")

        if self.technique:
            parts.append(f"{self.technique}技法")

        if self.primary_motif:
            parts.append(f"主题: {self.primary_motif}")
        if self.secondary_motifs:
            parts.append(f"辅纹: {'、'.join(self.secondary_motifs)}")

        if self.recommended_colors:
            parts.append(f"色彩: {'、'.join(self.recommended_colors)}")

        if self.symmetry_type:
            parts.append(self.symmetry_type)

        if self.complexity:
            parts.append(f"复杂度: {self.complexity}")

        if self.meaning:
            parts.append(f"寓意: {self.meaning}")

        if self.applicable_scenes:
            parts.append(f"适用: {'、'.join(self.applicable_scenes[:3])}")

        return "，".join(parts)


# ========== 知识库（内存版）==========

class PaperCutKnowledgeBase:
    """
    剪纸知识库
    - 从管道输出的 JSON 文件加载数据
    - 也可以连接到 Neo4j（预留）
    - 建立内存索引：类型索引、名称索引、标签索引、关键词索引
    """

    def __init__(self, entities_path: str = "", relations_path: str = "", logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()

        # 原始数据
        self.entities: List[Dict] = []
        self.relations: List[Dict] = []

        # 索引
        self._by_type: Dict[str, List[Dict]] = {}
        self._by_id: Dict[str, Dict] = {}
        self._by_name: Dict[str, List[Dict]] = {}
        self._keyword_index: Dict[str, List[Tuple[str, float, str]]] = {}  # word → [(id, score, field)]
        self._relation_index: Dict[str, List[Dict]] = {}   # source_id → relations
        self._reverse_relation_index: Dict[str, List[Dict]] = {}  # target_id → relations
        self._adjacency: Dict[str, Set[str]] = {}  # id → {connected_ids}

        # 统计
        self.stats: Dict = {}
        self._loaded = False

        if entities_path and relations_path:
            self.load(entities_path, relations_path)

    def _setup_logger(self):
        logger = logging.getLogger("PaperCutKB")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
        return logger

    def load(self, entities_path: str, relations_path: str):
        """从 JSON 文件加载知识库"""
        try:
            with open(entities_path, "r", encoding="utf-8") as f:
                raw_entities = json.load(f)
            with open(relations_path, "r", encoding="utf-8") as f:
                self.relations = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"知识库文件未找到: {e}")
            return
        except json.JSONDecodeError as e:
            self.logger.error(f"知识库文件格式错误: {e}")
            return

        # 归一化：管道输出的实体字段在 properties 里，展开到顶层
        # 种子数据加载的实体字段直接在顶层
        self.entities = []
        for entity in raw_entities:
            props = entity.get("properties", {})
            if props:
                # 管道输出格式: {entity_type, name, properties:{id,name,...}, ...}
                normalized = dict(props)  # 展开 properties 内容到顶层
                normalized["_entity_type"] = entity.get("entity_type",
                                           entity.get("_entity_type", "Unknown"))
                # 保留顶层字段，但 properties 内的覆盖
                for k, v in entity.items():
                    if k != "properties":
                        normalized[k] = v
                self.entities.append(normalized)
            else:
                # 已经是扁平格式（种子数据加载）
                entity["_entity_type"] = entity.get("_entity_type",
                                                    entity.get("entity_type", "Unknown"))
                self.entities.append(entity)

        self._build_index()
        self._loaded = True
        self.logger.info(f"知识库已加载: {len(self.entities)} 实体, {len(self.relations)} 关系")

    def load_from_pipeline_output(self, output_dir: str = "./pipeline_output"):
        """从管道输出目录自动加载最新的输出文件"""
        output_path = Path(output_dir)
        if not output_path.exists():
            self.logger.warning(f"输出目录不存在: {output_dir}")
            return

        # 找最新的报告
        reports = sorted(output_path.glob("report_run_*.json"))
        if not reports:
            self.logger.warning(f"输出目录中无报告文件: {output_dir}")
            return

        latest_report = reports[-1]
        try:
            with open(latest_report, "r", encoding="utf-8") as f:
                report = json.load(f)
        except Exception as e:
            self.logger.error(f"读取报告文件失败: {e}")
            return

        entities_file = report.get("output_files", {}).get("entities", "")
        relations_file = report.get("output_files", {}).get("relations", "")
        if entities_file and relations_file:
            self.load(entities_file, relations_file)

    def load_from_seed_data(self):
        """直接从 seed_data 模块加载"""
        try:
            from papercut_seed_data import get_all_seed_data
        except ImportError:
            self.logger.warning("无法加载 seed_data 模块")
            return

        seed = get_all_seed_data()

        # 从种子数据构建实体列表
        entities = []
        for item in seed.get("patterns", []):
            item["_entity_type"] = "PaperCutPattern"
            entities.append(item)
        for item in seed.get("motifs", []):
            item["_entity_type"] = "PaperCutMotif"
            entities.append(item)
        for item in seed.get("techniques", []):
            item["_entity_type"] = "Technique"
            entities.append(item)
        for item in seed.get("regional_styles", []):
            item["_entity_type"] = "RegionalStyle"
            entities.append(item)
        for item in seed.get("cultural_symbols", []):
            item["_entity_type"] = "CulturalSymbol"
            entities.append(item)
        for item in seed.get("inheritors", []):
            item["_entity_type"] = "PaperCutInheritor"
            entities.append(item)
        for item in seed.get("materials", []):
            item["_entity_type"] = "Material"
            entities.append(item)

        self.entities = entities

        # 从映射表构建关系
        relations = []
        for map_name, source_type, target_type, rel_type in [
            ("pattern_motif_map", "PaperCutPattern", "PaperCutMotif", "HAS_MOTIF"),
            ("pattern_technique_map", "PaperCutPattern", "Technique", "USES_TECHNIQUE"),
            ("pattern_style_map", "PaperCutPattern", "RegionalStyle", "BELONGS_TO_STYLE"),
            ("pattern_symbol_map", "PaperCutPattern", "CulturalSymbol", "HAS_SYMBOLISM"),
            ("pattern_material_map", "PaperCutPattern", "Material", "REQUIRES_MATERIAL"),
        ]:
            mapping = seed.get(map_name, {})
            for source_id, links in mapping.items():
                for link in links:
                    target_key = "motif_id" if "motif" in map_name else \
                                 "technique_id" if "technique" in map_name else \
                                 "style_id" if "style" in map_name else \
                                 "material_id" if "material" in map_name else \
                                 "target_id"
                    target_id = link.get(target_key, "")
                    props = {k: v for k, v in link.items() if k != target_key}
                    relations.append({
                        "relation_type": rel_type,
                        "source_id": source_id,
                        "target_id": target_id,
                        "source_type": source_type,
                        "target_type": target_type,
                        "properties": props,
                        "source": "seed_data",
                    })

        # motif_relations
        for rel in seed.get("motif_relations", []):
            relations.append({
                "relation_type": "RELATED_MOTIF",
                "source_id": rel["source_id"],
                "target_id": rel["target_id"],
                "source_type": "PaperCutMotif",
                "target_type": "PaperCutMotif",
                "properties": {
                    "relation_type": rel.get("relation_type", ""),
                    "description": rel.get("description", ""),
                    "strength": rel.get("strength", 0.5),
                },
                "source": "seed_data",
            })

        # 传承人→技法
        for item in seed.get("inheritors", []):
            inheritor_id = item.get("id", "")
            for speciality in item.get("specialties", []):
                for tech in seed.get("techniques", []):
                    if speciality in tech.get("name", "") or speciality in tech.get("description", ""):
                        relations.append({
                            "relation_type": "SPECIALIZES_IN",
                            "source_id": inheritor_id,
                            "target_id": tech["id"],
                            "source_type": "PaperCutInheritor",
                            "target_type": "Technique",
                            "properties": {"mastery_level": "大师级", "years": item.get("years_of_experience", 0)},
                            "source": "seed_data",
                        })

        self.relations = relations
        self._build_index()
        self._loaded = True
        self.logger.info(f"知识库已从种子数据加载: {len(self.entities)} 实体, {len(self.relations)} 关系")

    def _build_index(self):
        """构建内存索引"""
        self._by_type.clear()
        self._by_id.clear()
        self._by_name.clear()
        self._keyword_index.clear()
        self._relation_index.clear()
        self._reverse_relation_index.clear()
        self._adjacency.clear()

        # 实体索引
        for entity in self.entities:
            entity_type = entity.get("_entity_type", entity.get("entity_type", "Unknown"))
            entity_id = entity.get("id", "")
            name = entity.get("name", "")

            # 类型索引
            if entity_type not in self._by_type:
                self._by_type[entity_type] = []
            self._by_type[entity_type].append(entity)

            # ID 索引
            if entity_id:
                self._by_id[entity_id] = entity

            # 名称索引
            if name:
                if name not in self._by_name:
                    self._by_name[name] = []
                self._by_name[name].append(entity)

            # 关键词索引（从文本字段提取关键词）
            self._index_keywords(entity, entity_id)

            # 邻接表
            if entity_id not in self._adjacency:
                self._adjacency[entity_id] = set()

        # 关系索引
        for rel in self.relations:
            source_id = rel.get("source_id", "")
            target_id = rel.get("target_id", "")

            # 正向索引
            if source_id not in self._relation_index:
                self._relation_index[source_id] = []
            self._relation_index[source_id].append(rel)

            # 反向索引
            if target_id not in self._reverse_relation_index:
                self._reverse_relation_index[target_id] = []
            self._reverse_relation_index[target_id].append(rel)

            # 邻接表
            if source_id not in self._adjacency:
                self._adjacency[source_id] = set()
            if target_id not in self._adjacency:
                self._adjacency[target_id] = set()
            self._adjacency[source_id].add(target_id)
            self._adjacency[target_id].add(source_id)

    def _index_keywords(self, entity: Dict, entity_id: str):
        """从实体的关键文本字段提取关键词"""
        text_fields = ["name", "description", "meaning", "tags", "symbolic_meaning",
                       "cultural_context", "characteristics", "keywords"]
        seen_words = set()

        for field in text_fields:
            value = entity.get(field)
            if not value:
                continue

            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item not in seen_words:
                        seen_words.add(item)
                        if item not in self._keyword_index:
                            self._keyword_index[item] = []
                        self._keyword_index[item].append((entity_id, 1.0, field))
                continue

            if isinstance(value, str):
                # 精确值
                if value not in seen_words:
                    seen_words.add(value)
                    if value not in self._keyword_index:
                        self._keyword_index[value] = []
                    self._keyword_index[value].append((entity_id, 1.0, field))

                # 拆分关键词（中文分词简化版）
                for word in self._simple_segment(value):
                    if len(word) < 2 or word in seen_words:
                        continue
                    seen_words.add(word)
                    if word not in self._keyword_index:
                        self._keyword_index[word] = []
                    self._keyword_index[word].append((entity_id, 0.6, field))

    def _simple_segment(self, text: str) -> List[str]:
        """简易中文分词：基于常见剪纸领域词典"""
        # 剪纸领域专有名词优先匹配
        domain_terms = [
            "吉祥如意", "龙凤呈祥", "龙凤", "福寿双全", "喜上眉梢", "年年有余",
            "花开富贵", "五福捧寿", "麒麟送子", "连年有余", "福寿", "富贵",
            "龙凤", "麒麟", "鲤鱼", "莲花", "牡丹", "梅花", "兰花", "竹子", "菊花",
            "喜鹊", "蝙蝠", "蝴蝶", "龙纹", "凤纹", "云纹", "寿桃", "福字",
            "阴刻", "阳刻", "套色", "染色", "单色", "剪刻",
            "蔚县", "扬州", "陕北", "佛山", "高密", "漳浦",
            "春节", "婚庆", "祝寿", "乔迁", "镇宅", "祈福", "辟邪",
            "传统", "吉祥", "喜庆", "生肖", "年俗", "瑞兽",
            "对称", "团花", "窗花", "门笺", "长卷",
        ]

        result = []
        # 先匹配专有名词
        remaining = text
        matched_positions = set()

        for term in sorted(domain_terms, key=len, reverse=True):
            if term in remaining:
                result.append(term)
                # 标记已匹配的位置
                idx = remaining.index(term)
                for i in range(idx, idx + len(term)):
                    matched_positions.add(i)

        # 单字分词（仅取有意义的字词组合）
        for i, char in enumerate(remaining):
            if char not in matched_positions:
                if len(char.strip()) > 0:
                    result.append(char)

        return result

    # ========== 查询方法 ==========

    def is_loaded(self) -> bool:
        return self._loaded

    def get_all_of_type(self, entity_type: str) -> List[Dict]:
        return self._by_type.get(entity_type, [])

    def get_by_id(self, entity_id: str) -> Optional[Dict]:
        return self._by_id.get(entity_id)

    def get_by_name(self, name: str) -> List[Dict]:
        return self._by_name.get(name, [])

    def search_by_name_fuzzy(self, keyword: str) -> List[Tuple[Dict, float, str]]:
        """模糊名称搜索"""
        results = []
        keyword_lower = keyword.lower()
        for entity in self.entities:
            name = entity.get("name", "")
            full_name = entity.get("full_name", "")
            score = 0

            if name == keyword or full_name == keyword:
                score = 1.0
            elif keyword in name or keyword in full_name:
                score = 0.8
            elif keyword_lower in name.lower() or keyword_lower in full_name.lower():
                score = 0.7
            else:
                # 部分匹配
                kw_chars = set(keyword)
                name_chars = set(name)
                overlap = len(kw_chars & name_chars) / max(len(kw_chars), 1)
                if overlap > 0.5:
                    score = overlap * 0.4

            if score > 0:
                entity_type = entity.get("_entity_type", entity.get("entity_type", ""))
                results.append((entity, score, "name"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:20]

    def search_by_keyword(self, keyword: str) -> List[Tuple[Dict, float, str]]:
        """关键词索引搜索"""
        results = []
        seen_ids = set()

        # 直接命中关键词索引
        for word, entries in self._keyword_index.items():
            if keyword in word:
                for entity_id, score, field in entries:
                    entity = self._by_id.get(entity_id)
                    if entity and entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        results.append((entity, score, f"keyword:{field}"))

        # 命中部分词
        for word, entries in self._keyword_index.items():
            if any(char in word for char in keyword if len(keyword) >= 2):
                for entity_id, score, field in entries:
                    entity = self._by_id.get(entity_id)
                    if entity and entity_id not in seen_ids:
                        seen_ids.add(entity_id)
                        results.append((entity, score * 0.5, f"partial:{field}"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:30]

    def get_relations(self, entity_id: str) -> List[Dict]:
        return self._relation_index.get(entity_id, [])

    def get_reverse_relations(self, entity_id: str) -> List[Dict]:
        return self._reverse_relation_index.get(entity_id, [])

    def get_connected_ids(self, entity_id: str) -> Set[str]:
        return self._adjacency.get(entity_id, set())

    def traverse(self, start_ids: Set[str], max_depth: int = 2) -> Dict[str, List[Dict]]:
        """图遍历：从起始节点出发，沿关系探索"""
        visited = set(start_ids)
        frontier = set(start_ids)
        result: Dict[str, List[Dict]] = {}  # entity_id → [related_entities]

        for depth in range(max_depth):
            next_frontier = set()
            for eid in frontier:
                connected = self.get_connected_ids(eid)
                for cid in connected:
                    if cid not in visited:
                        visited.add(cid)
                        next_frontier.add(cid)
                        entity = self._by_id.get(cid)
                        if entity:
                            if eid not in result:
                                result[eid] = []
                            result[eid].append(entity)
            frontier = next_frontier
            if not frontier:
                break

        return result

    def get_stats(self) -> Dict:
        """返回知识库统计"""
        type_counts = {}
        for entity in self.entities:
            t = entity.get("_entity_type", entity.get("entity_type", "Unknown"))
            type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "by_type": type_counts,
            "keywords_indexed": len(self._keyword_index),
            "loaded": self._loaded,
        }


# ========== 查询分析器 ==========

class PaperCutQueryAnalyzer:
    """
    剪纸查询分析器
    - 意图识别
    - 实体识别（母题、技法、流派、场景等）
    - 查询扩展（同义词、关联词）
    """

    # 意图 → 关键词映射
    INTENT_PATTERNS: Dict[str, List[str]] = {
        "婚庆": ["婚庆", "结婚", "婚礼", "嫁娶", "喜庆", "龙凤", "双喜", "囍"],
        "祝寿": ["祝寿", "寿辰", "生日", "长寿", "寿桃", "福寿", "长辈", "祝寿"],
        "春节": ["春节", "过年", "新春", "除夕", "年俗", "春联", "窗花", "福"],
        "节庆": ["元宵", "端午", "中秋", "重阳", "节日", "节庆", "灯笼"],
        "乔迁": ["乔迁", "搬家", "新房", "新居", "迁居"],
        "镇宅": ["镇宅", "辟邪", "驱邪", "避邪", "平安", "保佑", "门神"],
        "祈福": ["祈福", "许愿", "保佑", "好运", "如意", "吉祥"],
        "生肖": ["生肖", "本命年", "属相", "龙年", "虎年", "兔年", "生肖"],
        "装饰": ["装饰", "美化", "布置", "贴花", "装裱", "挂件"],
        "礼品": ["送礼", "礼品", "伴手礼", "馈赠", "送人"],
        "教学": ["教学", "学习", "教程", "入门", "步骤", "方法"],
    }

    # 母题关键词
    MOTIF_KEYWORDS = {
        "动物": ["龙", "凤", "麒麟", "狮子", "虎", "鹿", "鹤", "蝙蝠", "蝴蝶",
                 "喜鹊", "燕子", "鸳鸯", "鱼", "鲤鱼", "金鱼", "龟", "蟾蜍"],
        "植物": ["牡丹", "梅花", "兰花", "竹子", "菊花", "莲花", "荷花", "梅花",
                 "灵芝", "松树", "石榴", "葫芦", "寿桃", "桃", "葡萄", "水仙"],
        "几何": ["万字", "回纹", "云纹", "水纹", "铜钱", "如意", "方胜", "盘长"],
        "人物": ["福星", "寿星", "财神", "童子", "仕女", "戏曲", "历史人物"],
        "文字": ["福", "寿", "喜", "财", "禄", "吉祥", "如意"],
    }

    # 风格关键词
    STYLE_KEYWORDS = {
        "蔚县剪纸": ["蔚县", "河北", "北方", "刻纸", "染色", "色彩丰富"],
        "扬州剪纸": ["扬州", "江苏", "南方", "细腻", "雅致", "文人", "花卉"],
        "陕北剪纸": ["陕北", "陕西", "粗犷", "古朴", "民俗", "图腾", "原始"],
        "佛山剪纸": ["佛山", "广东", "金碧", "铜凿", "金纸", "华丽"],
        "高密剪纸": ["高密", "山东", "拙朴", "水浒", "金石"],
        "漳浦剪纸": ["漳浦", "福建", "纤巧", "秀丽", "女性"],
    }

    # 技法关键词
    TECHNIQUE_KEYWORDS = {
        "阴刻": ["阴刻", "刻线", "白线"],
        "阳刻": ["阳刻", "红面", "块面"],
        "阴阳刻结合": ["阴阳刻", "阴阳", "虚实"],
        "套色剪纸": ["套色", "彩色", "多色", "拼色"],
        "染色剪纸": ["染色", "点染", "上色", "蔚县染色"],
        "单色剪纸": ["单色", "纯色", "红色"],
    }

    # 对称类型关键词
    SYMMETRY_KEYWORDS = {
        "轴对称": ["对称", "左右对称", "对折"],
        "中心对称": ["中心对称", "团花", "圆形", "旋转对称"],
        "不对称": ["自由式", "不对称", "写实"],
    }

    # 同义词/近义词扩展
    SYNONYM_MAP = {
        "龙": ["龙纹", "金龙", "腾龙", "飞龙", "祥龙"],
        "凤": ["凤纹", "凤凰", "丹凤", "彩凤"],
        "福": ["福气", "福运", "幸福", "蝙蝠"],
        "寿": ["长寿", "寿星", "延年", "不老"],
        "喜": ["喜庆", "欢乐", "吉庆", "双喜"],
        "红": ["红色", "大红", "赤色", "朱红"],
        "金": ["金色", "金黄", "金碧"],
        "结婚": ["婚庆", "婚嫁", "婚礼", "嫁娶"],
        "过年": ["春节", "新春", "除夕", "年俗"],
        "龙年": ["生肖龙", "龙", "龙纹"],
        "北方": ["蔚县", "陕北", "高密", "粗犷"],
        "南方": ["扬州", "佛山", "漳浦", "细腻"],
    }

    def analyze(self, raw_input: str) -> QueryAnalysis:
        """分析用户输入"""
        entities: Dict[str, List[str]] = {
            "motif": [],
            "style": [],
            "technique": [],
            "scene": [],
            "symbol": [],
            "symmetry": [],
        }

        intents = self._detect_intents(raw_input)
        entities = self._recognize_entities(raw_input)
        expanded = self._expand_query(raw_input)

        return QueryAnalysis(
            raw_input=raw_input,
            intents=intents,
            recognized_entities=entities,
            expanded_terms=list(set(expanded)),
            confidence=1.0 if intents else 0.5,
        )

    def _detect_intents(self, text: str) -> List[str]:
        """检测意图"""
        intents = []
        for intent, keywords in self.INTENT_PATTERNS.items():
            if any(kw in text for kw in keywords):
                intents.append(intent)

        # 如果没有匹配到特定意图，标记为"通用"
        if not intents:
            intents.append("通用")

        return intents

    def _recognize_entities(self, text: str) -> Dict[str, List[str]]:
        """识别各类实体"""
        entities: Dict[str, List[str]] = {
            "motif": [], "style": [], "technique": [],
            "scene": [], "symbol": [], "symmetry": [],
        }

        # 识别母题
        for category, keywords in self.MOTIF_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    entities["motif"].append(kw)

        # 识别风格/流派
        for style, keywords in self.STYLE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                entities["style"].append(style)

        # 识别技法
        for tech, keywords in self.TECHNIQUE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                entities["technique"].append(tech)

        # 识别对称类型
        for sym, keywords in self.SYMMETRY_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                entities["symmetry"].append(sym)

        # 识别场景
        for scene, keywords in self.INTENT_PATTERNS.items():
            if any(kw in text for kw in keywords):
                entities["scene"].append(scene)

        return entities

    def _expand_query(self, text: str) -> List[str]:
        """扩展查询（同义词+关联词）"""
        expanded = set()
        for word, synonyms in self.SYNONYM_MAP.items():
            if word in text:
                expanded.update(synonyms)
        return list(expanded)


# ========== 检索器 ==========

class PaperCutRetriever:
    """
    剪纸知识检索器
    三路检索：精确匹配 → 关键词匹配 → 图遍历关联
    """

    def __init__(self, knowledge_base: PaperCutKnowledgeBase, logger: Optional[logging.Logger] = None):
        self.kb = knowledge_base
        self.logger = logger or logging.getLogger("PaperCutRetriever")

    def retrieve(self, analysis: QueryAnalysis, top_k: int = 15) -> List[RetrievalResult]:
        """三路检索并合并"""
        result_map: Dict[str, RetrievalResult] = {}

        # ---- 路 1: 精确名称匹配 ----
        self._exact_match(analysis, result_map)

        # ---- 路 2: 关键词匹配 ----
        self._keyword_search(analysis, result_map)

        # ---- 路 3: 图遍历关联 ----
        if result_map:
            self._graph_traverse(set(result_map.keys()), result_map)

        # 排序
        results = sorted(result_map.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _exact_match(self, analysis: QueryAnalysis, result_map: Dict[str, RetrievalResult]):
        """精确名称匹配"""
        search_terms = []
        for motif_list in analysis.recognized_entities.values():
            search_terms.extend(motif_list)
        search_terms.extend(analysis.expanded_terms)

        # 直接匹配用户输入的完整关键词
        search_terms.append(analysis.raw_input)

        for term in search_terms:
            # 精确名称匹配
            matches = self.kb.search_by_name_fuzzy(term)
            for entity, score, field in matches:
                self._add_result(entity, score, "exact_match", result_map)

    def _keyword_search(self, analysis: QueryAnalysis, result_map: Dict[str, RetrievalResult]):
        """关键词索引搜索"""
        # 搜索原始输入
        matches = self.kb.search_by_keyword(analysis.raw_input)
        for entity, score, field in matches:
            existing = result_map.get(entity.get("id", ""))
            if existing:
                existing.score = max(existing.score, score * 0.85)
            else:
                self._add_result(entity, score * 0.85, "semantic", result_map)

        # 搜索扩展词
        for term in analysis.expanded_terms:
            matches = self.kb.search_by_keyword(term)
            for entity, score, field in matches:
                entity_id = entity.get("id", "")
                if entity_id not in result_map:
                    self._add_result(entity, score * 0.7, "semantic", result_map)

        # 搜索已识别的实体名
        for category, terms in analysis.recognized_entities.items():
            for term in terms:
                matches = self.kb.search_by_keyword(term)
                for entity, score, field in matches:
                    entity_id = entity.get("id", "")
                    if entity_id not in result_map:
                        self._add_result(entity, score * 0.75, "semantic", result_map)

    def _graph_traverse(self, start_ids: Set[str], result_map: Dict[str, RetrievalResult]):
        """图遍历：从已匹配的节点探索关联节点"""
        traversed = self.kb.traverse(start_ids, max_depth=2)

        for source_id, related_entities in traversed.items():
            source_score = result_map.get(source_id, RetrievalResult("", "", "", 0, ""))
            base_score = source_score.score * 0.5

            for entity in related_entities:
                entity_id = entity.get("id", "")
                if entity_id not in result_map:
                    self._add_result(entity, base_score * 0.8, "graph", result_map)

    def _add_result(self, entity: Dict, score: float, source: str,
                    result_map: Dict[str, RetrievalResult]):
        """添加检索结果到结果集"""
        entity_id = entity.get("id", "")
        if not entity_id:
            return

        entity_type = entity.get("_entity_type", entity.get("entity_type", "Unknown"))
        name = entity.get("name", "")

        if entity_id in result_map:
            result_map[entity_id].score = max(result_map[entity_id].score, score)
            return

        result_map[entity_id] = RetrievalResult(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            score=round(score, 4),
            source=source,
            properties=entity,
        )


# ========== 约束推理引擎 ==========

class PaperCutConstraintReasoner:
    """
    剪纸约束推理引擎
    基于检索结果，应用领域规则，生成图像生成约束参数
    """

    def __init__(self, knowledge_base: PaperCutKnowledgeBase):
        self.kb = knowledge_base

    def generate(self, analysis: QueryAnalysis,
                 retrieval_results: List[RetrievalResult]) -> ConstraintParameters:
        """从检索结果推理生成约束参数"""
        params = ConstraintParameters()

        if not retrieval_results:
            return params

        # 步骤 1: 提取纹样/母题约束
        self._extract_motifs(analysis, retrieval_results, params)

        # 步骤 2: 提取风格约束
        self._extract_style(analysis, retrieval_results, params)

        # 步骤 3: 提取技法约束
        self._extract_technique(analysis, retrieval_results, params)

        # 步骤 4: 提取色彩约束
        self._extract_colors(analysis, retrieval_results, params)

        # 步骤 5: 提取构图约束
        self._extract_composition(analysis, retrieval_results, params)

        # 步骤 6: 提取文化语义
        self._extract_cultural(analysis, retrieval_results, params)

        # 步骤 7: 排序和推荐原因
        self._rank_and_reason(analysis, params, retrieval_results)

        # 步骤 8: 生成汇总提示词
        params.generation_prompt = params.to_generation_prompt()

        return params

    def _extract_motifs(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                        params: ConstraintParameters):
        """提取母题约束——优先匹配用户意图，再从最佳纹样的关联母题中提取"""
        motifs_seen = set()

        # 从已识别母题中提取（用户直接输入了母题名）
        for motif_name in analysis.recognized_entities.get("motif", []):
            if motif_name not in motifs_seen:
                motifs_seen.add(motif_name)
                if not params.primary_motif:
                    params.primary_motif = motif_name
                else:
                    params.secondary_motifs.append(motif_name)

        # 从最佳匹配的纹样通过图关系找母题
        # 先找最高分的 PaperCutPattern
        best_pattern = None
        for result in results:
            if result.entity_type == "PaperCutPattern":
                best_pattern = result
                break

        if best_pattern:
            # 通过 HAS_MOTIF 关系找母题
            rels = self.kb.get_relations(best_pattern.entity_id)
            motif_candidates = []
            for rel in rels:
                if rel.get("relation_type") == "HAS_MOTIF":
                    target_id = rel.get("target_id", "")
                    target_entity = self.kb.get_by_id(target_id)
                    if target_entity:
                        m_name = target_entity.get("name", "")
                        prominence = rel.get("properties", {}).get("prominence", 0.5)
                        is_primary = rel.get("properties", {}).get("is_primary", False)
                        motif_candidates.append((m_name, prominence, is_primary))

            # 按显著度排序
            motif_candidates.sort(key=lambda x: x[1], reverse=True)

            # 当用户没有在输入中指定具体母题时，使用纹样全名作为主要推荐
            # 这样用户能获得完整的纹样概念（如"龙凤呈祥"而非"龙纹"）
            user_specified_motif = bool(analysis.recognized_entities.get("motif", []))
            if not user_specified_motif:
                params.primary_motif = best_pattern.name
                # 图关系中的母题作为辅助信息
                for m_name, prominence, is_primary in motif_candidates:
                    if m_name not in motifs_seen:
                        motifs_seen.add(m_name)
                        params.motifs.append({
                            "name": m_name, "prominence": prominence,
                            "position": "", "is_primary": is_primary,
                        })
                        if m_name != params.primary_motif and m_name not in params.secondary_motifs:
                            params.secondary_motifs.append(m_name)
            else:
                # 用户指定了母题：图关系中 is_primary 的母题作为主推荐
                for m_name, prominence, is_primary in motif_candidates:
                    if m_name not in motifs_seen:
                        motifs_seen.add(m_name)
                        params.motifs.append({
                            "name": m_name, "prominence": prominence,
                            "position": "", "is_primary": is_primary,
                        })
                        if is_primary and not params.primary_motif:
                            params.primary_motif = m_name
                        elif m_name not in params.secondary_motifs:
                            params.secondary_motifs.append(m_name)

        # 如果仍然没有主母题，用最高分的实体名
        if not params.primary_motif and results:
            params.primary_motif = results[0].name

        # 去重辅助母题
        params.secondary_motifs = list(dict.fromkeys(params.secondary_motifs))

    def _extract_style(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                       params: ConstraintParameters):
        """提取风格约束——优先从最佳匹配纹样的关联流派取，而非任意连接"""
        # 从已识别的风格中提取（用户明确提到风格名）
        styles_from_input = analysis.recognized_entities.get("style", [])
        if styles_from_input:
            params.style = styles_from_input[0]
            # 从 KB 获取风格标签
            for result in results:
                if result.entity_type == "RegionalStyle" and result.name == params.style:
                    tags = result.properties.get("style_tags", [])
                    params.style_tags.extend(tags)
            params.style_tags = list(dict.fromkeys(params.style_tags))[:5]
            return

        # 从最佳匹配纹样的流派关系取
        pattern_style_map = {}  # style_name → {weight, tags}
        for result in results:
            if result.entity_type == "PaperCutPattern":
                rels = self.kb.get_relations(result.entity_id)
                for rel in rels:
                    if rel.get("relation_type") == "BELONGS_TO_STYLE":
                        target_id = rel.get("target_id", "")
                        target_entity = self.kb.get_by_id(target_id)
                        if target_entity:
                            sname = target_entity.get("name", "")
                            weight = rel.get("properties", {}).get("influence_degree", 0.5) * result.score
                            if sname not in pattern_style_map or weight > pattern_style_map[sname]["weight"]:
                                pattern_style_map[sname] = {
                                    "weight": weight,
                                    "tags": target_entity.get("style_tags", []),
                                }

        if pattern_style_map:
            best_style = max(pattern_style_map, key=lambda s: pattern_style_map[s]["weight"])
            params.style = best_style
            params.style_tags = pattern_style_map[best_style]["tags"][:5]

        # 如果仍未匹配到流派，从最高分的 RegionalStyle 取
        if not params.style:
            for result in results:
                if result.entity_type == "RegionalStyle":
                    params.style = result.name
                    tags = result.properties.get("style_tags", [])
                    params.style_tags.extend(tags)
                    params.style_tags = list(dict.fromkeys(params.style_tags))[:5]
                    break

    def _extract_technique(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                           params: ConstraintParameters):
        """提取技法约束"""
        # 从已识别的技法中提取
        techs_from_input = analysis.recognized_entities.get("technique", [])
        if techs_from_input:
            params.technique = techs_from_input[0]

        # 从检索结果中找技法
        for result in results:
            if result.entity_type == "Technique":
                if not params.technique:
                    params.technique = result.name
                else:
                    params.alternative_techniques.append(result.name)

            # 纹样的用法关系
            if result.entity_type == "PaperCutPattern":
                rels = self.kb.get_relations(result.entity_id)
                for rel in rels:
                    if rel.get("relation_type") == "USES_TECHNIQUE":
                        target_id = rel.get("target_id", "")
                        target_entity = self.kb.get_by_id(target_id)
                        if target_entity:
                            t_name = target_entity.get("name", "")
                            is_primary = rel.get("properties", {}).get("is_primary", False)
                            if is_primary and not params.technique:
                                params.technique = t_name
                            else:
                                params.alternative_techniques.append(t_name)

        params.alternative_techniques = list(dict.fromkeys(params.alternative_techniques))[:3]

    def _extract_colors(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                        params: ConstraintParameters):
        """提取色彩约束"""
        # 策略：从最佳匹配的纹样（最高分）取配色，而非聚合所有
        best_pattern = None
        for result in results:
            if result.entity_type == "PaperCutPattern":
                best_pattern = result
                break

        if best_pattern:
            colors = best_pattern.properties.get("recommended_colors", [])
            if isinstance(colors, list):
                params.recommended_colors = colors
            params.color_count = best_pattern.properties.get("color_count")
        else:
            # 从最匹配的文化象征推断
            for result in results:
                if result.entity_type == "CulturalSymbol":
                    symbol_name = result.properties.get("name", "")
                    if symbol_name in ("福", "寿"):
                        params.recommended_colors = ["大红", "金色"]
                        params.color_count = 2
                        break

        # 如果没从检索结果中拿到，从意图推断
        if not params.recommended_colors:
            if "婚庆" in analysis.intents:
                params.recommended_colors = ["大红", "金色"]
                params.color_count = 2
            elif "春节" in analysis.intents:
                params.recommended_colors = ["大红"]
                params.color_count = 1
            elif "祝寿" in analysis.intents:
                params.recommended_colors = ["大红", "金色"]
                params.color_count = 2
            else:
                params.recommended_colors = ["大红"]
                params.color_count = 1

        # 色彩方案推断
        if params.color_count and params.color_count >= 3:
            params.color_scheme = "套色"
        elif params.color_count == 2:
            params.color_scheme = "双色"
        else:
            params.color_scheme = "单色"

    def _extract_composition(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                             params: ConstraintParameters):
        """提取构图约束"""
        # 从已识别的对称类型
        sym_from_input = analysis.recognized_entities.get("symmetry", [])
        if sym_from_input:
            params.symmetry_type = sym_from_input[0]

        # 从纹样属性中提取
        for result in results:
            if result.entity_type == "PaperCutPattern":
                sym = result.properties.get("symmetry_type")
                if sym and not params.symmetry_type:
                    params.symmetry_type = sym
                comp = result.properties.get("complexity")
                if comp and not params.complexity:
                    params.complexity = comp

        # 从意图推断复杂度
        if not params.complexity:
            if "教学" in analysis.intents:
                params.complexity = "简单"
            elif "礼品" in analysis.intents:
                params.complexity = "中等"
            else:
                # 取平均分
                complexities = []
                for result in results:
                    c = result.properties.get("complexity")
                    if c:
                        complexities.append(c)
                if complexities:
                    params.complexity = Counter(complexities).most_common(1)[0][0]

    def _extract_cultural(self, analysis: QueryAnalysis, results: List[RetrievalResult],
                          params: ConstraintParameters):
        """提取文化语义约束"""
        # 从意图推导应用场景
        for intent in analysis.intents:
            if intent in ("婚庆", "祝寿", "春节", "乔迁", "镇宅", "祈福", "生肖"):
                params.applicable_scenes.append(intent)

        # ---- 寓意选取策略：优先选与文化象征最匹配的 ----
        # 检查已识别的母题关键词，找到对应的文化象征
        motif_names = analysis.recognized_entities.get("motif", [])
        scene_names = analysis.recognized_entities.get("scene", [])

        # 从文化象征实体中找最佳寓意
        best_meaning = None
        best_match_score = 0

        for result in results:
            if result.entity_type == "CulturalSymbol":
                meaning = result.properties.get("meaning", "")
                symbol_name = result.properties.get("name", "")

                # 检查是否与意图/母题匹配
                score = 0
                for motif in motif_names:
                    if motif in symbol_name or motif in meaning:
                        score += 2
                for scene in scene_names:
                    if scene in symbol_name or scene in meaning:
                        score += 1.5

                if score > best_match_score:
                    best_match_score = score
                    best_meaning = meaning

        if best_meaning:
            params.meaning = best_meaning

        # 如果没有找到匹配的文化象征，从最高分的纹样取寓意
        if not params.meaning:
            for result in results:
                if result.entity_type == "PaperCutPattern" and result.score > 0.7:
                    params.meaning = result.properties.get("meaning", "")
                    if params.meaning:
                        break

        # 文化背景
        if not params.cultural_context:
            for result in results:
                if result.entity_type == "PaperCutPattern":
                    ctx = result.properties.get("cultural_context", "")
                    if ctx:
                        params.cultural_context = ctx
                        break

        # 场景去重
        params.applicable_scenes = list(dict.fromkeys(params.applicable_scenes))[:5]

    def _rank_and_reason(self, analysis: QueryAnalysis, params: ConstraintParameters,
                         results: List[RetrievalResult]):
        """排序结果并生成推荐理由"""
        # 找到最匹配的纹样作为参考图
        for result in results:
            if result.entity_type == "PaperCutPattern":
                img_url = result.properties.get("reference_image_url", "")
                if img_url:
                    params.reference_image_urls.append(img_url)

        # 生成推荐理由
        reasons = []
        if params.primary_motif:
            reasons.append(f"母题「{params.primary_motif}」")
        if params.meaning:
            reasons.append(f"寓意「{params.meaning[:20]}」")
        if params.style:
            reasons.append(f"{params.style}风格")
        if params.technique:
            reasons.append(f"{params.technique}技法")
        if analysis.intents and "通用" not in analysis.intents:
            reasons.append(f"适用于{analysis.intents[0]}场景")

        if reasons:
            params.ranking_reason = "根据您的需求，推荐" + "、".join(reasons) + "的剪纸方案"
        else:
            params.ranking_reason = "为您推荐以下剪纸方案"


# ========== 约束生成引擎（主入口）==========

class PaperCutConstraintGenerator:
    """
    Graph RAG 约束生成引擎（主入口）
    接收 → 分析 → 检索 → 推理 → 输出结构化参数
    """

    def __init__(self, knowledge_base: Optional[PaperCutKnowledgeBase] = None,
                 logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_logger()

        # 知识库（可外部注入，也可自动加载）
        self.kb = knowledge_base or PaperCutKnowledgeBase()
        self.analyzer = PaperCutQueryAnalyzer()
        self.reasoner = PaperCutConstraintReasoner(self.kb)

        self._loaded = False

    def _setup_logger(self):
        logger = logging.getLogger("PaperCutCG")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            logger.addHandler(handler)
        return logger

    def load_knowledge(self, source: str = "auto", entities_path: str = "",
                       relations_path: str = "", output_dir: str = "./pipeline_output"):
        """
        加载知识库

        Args:
            source: "auto"=自动查找, "seed"=从种子数据加载,
                    "files"=从指定文件加载, "pipeline"=从管道输出加载
            entities_path: 实体 JSON 路径（source="files"时）
            relations_path: 关系 JSON 路径（source="files"时）
            output_dir: 管道输出目录（source="pipeline"时）
        """
        if source == "seed":
            self.kb.load_from_seed_data()
            self._loaded = self.kb.is_loaded()
        elif source == "files":
            self.kb.load(entities_path, relations_path)
            self._loaded = self.kb.is_loaded()
        elif source == "pipeline":
            self.kb.load_from_pipeline_output(output_dir)
            self._loaded = self.kb.is_loaded()
        else:  # auto
            # 尝试管道输出 → 种子数据
            self.kb.load_from_pipeline_output(output_dir)
            if not self.kb.is_loaded():
                self.kb.load_from_seed_data()
            self._loaded = self.kb.is_loaded()

        if self._loaded:
            self.logger.info(f"知识库就绪: {json.dumps(self.kb.get_stats(), ensure_ascii=False)}")
        else:
            self.logger.warning("知识库未加载，请先运行提取管道或指定数据文件")

        return self._loaded

    def generate(self, keyword: str, top_k: int = 15,
                 return_full: bool = False) -> Dict:
        """
        核心方法：输入关键词，输出约束参数

        Args:
            keyword: 用户输入关键词（如"龙年吉祥"、"婚庆剪纸"）
            top_k: 检索 TOP-K
            return_full: 是否返回中间分析结果

        Returns:
            约束参数字典（可直接作为 JSON 传给图像生成模型）
        """
        if not self._loaded:
            self.load_knowledge()

        self.logger.info(f"输入: '{keyword}'")

        # 1. 查询分析
        analysis = self.analyzer.analyze(keyword)
        self.logger.info(f"  意图: {analysis.intents}")
        self.logger.info(f"  实体: {analysis.recognized_entities}")
        self.logger.info(f"  扩展: {analysis.expanded_terms}")

        # 2. 多路检索
        retriever = PaperCutRetriever(self.kb)
        results = retriever.retrieve(analysis, top_k=top_k)
        self.logger.info(f"  检索: {len(results)} 条结果")

        # 3. 推理生成约束
        params = self.reasoner.generate(analysis, results)
        self.logger.info(f"  主母题: {params.primary_motif}")
        self.logger.info(f"  风格: {params.style}")
        self.logger.info(f"  技法: {params.technique}")
        self.logger.info(f"  配色: {params.recommended_colors}")

        # 4. 输出
        output = params.to_dict()
        if return_full:
            output["_analysis"] = analysis.to_dict()
            output["_retrieval_results"] = [
                {"name": r.name, "type": r.entity_type,
                 "score": r.score, "source": r.source}
                for r in results[:5]
            ]

        self.logger.info(f"  提示词: {params.generation_prompt}")

        return output

    def batch_generate(self, keywords: List[str], top_k: int = 15) -> List[Dict]:
        """批量生成"""
        return [self.generate(kw, top_k=top_k) for kw in keywords]


# ========== CLI 交互式演示 ==========

def main():
    """交互式 CLI 演示"""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="非遗剪纸 Graph RAG 约束生成引擎")
    parser.add_argument("keyword", nargs="?", help="输入关键词")
    parser.add_argument("--top-k", type=int, default=15, help="检索 TOP-K")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--json", "-j", action="store_true", help="JSON 格式输出")
    parser.add_argument("--load", "-l", choices=["auto", "seed", "pipeline"],
                        default="auto", help="知识库加载方式")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # 初始化引擎
    engine = PaperCutConstraintGenerator()
    engine.load_knowledge(source=args.load)

    if args.keyword:
        result = engine.generate(args.keyword, top_k=args.top_k,
                                 return_full=args.verbose)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("\n" + "=" * 60)
            print("约束生成结果")
            print("=" * 60)
            print(f"  提示词: {result.get('generation_prompt', '')}")
            print()
            if result.get("ranking_reason"):
                print(f"  推荐理由: {result['ranking_reason']}")
                print()
            print(f"  主母题: {result.get('primary_motif', '-')}")
            print(f"  辅纹: {', '.join(result.get('secondary_motifs', []) or ['-'])}")
            print(f"  流派: {result.get('style', '-')}")
            print(f"  技法: {result.get('technique', '-')}")
            print(f"  配色: {', '.join(result.get('recommended_colors', []) or ['-'])}")
            print(f"  对称: {result.get('symmetry_type', '-')}")
            print(f"  复杂度: {result.get('complexity', '-')}")
            print(f"  寓意: {(result.get('meaning') or '-')[:50]}")
            print(f"  场景: {', '.join(result.get('applicable_scenes', []) or ['-'])}")

    else:
        # 交互模式
        print("=" * 60)
        print("非遗剪纸 Graph RAG 约束生成引擎 (交互模式)")
        print("输入关键词查看约束生成结果，输入 'quit' 退出")
        print("示例: 龙年吉祥, 婚庆剪纸, 春节福字, 送给长辈, 北方风格")
        print("=" * 60)

        while True:
            try:
                keyword = input("\n关键词: ").strip()
                if not keyword or keyword.lower() in ("quit", "exit", "q"):
                    break

                result = engine.generate(keyword, top_k=args.top_k)

                print("\n" + "-" * 40)
                print(f"提示词: {result.get('generation_prompt', '')}")
                print()
                if result.get("ranking_reason"):
                    print(f"推荐理由: {result['ranking_reason']}")
                print()
                items = [
                    ("母题", result.get("primary_motif", "-")),
                    ("辅纹", ', '.join(result.get("secondary_motifs", []) or ['-'])),
                    ("流派", result.get("style", "-")),
                    ("技法", result.get("technique", "-")),
                    ("配色", ', '.join(result.get("recommended_colors", []) or ['-'])),
                    ("对称", result.get("symmetry_type", "-")),
                    ("复杂度", result.get("complexity", "-")),
                    ("寓意", (result.get("meaning") or "-")[:40]),
                ]
                for k, v in items:
                    print(f"  {k}: {v}")
                print("-" * 40)

            except KeyboardInterrupt:
                print()
                break

    return 0


if __name__ == "__main__":
    exit(main())
