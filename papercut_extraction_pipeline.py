"""
非遗剪纸知识图谱 — 数据抽取管道（支持断点续传）
负责从多源数据（种子数据、文件、网页、API）中抽取实体和关系
支持按阶段断点续传，中断后从断点恢复
"""

import json
import os
import logging
import hashlib
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from pathlib import Path


# ========== 阶段定义 ==========

class PipelineStage(str, Enum):
    """管道阶段枚举（断点粒度）"""
    LOAD_SEED_DATA = "load_seed_data"
    COLLECT_DATA = "collect_data"
    EXTRACT_ENTITIES = "extract_entities"
    EXTRACT_RELATIONS = "extract_relations"
    FUSE_ENTITIES = "fuse_entities"
    OUTPUT_RESULTS = "output_results"


STAGE_ORDER = [
    PipelineStage.LOAD_SEED_DATA,
    PipelineStage.COLLECT_DATA,
    PipelineStage.EXTRACT_ENTITIES,
    PipelineStage.EXTRACT_RELATIONS,
    PipelineStage.FUSE_ENTITIES,
    PipelineStage.OUTPUT_RESULTS,
]

# ========== 数据结构 ==========

@dataclass
class CheckpointState:
    """断点状态"""
    run_id: str
    config_hash: str
    completed_stages: List[str] = field(default_factory=list)
    current_stage: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    statistics: Dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class ExtractedEntity:
    """抽取的实体"""
    entity_type: str
    name: str
    properties: Dict
    confidence: float = 1.0
    source: str = "pipeline"
    source_id: str = ""

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


@dataclass
class ExtractedRelation:
    """抽取的关系"""
    relation_type: str
    source_id: str
    target_id: str
    source_type: str
    target_type: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0
    source: str = "pipeline"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


# ========== 管道引擎 ==========

class ExtractionPipeline:
    """
    剪纸知识抽取管道

    使用方法：
        pipeline = ExtractionPipeline(config_path="papercut_data_sources.json")
        result = pipeline.run()          # 完整运行
        result = pipeline.run(resume=True)  # 从上次断点恢复

    特性：
        - 断点续传：每个阶段完成后写状态文件，重启跳过已完成阶段
        - 可插拔数据源：通过 JSON 配置控制启用/关闭
        - 多源汇聚：种子数据 + 文件导入 + 网页/API（预留）
    """

    def __init__(
        self,
        config_path: str = "papercut_data_sources.json",
        state_dir: str = "./pipeline_state",
        logger: Optional[logging.Logger] = None,
    ):
        self.config_path = Path(config_path)
        self.state_dir = Path(state_dir)
        self.logger = logger or self._setup_logger()

        # 运行时状态
        self.config: Dict = {}
        self.state: Optional[CheckpointState] = None
        self.collected_data: Dict[str, Any] = {}
        self.entities: List[ExtractedEntity] = []
        self.relations: List[ExtractedRelation] = []

        # 缓存（方便跨阶段查找）
        self._entity_by_id: Dict[str, ExtractedEntity] = {}
        self._entity_by_type_name: Dict[str, Dict[str, ExtractedEntity]] = {}

        # 确保状态目录存在
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("PaperCutPipeline")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s",
                datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
        return logger

    # ========== 配置加载 ==========

    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not self.config_path.exists():
            self.logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            return self._default_config()

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.logger.info(f"已加载配置文件: {self.config_path}")
        return config

    def _default_config(self) -> Dict:
        return {
            "pipeline_name": "剪纸知识图谱抽取管道",
            "state_dir": "./pipeline_state",
            "sources": [
                {"id": "seed_builtin", "type": "seed", "enabled": True,
                 "description": "内置种子数据"}
            ],
            "extraction": {
                "confidence_threshold": 0.6,
                "enable_relation_extraction": True,
                "enable_entity_fusion": False,
            },
            "checkpoint": {
                "stages": [s.value for s in STAGE_ORDER],
            },
        }

    def _config_hash(self) -> str:
        """生成配置的哈希值，用于检测配置变更"""
        raw = json.dumps(self.config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    # ========== 断点管理 ==========

    def _state_path(self, run_id: str) -> Path:
        return self.state_dir / f"checkpoint_{run_id}.json"

    def _next_run_id(self) -> str:
        """生成新的运行ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"run_{timestamp}"

    def _load_state(self, run_id: Optional[str] = None) -> Optional[CheckpointState]:
        """从磁盘加载断点状态"""
        if run_id:
            state_file = self._state_path(run_id)
            if state_file.exists():
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return CheckpointState.from_dict(data)
        return None

    def _save_state(self):
        """保存当前断点状态到磁盘"""
        if not self.state:
            return
        self.state.updated_at = datetime.now().isoformat()
        state_file = self._state_path(self.state.run_id)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.state.to_dict(), f, ensure_ascii=False, indent=2)
        self.logger.debug(f"断点已保存: {state_file}")

    def _stage_completed(self, stage: PipelineStage) -> bool:
        """检查阶段是否已完成"""
        if not self.state:
            return False
        return stage.value in self.state.completed_stages

    def _mark_stage_start(self, stage: PipelineStage):
        """标记阶段开始"""
        if not self.state:
            return
        self.state.current_stage = stage.value
        self._save_state()
        self.logger.info(f"[阶段] {stage.value}")

    def _mark_stage_done(self, stage: PipelineStage, stats: Dict = None):
        """标记阶段完成"""
        if not self.state:
            return
        if stage.value not in self.state.completed_stages:
            self.state.completed_stages.append(stage.value)
        if stats:
            self.state.statistics[stage.value] = stats
        self.state.current_stage = None
        self._save_state()
        self.logger.info(f"[完成] {stage.value}  |  统计: {stats}")

    def _mark_error(self, stage: PipelineStage, error: str):
        """标记阶段出错"""
        if not self.state:
            return
        self.state.error = f"[{stage.value}] {error}"
        self.state.current_stage = None
        self._save_state()
        self.logger.error(f"[错误] {stage.value}: {error}")

    # ========== 阶段 1: 加载种子数据 ==========

    def _stage_load_seed_data(self) -> Dict:
        """
        从 papercut_seed_data.py 加载内置种子数据
        这是断点友好的 —— 种子数据作为代码内嵌不变，
        不会因外部网络问题中断
        """
        try:
            from papercut_seed_data import get_all_seed_data, get_seed_summary
        except ImportError as e:
            raise RuntimeError(f"无法加载种子数据模块: {e}")

        seed_data = get_all_seed_data()
        summary = get_seed_summary()

        # 将种子数据存入 collected_data
        self.collected_data["seed_builtin"] = seed_data

        self.logger.info(f"内置种子数据已加载:")
        self.logger.info(f"  纹样 {summary['patterns']} | 母题 {summary['motifs']} | "
                         f"技法 {summary['techniques']} | 流派 {summary['regional_styles']}")
        self.logger.info(f"  象征 {summary['cultural_symbols']} | 传承人 {summary['inheritors']} | "
                         f"材料 {summary['materials']}")
        self.logger.info(f"  关系映射共 {summary['pattern_motif_links'] + summary['pattern_technique_links'] + summary['pattern_style_links'] + summary['pattern_symbol_links'] + summary['pattern_material_links']} 条")

        return summary

    # ========== 阶段 2: 数据采集（含断点续传） ==========

    def _stage_collect_data(self) -> Dict:
        """
        从所有启用的数据源采集数据
        可扩展：后续对接 web/API 数据源时在这里添加
        """
        stats = {"sources_loaded": 0, "sources_failed": 0, "total_items": 0}

        sources = self.config.get("sources", [])
        for source in sources:
            if not source.get("enabled", False):
                self.logger.info(f"  [跳过] 数据源已禁用: {source.get('id')}")
                continue

            source_id = source["id"]
            source_type = source["type"]

            # 种子数据已在阶段 1 加载
            if source_type == "seed":
                stats["sources_loaded"] += 1
                continue

            # ---- 以下为预留数据源接口 ----

            elif source_type == "file":
                path = source.get("path", "")
                if not path or not os.path.exists(path):
                    self.logger.warning(f"  [跳过] 文件不存在: {path}")
                    stats["sources_failed"] += 1
                    continue

                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.collected_data[source_id] = data
                    items = len(data) if isinstance(data, list) else 1
                    stats["sources_loaded"] += 1
                    stats["total_items"] += items
                    self.logger.info(f"  [文件] {path} → {items} 条数据")
                except Exception as e:
                    self.logger.error(f"  [失败] 读取文件 {path}: {e}")
                    stats["sources_failed"] += 1

            elif source_type == "web":
                url = source.get("url", "")
                if not url:
                    stats["sources_failed"] += 1
                    continue
                self.logger.info(f"  [预留] 网页数据采集待实现: {url}")
                # 实际的爬取逻辑后续添加
                stats["sources_failed"] += 1

            elif source_type == "api":
                url = source.get("url", "")
                if not url:
                    stats["sources_failed"] += 1
                    continue
                self.logger.info(f"  [预留] API数据采集待实现: {url}")
                stats["sources_failed"] += 1

            else:
                self.logger.warning(f"  [未知] 数据源类型: {source_type}")
                stats["sources_failed"] += 1

        return stats

    # ========== 阶段 3: 实体抽取 ==========

    def _stage_extract_entities(self) -> Dict:
        """
        从 collected_data 中抽取实体
        种子数据已经是结构化数据，直接转换即可
        """
        stats = {"entities_created": 0}

        seed_data = self.collected_data.get("seed_builtin", {})
        if not seed_data:
            self.logger.warning("无种子数据可供抽取")
            return stats

        # ---- 纹样实体 ----
        for item in seed_data.get("patterns", []):
            entity = ExtractedEntity(
                entity_type="PaperCutPattern",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 母题实体 ----
        for item in seed_data.get("motifs", []):
            entity = ExtractedEntity(
                entity_type="PaperCutMotif",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 技法实体 ----
        for item in seed_data.get("techniques", []):
            entity = ExtractedEntity(
                entity_type="Technique",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 流派实体 ----
        for item in seed_data.get("regional_styles", []):
            entity = ExtractedEntity(
                entity_type="RegionalStyle",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 文化象征实体 ----
        for item in seed_data.get("cultural_symbols", []):
            entity = ExtractedEntity(
                entity_type="CulturalSymbol",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 传承人实体 ----
        for item in seed_data.get("inheritors", []):
            entity = ExtractedEntity(
                entity_type="PaperCutInheritor",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # ---- 材料实体 ----
        for item in seed_data.get("materials", []):
            entity = ExtractedEntity(
                entity_type="Material",
                name=item.get("name", ""),
                properties=item,
                confidence=1.0,
                source="seed_data",
                source_id=item.get("id", ""),
            )
            self.entities.append(entity)
            self._cache_entity(entity)
            stats["entities_created"] += 1

        # 按类型统计
        type_counts = {}
        for e in self.entities:
            type_counts[e.entity_type] = type_counts.get(e.entity_type, 0) + 1
        stats["by_type"] = type_counts

        self.logger.info(f"实体抽取完成: {' | '.join(f'{k}:{v}' for k, v in type_counts.items())}")
        return stats

    def _cache_entity(self, entity: ExtractedEntity):
        """缓存实体以便快速查找"""
        self._entity_by_id[entity.source_id] = entity
        if entity.entity_type not in self._entity_by_type_name:
            self._entity_by_type_name[entity.entity_type] = {}
        self._entity_by_type_name[entity.entity_type][entity.name] = entity

    def _find_entity_by_id(self, entity_id: str) -> Optional[ExtractedEntity]:
        return self._entity_by_id.get(entity_id)

    # ========== 阶段 4: 关系抽取 ==========

    def _stage_extract_relations(self) -> Dict:
        """
        从种子数据的映射表中抽取关系
        """
        stats = {"relations_created": 0}

        seed_data = self.collected_data.get("seed_builtin", {})
        if not seed_data:
            return stats

        # ---- 纹样→母题关系 ----
        motif_map = seed_data.get("pattern_motif_map", {})
        for pattern_id, links in motif_map.items():
            pattern = self._find_entity_by_id(pattern_id)
            if not pattern:
                continue
            for link in links:
                target = self._find_entity_by_id(link["motif_id"])
                if not target:
                    continue
                self.relations.append(ExtractedRelation(
                    relation_type="HAS_MOTIF",
                    source_id=pattern_id,
                    target_id=link["motif_id"],
                    source_type="PaperCutPattern",
                    target_type="PaperCutMotif",
                    properties={
                        "prominence": link.get("prominence", 0.5),
                        "position": link.get("position", ""),
                        "is_primary": link.get("is_primary", False),
                    },
                    source="seed_data",
                ))
                stats["relations_created"] += 1

        # ---- 纹样→技法关系 ----
        tech_map = seed_data.get("pattern_technique_map", {})
        for pattern_id, links in tech_map.items():
            pattern = self._find_entity_by_id(pattern_id)
            if not pattern:
                continue
            for link in links:
                target = self._find_entity_by_id(link["technique_id"])
                if not target:
                    continue
                self.relations.append(ExtractedRelation(
                    relation_type="USES_TECHNIQUE",
                    source_id=pattern_id,
                    target_id=link["technique_id"],
                    source_type="PaperCutPattern",
                    target_type="Technique",
                    properties={
                        "is_primary": link.get("is_primary", False),
                        "order": link.get("order", 1),
                    },
                    source="seed_data",
                ))
                stats["relations_created"] += 1

        # ---- 纹样→流派关系 ----
        style_map = seed_data.get("pattern_style_map", {})
        for pattern_id, links in style_map.items():
            pattern = self._find_entity_by_id(pattern_id)
            if not pattern:
                continue
            for link in links:
                target = self._find_entity_by_id(link["style_id"])
                if not target:
                    continue
                self.relations.append(ExtractedRelation(
                    relation_type="BELONGS_TO_STYLE",
                    source_id=pattern_id,
                    target_id=link["style_id"],
                    source_type="PaperCutPattern",
                    target_type="RegionalStyle",
                    properties={
                        "influence_degree": link.get("influence_degree", 0.5),
                        "is_representative": link.get("is_representative", False),
                    },
                    source="seed_data",
                ))
                stats["relations_created"] += 1

        # ---- 纹样→象征关系 ----
        symbol_map = seed_data.get("pattern_symbol_map", {})
        for pattern_id, links in symbol_map.items():
            pattern = self._find_entity_by_id(pattern_id)
            if not pattern:
                continue
            for link in links:
                target = self._find_entity_by_id(link["target_id"])
                if not target:
                    continue
                self.relations.append(ExtractedRelation(
                    relation_type="HAS_SYMBOLISM",
                    source_id=pattern_id,
                    target_id=link["target_id"],
                    source_type="PaperCutPattern",
                    target_type="CulturalSymbol",
                    properties={
                        "significance": link.get("significance", 0.5),
                        "cultural_note": link.get("cultural_note", ""),
                    },
                    source="seed_data",
                ))
                stats["relations_created"] += 1

        # ---- 纹样→材料关系 ----
        material_map = seed_data.get("pattern_material_map", {})
        for pattern_id, links in material_map.items():
            pattern = self._find_entity_by_id(pattern_id)
            if not pattern:
                continue
            for link in links:
                target = self._find_entity_by_id(link["material_id"])
                if not target:
                    continue
                self.relations.append(ExtractedRelation(
                    relation_type="REQUIRES_MATERIAL",
                    source_id=pattern_id,
                    target_id=link["material_id"],
                    source_type="PaperCutPattern",
                    target_type="Material",
                    properties={
                        "is_default": link.get("is_default", True),
                        "specification_detail": link.get("specification_detail", ""),
                    },
                    source="seed_data",
                ))
                stats["relations_created"] += 1

        # ---- 母题间关联关系 ----
        motif_relations = seed_data.get("motif_relations", [])
        for rel in motif_relations:
            source = self._find_entity_by_id(rel["source_id"])
            target = self._find_entity_by_id(rel["target_id"])
            if not source or not target:
                continue
            self.relations.append(ExtractedRelation(
                relation_type="RELATED_MOTIF",
                source_id=rel["source_id"],
                target_id=rel["target_id"],
                source_type="PaperCutMotif",
                target_type="PaperCutMotif",
                properties={
                    "relation_type": rel.get("relation_type", ""),
                    "description": rel.get("description", ""),
                    "strength": rel.get("strength", 0.5),
                },
                source="seed_data",
            ))
            stats["relations_created"] += 1

        # ---- 传承人→技法关系（从传承人的 specialties 字段） ----
        for item in seed_data.get("inheritors", []):
            inheritor_id = item.get("id", "")
            specialties = item.get("specialties", [])
            # 查找对应的技法实体
            for speciality in specialties:
                for tech_entity in self.entities:
                    if tech_entity.entity_type != "Technique":
                        continue
                    if speciality in tech_entity.name or speciality in (tech_entity.properties.get("description", "")):
                        self.relations.append(ExtractedRelation(
                            relation_type="SPECIALIZES_IN",
                            source_id=inheritor_id,
                            target_id=tech_entity.source_id,
                            source_type="PaperCutInheritor",
                            target_type="Technique",
                            properties={
                                "mastery_level": "大师级",
                                "years": item.get("years_of_experience", 0),
                            },
                            source="seed_data",
                        ))
                        stats["relations_created"] += 1
                        break

        self.logger.info(f"关系抽取完成: {stats['relations_created']} 条")
        return stats

    # ========== 阶段 5: 实体融合 ==========

    def _stage_fuse_entities(self) -> Dict:
        """
        实体融合（去重）
        当前种子数据无重复，此阶段为预留
        """
        stats = {
            "before_fusion": len(self.entities),
            "after_fusion": len(self.entities),
            "duplicates_removed": 0,
        }
        self.logger.info(f"实体融合完成: {stats['after_fusion']} 个 (去重 {stats['duplicates_removed']} 个)")
        return stats

    # ========== 阶段 6: 输出结果 ==========

    def _stage_output_results(self) -> Dict:
        """
        输出结果：生成 JSON 文件和统计数据
        """
        output_dir = Path("./pipeline_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        run_id = self.state.run_id if self.state else "unknown"

        # ---- 输出实体 JSON ----
        entities_path = output_dir / f"entities_{run_id}.json"
        with open(entities_path, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self.entities],
                f, ensure_ascii=False, indent=2,
            )

        # ---- 输出关系 JSON ----
        relations_path = output_dir / f"relations_{run_id}.json"
        with open(relations_path, "w", encoding="utf-8") as f:
            json.dump(
                [r.to_dict() for r in self.relations],
                f, ensure_ascii=False, indent=2,
            )

        # ---- 输出汇总报告 ----
        entity_type_counts = {}
        for e in self.entities:
            entity_type_counts[e.entity_type] = entity_type_counts.get(e.entity_type, 0) + 1

        relation_type_counts = {}
        for r in self.relations:
            relation_type_counts[r.relation_type] = relation_type_counts.get(r.relation_type, 0) + 1

        report = {
            "run_id": run_id,
            "completed_at": datetime.now().isoformat(),
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entities_by_type": entity_type_counts,
            "relations_by_type": relation_type_counts,
            "output_files": {
                "entities": str(entities_path),
                "relations": str(relations_path),
            },
        }

        report_path = output_dir / f"report_{run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"结果已输出:")
        self.logger.info(f"  实体: {entities_path}")
        self.logger.info(f"  关系: {relations_path}")
        self.logger.info(f"  报告: {report_path}")

        return report

    # ========== 主流程 ==========

    def run(self, resume: bool = False) -> Dict:
        """
        运行完整管道（支持断点续传）

        Args:
            resume: 是否从上次断点恢复。True=跳过已完成阶段；False=从头运行

        Returns:
            汇总报告字典
        """
        # 1. 加载配置
        self.config = self._load_config()
        config_hash = self._config_hash()

        # 2. 初始化或恢复断点
        if resume:
            # 查找最近的断点文件
            state_files = sorted(self.state_dir.glob("checkpoint_run_*.json"))
            if state_files:
                with open(state_files[-1], "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.state = CheckpointState.from_dict(data)
                self.logger.info(f"恢复断点: {self.state.run_id}")
                self.logger.info(f"  已完成阶段: {self.state.completed_stages}")

                if self.state.error:
                    self.logger.warning(f"  上次运行错误: {self.state.error}")
        else:
            run_id = self._next_run_id()
            self.state = CheckpointState(
                run_id=run_id,
                config_hash=config_hash,
                started_at=datetime.now().isoformat(),
            )
            self._save_state()
            self.logger.info(f"新运行启动: {run_id}")

        stages = [
            (PipelineStage.LOAD_SEED_DATA, self._stage_load_seed_data),
            (PipelineStage.COLLECT_DATA, self._stage_collect_data),
            (PipelineStage.EXTRACT_ENTITIES, self._stage_extract_entities),
            (PipelineStage.EXTRACT_RELATIONS, self._stage_extract_relations),
            (PipelineStage.FUSE_ENTITIES, self._stage_fuse_entities),
            (PipelineStage.OUTPUT_RESULTS, self._stage_output_results),
        ]

        for stage, func in stages:
            # 断点检查
            if self._stage_completed(stage):
                self.logger.info(f"[跳过] 阶段 {stage.value} 已完成")
                continue

            try:
                self._mark_stage_start(stage)
                stats = func()
                self._mark_stage_done(stage, stats)
            except Exception as e:
                self._mark_error(stage, str(e))
                raise RuntimeError(f"管道在阶段 [{stage.value}] 失败: {e}")

        # 最终报告（从输出文件读取已持久化的结果）
        report_path = Path("./pipeline_output") / f"report_{self.state.run_id}.json"
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                final_stats = json.load(f)
            self.logger.info(f"从输出文件读取已完成的结果: {report_path}")
        else:
            final_stats = {
                "run_id": self.state.run_id if self.state else "unknown",
                "total_entities": len(self.entities),
                "total_relations": len(self.relations),
                "status": "completed",
            }

            entity_types = {}
            for e in self.entities:
                entity_types[e.entity_type] = entity_types.get(e.entity_type, 0) + 1
            final_stats["entities_by_type"] = entity_types

            relation_types = {}
            for r in self.relations:
                relation_types[r.relation_type] = relation_types.get(r.relation_type, 0) + 1
            final_stats["relations_by_type"] = relation_types

        self.logger.info("=" * 50)
        self.logger.info("管道运行完成")
        self.logger.info(f"  实体总数: {final_stats.get('total_entities', 0)}")
        for t, c in final_stats.get("entities_by_type", {}).items():
            self.logger.info(f"    {t}: {c}")
        self.logger.info(f"  关系总数: {final_stats.get('total_relations', 0)}")
        for t, c in final_stats.get("relations_by_type", {}).items():
            self.logger.info(f"    {t}: {c}")
        self.logger.info("=" * 50)

        return final_stats


# ========== CLI 入口 ==========

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="非遗剪纸知识图谱抽取管道",
    )
    parser.add_argument(
        "--config", "-c",
        default="papercut_data_sources.json",
        help="配置文件路径 (默认: papercut_data_sources.json)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="从上次断点恢复运行",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # 日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # 运行管道
    pipeline = ExtractionPipeline(config_path=args.config)
    try:
        result = pipeline.run(resume=args.resume)
        print(f"\n运行成功！共抽取 {result['total_entities']} 个实体, "
              f"{result['total_relations']} 条关系")
    except RuntimeError as e:
        print(f"\n运行失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\n用户中断。断点已保存，下次使用 --resume 可恢复。")
        return 0

    return 0


if __name__ == "__main__":
    exit(main())
