import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class EdgeDirection(Enum):
    UP = "上"
    DOWN = "下"
    LEFT = "左"
    RIGHT = "右"
    UPPER_LEFT = "左上"
    UPPER_RIGHT = "右上"
    LOWER_LEFT = "左下"
    LOWER_RIGHT = "右下"


class DefectType(Enum):
    CRACK = "裂纹"
    MISSING = "缺损"
    WEAR = "磨损"
    EROSION = "侵蚀"
    INTACT = "完好"


class EdgeType(Enum):
    STRAIGHT = "直线"
    IRREGULAR = "不规则"
    CURVED = "弧形"
    BROKEN = "断裂"
    SERRATED = "锯齿状"


class ReviewStatus(Enum):
    PENDING = "待审核"
    APPROVED = "已通过"
    REJECTED = "已拒绝"
    FLAGGED = "存疑"


class OperationType(Enum):
    FRAGMENT_ADD = "添加残片"
    FRAGMENT_DELETE = "删除残片"
    FRAGMENT_UPDATE = "更新残片"
    EDGE_ADD = "添加边缘"
    EDGE_DELETE = "删除边缘"
    SCHEME_CREATE = "创建方案"
    SCHEME_DELETE = "删除方案"
    SCHEME_UPDATE = "更新方案"
    SCHEME_LOCK = "锁定方案"
    SCHEME_UNLOCK = "解锁方案"
    SCHEME_VERSION_CREATE = "创建方案版本"
    SCHEME_VERSION_RESTORE = "恢复方案版本"
    ANALYSIS_RUN = "执行候选分析"
    REVIEW_APPROVE = "审核通过"
    REVIEW_REJECT = "审核拒绝"
    REVIEW_FLAG = "审核存疑"
    DATA_IMPORT = "导入数据"
    DATA_EXPORT = "导出数据"
    CONFLICT_DETECT = "冲突检测"
    TAG_ADD = "添加标签"
    TAG_DELETE = "删除标签"
    TAG_UPDATE = "更新标签"
    TAG_RELATION_ADD = "添加标签关系"
    TAG_RELATION_DELETE = "删除标签关系"
    ANNOTATION_ADD = "添加标注"
    ANNOTATION_UPDATE = "更新标注"
    ANNOTATION_DELETE = "删除标注"
    SEARCH_RUN = "执行检索"
    CLUSTER_RUN = "执行聚类"
    RECOMMEND_RUN = "执行推荐"
    EVIDENCE_ADD = "添加证据"
    EVIDENCE_DELETE = "删除证据"
    EVIDENCE_CHAIN_CREATE = "创建证据链"
    EVIDENCE_CHAIN_UPDATE = "更新证据链"
    EVIDENCE_CHAIN_DELETE = "删除证据链"
    CONCLUSION_CREATE = "创建研究结论"
    CONCLUSION_UPDATE = "更新研究结论"
    CONCLUSION_DELETE = "删除研究结论"
    CONCLUSION_REVIEW = "审核研究结论"
    KNOWLEDGE_GRAPH_BUILD = "构建知识图谱"
    INFERENCE_RUN = "执行推理"
    DEEP_COMPARE_RUN = "执行深度比对"


class ConflictType(Enum):
    EDGE_OVERLAP = "边缘重叠冲突"
    FRAGMENT_LOCK = "锁定残片冲突"
    SCHEME_DIRECTION = "方向矛盾冲突"
    REVIEW_DISPUTE = "争议匹配冲突"


class TagCategory(Enum):
    GLYPH_COMPONENT = "字形部件"
    KNIFE_MARK = "刀痕形态"
    WEATHERING = "风化特征"
    EDGE_TEXTURE = "边缘纹理"
    INSCRIPTION_TAG = "题跋标签"
    RESEARCH_TAG = "研究标签"
    CUSTOM = "自定义"


class SchemeOwnershipStatus(Enum):
    UNASSIGNED = "未归属"
    CANDIDATE = "候选归属"
    LOCKED = "已锁定归属"
    CONFLICTED = "归属冲突"


@dataclass
class EdgeFeature:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    fragment_id: str = ""
    direction: EdgeDirection = EdgeDirection.UP
    edge_type: EdgeType = EdgeType.BROKEN
    defect_type: DefectType = DefectType.INTACT
    length_px: int = 0
    curvature: float = 0.0
    roughness: float = 0.0
    descriptor: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "fragment_id": self.fragment_id,
            "direction": self.direction.value,
            "edge_type": self.edge_type.value,
            "defect_type": self.defect_type.value,
            "length_px": self.length_px,
            "curvature": self.curvature,
            "roughness": self.roughness,
            "descriptor": self.descriptor,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            fragment_id=d.get("fragment_id", ""),
            direction=EdgeDirection(d.get("direction", "上")),
            edge_type=EdgeType(d.get("edge_type", "断裂")),
            defect_type=DefectType(d.get("defect_type", "完好")),
            length_px=d.get("length_px", 0),
            curvature=d.get("curvature", 0.0),
            roughness=d.get("roughness", 0.0),
            descriptor=d.get("descriptor", []),
        )


@dataclass
class Fragment:
    id: str = ""
    image_data: bytes = b""
    image_format: str = ""
    image_width: int = 0
    image_height: int = 0
    edges: list = field(default_factory=list)
    inscriptions: str = ""
    is_locked: bool = False
    locked_scheme_id: Optional[str] = None
    notes: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        import base64
        return {
            "id": self.id,
            "image_data_b64": base64.b64encode(self.image_data).decode("ascii") if self.image_data else "",
            "image_format": self.image_format,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "edges": [e.to_dict() for e in self.edges],
            "inscriptions": self.inscriptions,
            "is_locked": self.is_locked,
            "locked_scheme_id": self.locked_scheme_id,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d):
        import base64
        edges = [EdgeFeature.from_dict(e) for e in d.get("edges", [])]
        img_b64 = d.get("image_data_b64", "")
        img_data = base64.b64decode(img_b64) if img_b64 else b""
        return cls(
            id=d.get("id", ""),
            image_data=img_data,
            image_format=d.get("image_format", ""),
            image_width=d.get("image_width", 0),
            image_height=d.get("image_height", 0),
            edges=edges,
            inscriptions=d.get("inscriptions", ""),
            is_locked=d.get("is_locked", False),
            locked_scheme_id=d.get("locked_scheme_id", None),
            notes=d.get("notes", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class EdgeMatch:
    edge_a_id: str = ""
    edge_a_fragment_id: str = ""
    edge_b_id: str = ""
    edge_b_fragment_id: str = ""
    similarity: float = 0.0
    review_status: ReviewStatus = ReviewStatus.PENDING
    review_comment: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""

    def to_dict(self):
        return {
            "edge_a_id": self.edge_a_id,
            "edge_a_fragment_id": self.edge_a_fragment_id,
            "edge_b_id": self.edge_b_id,
            "edge_b_fragment_id": self.edge_b_fragment_id,
            "similarity": self.similarity,
            "review_status": self.review_status.value,
            "review_comment": self.review_comment,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            edge_a_id=d.get("edge_a_id", ""),
            edge_a_fragment_id=d.get("edge_a_fragment_id", ""),
            edge_b_id=d.get("edge_b_id", ""),
            edge_b_fragment_id=d.get("edge_b_fragment_id", ""),
            similarity=d.get("similarity", 0.0),
            review_status=ReviewStatus(d.get("review_status", "待审核")),
            review_comment=d.get("review_comment", ""),
            reviewed_by=d.get("reviewed_by", ""),
            reviewed_at=d.get("reviewed_at", ""),
        )


@dataclass
class SchemeVersion:
    version_number: int = 1
    scheme_id: str = ""
    scheme_name: str = ""
    matches: list = field(default_factory=list)
    is_locked: bool = False
    description: str = ""
    created_at: str = ""
    created_by: str = ""

    def to_dict(self):
        return {
            "version_number": self.version_number,
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "matches": [m.to_dict() for m in self.matches],
            "is_locked": self.is_locked,
            "description": self.description,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, d):
        matches = [EdgeMatch.from_dict(m) for m in d.get("matches", [])]
        return cls(
            version_number=d.get("version_number", 1),
            scheme_id=d.get("scheme_id", ""),
            scheme_name=d.get("scheme_name", ""),
            matches=matches,
            is_locked=d.get("is_locked", False),
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
            created_by=d.get("created_by", ""),
        )


@dataclass
class Scheme:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    matches: list = field(default_factory=list)
    is_locked: bool = False
    versions: list = field(default_factory=list)
    current_version: int = 1
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "matches": [m.to_dict() for m in self.matches],
            "is_locked": self.is_locked,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, d):
        matches = [EdgeMatch.from_dict(m) for m in d.get("matches", [])]
        versions = [SchemeVersion.from_dict(v) for v in d.get("versions", [])]
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            name=d.get("name", ""),
            matches=matches,
            is_locked=d.get("is_locked", False),
            versions=versions,
            current_version=d.get("current_version", 1),
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            created_by=d.get("created_by", ""),
        )

    def get_involved_fragment_ids(self):
        ids = set()
        for m in self.matches:
            ids.add(m.edge_a_fragment_id)
            ids.add(m.edge_b_fragment_id)
        return ids

    def get_involved_edge_ids(self):
        ids = set()
        for m in self.matches:
            ids.add(m.edge_a_id)
            ids.add(m.edge_b_id)
        return ids

    def create_version(self, description="", created_by=""):
        import datetime
        version = SchemeVersion(
            version_number=len(self.versions) + 1,
            scheme_id=self.id,
            scheme_name=self.name,
            matches=[m.__class__.from_dict(m.to_dict()) for m in self.matches],
            is_locked=self.is_locked,
            description=description,
            created_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            created_by=created_by,
        )
        self.versions.append(version)
        self.current_version = version.version_number
        return version

    def restore_version(self, version_number):
        target = None
        for v in self.versions:
            if v.version_number == version_number:
                target = v
                break
        if target:
            self.matches = [m.__class__.from_dict(m.to_dict()) for m in target.matches]
            self.is_locked = target.is_locked
            self.name = target.scheme_name
            return True
        return False


@dataclass
class OperationLog:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    operation_type: OperationType = OperationType.FRAGMENT_ADD
    target_type: str = ""
    target_id: str = ""
    description: str = ""
    operator: str = ""
    timestamp: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "operation_type": self.operation_type.value,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "description": self.description,
            "operator": self.operator,
            "timestamp": self.timestamp,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            operation_type=OperationType(d.get("operation_type", "添加残片")),
            target_type=d.get("target_type", ""),
            target_id=d.get("target_id", ""),
            description=d.get("description", ""),
            operator=d.get("operator", ""),
            timestamp=d.get("timestamp", ""),
            details=d.get("details", {}),
        )


@dataclass
class ImportValidationIssue:
    level: str = "error"
    code: str = ""
    message: str = ""
    location: str = ""
    suggestion: str = ""

    def to_dict(self):
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            level=d.get("level", "error"),
            code=d.get("code", ""),
            message=d.get("message", ""),
            location=d.get("location", ""),
            suggestion=d.get("suggestion", ""),
        )


@dataclass
class ImportValidationReport:
    total_fragments: int = 0
    total_schemes: int = 0
    valid_fragments: int = 0
    valid_schemes: int = 0
    issues: list = field(default_factory=list)
    is_valid: bool = True
    summary: str = ""

    def to_dict(self):
        return {
            "total_fragments": self.total_fragments,
            "total_schemes": self.total_schemes,
            "valid_fragments": self.valid_fragments,
            "valid_schemes": self.valid_schemes,
            "issues": [i.to_dict() for i in self.issues],
            "is_valid": self.is_valid,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d):
        issues = [ImportValidationIssue.from_dict(i) for i in d.get("issues", [])]
        return cls(
            total_fragments=d.get("total_fragments", 0),
            total_schemes=d.get("total_schemes", 0),
            valid_fragments=d.get("valid_fragments", 0),
            valid_schemes=d.get("valid_schemes", 0),
            issues=issues,
            is_valid=d.get("is_valid", True),
            summary=d.get("summary", ""),
        )


@dataclass
class SchemeDiff:
    scheme_a_name: str = ""
    scheme_b_name: str = ""
    added_matches: list = field(default_factory=list)
    removed_matches: list = field(default_factory=list)
    common_matches: list = field(default_factory=list)
    similarity_score: float = 0.0
    coverage_diff: float = 0.0
    conflict_diff: int = 0

    def to_dict(self):
        return {
            "scheme_a_name": self.scheme_a_name,
            "scheme_b_name": self.scheme_b_name,
            "added_matches": [m.to_dict() for m in self.added_matches],
            "removed_matches": [m.to_dict() for m in self.removed_matches],
            "common_matches": [m.to_dict() for m in self.common_matches],
            "similarity_score": self.similarity_score,
            "coverage_diff": self.coverage_diff,
            "conflict_diff": self.conflict_diff,
        }

    @classmethod
    def from_dict(cls, d):
        added = [EdgeMatch.from_dict(m) for m in d.get("added_matches", [])]
        removed = [EdgeMatch.from_dict(m) for m in d.get("removed_matches", [])]
        common = [EdgeMatch.from_dict(m) for m in d.get("common_matches", [])]
        return cls(
            scheme_a_name=d.get("scheme_a_name", ""),
            scheme_b_name=d.get("scheme_b_name", ""),
            added_matches=added,
            removed_matches=removed,
            common_matches=common,
            similarity_score=d.get("similarity_score", 0.0),
            coverage_diff=d.get("coverage_diff", 0.0),
            conflict_diff=d.get("conflict_diff", 0),
        )


@dataclass
class ConflictInfo:
    conflict_type: ConflictType = ConflictType.EDGE_OVERLAP
    severity: str = "high"
    description: str = ""
    involved_schemes: list = field(default_factory=list)
    involved_fragments: list = field(default_factory=list)
    involved_edges: list = field(default_factory=list)
    details: str = ""

    def to_dict(self):
        return {
            "conflict_type": self.conflict_type.value,
            "severity": self.severity,
            "description": self.description,
            "involved_schemes": self.involved_schemes,
            "involved_fragments": self.involved_fragments,
            "involved_edges": self.involved_edges,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            conflict_type=ConflictType(d.get("conflict_type", "边缘重叠冲突")),
            severity=d.get("severity", "high"),
            description=d.get("description", ""),
            involved_schemes=d.get("involved_schemes", []),
            involved_fragments=d.get("involved_fragments", []),
            involved_edges=d.get("involved_edges", []),
            details=d.get("details", ""),
        )


@dataclass
class CustomTag:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    category: TagCategory = TagCategory.CUSTOM
    description: str = ""
    color: str = "#4ECDC4"
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "color": self.color,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            name=d.get("name", ""),
            category=TagCategory(d.get("category", "自定义")),
            description=d.get("description", ""),
            color=d.get("color", "#4ECDC4"),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class SemanticAnnotation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    fragment_id: str = ""
    glyph_components: list = field(default_factory=list)
    knife_marks: list = field(default_factory=list)
    weathering_features: list = field(default_factory=list)
    edge_textures: list = field(default_factory=list)
    inscription_content: str = ""
    inscription_tags: list = field(default_factory=list)
    research_notes: str = ""
    research_tags: list = field(default_factory=list)
    custom_tags: list = field(default_factory=list)
    custom_tag_ids: list = field(default_factory=list)
    annotated_by: str = ""
    annotated_at: str = ""
    updated_at: str = ""

    def get_all_tag_ids(self):
        all_tags = set(self.glyph_components) | set(self.knife_marks) | \
                   set(self.weathering_features) | set(self.edge_textures) | \
                   set(self.inscription_tags) | set(self.research_tags) | \
                   set(self.custom_tag_ids)
        return all_tags

    def to_dict(self):
        return {
            "id": self.id,
            "fragment_id": self.fragment_id,
            "glyph_components": self.glyph_components,
            "knife_marks": self.knife_marks,
            "weathering_features": self.weathering_features,
            "edge_textures": self.edge_textures,
            "inscription_content": self.inscription_content,
            "inscription_tags": self.inscription_tags,
            "research_notes": self.research_notes,
            "research_tags": self.research_tags,
            "custom_tags": self.custom_tags,
            "custom_tag_ids": self.custom_tag_ids,
            "annotated_by": self.annotated_by,
            "annotated_at": self.annotated_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            fragment_id=d.get("fragment_id", ""),
            glyph_components=d.get("glyph_components", []),
            knife_marks=d.get("knife_marks", []),
            weathering_features=d.get("weathering_features", []),
            edge_textures=d.get("edge_textures", []),
            inscription_content=d.get("inscription_content", ""),
            inscription_tags=d.get("inscription_tags", []),
            research_notes=d.get("research_notes", ""),
            research_tags=d.get("research_tags", []),
            custom_tags=d.get("custom_tags", []),
            custom_tag_ids=d.get("custom_tag_ids", []),
            annotated_by=d.get("annotated_by", ""),
            annotated_at=d.get("annotated_at", ""),
            updated_at=d.get("updated_at", ""),
        )


@dataclass
class FragmentCluster:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    fragment_ids: list = field(default_factory=list)
    cluster_method: str = ""
    cluster_params: dict = field(default_factory=dict)
    representative_fragment_id: str = ""
    description: str = ""
    created_by: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "fragment_ids": self.fragment_ids,
            "cluster_method": self.cluster_method,
            "cluster_params": self.cluster_params,
            "representative_fragment_id": self.representative_fragment_id,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            name=d.get("name", ""),
            fragment_ids=d.get("fragment_ids", []),
            cluster_method=d.get("cluster_method", ""),
            cluster_params=d.get("cluster_params", {}),
            representative_fragment_id=d.get("representative_fragment_id", ""),
            description=d.get("description", ""),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
        )


@dataclass
class SearchResult:
    fragment_id: str = ""
    relevance_score: float = 0.0
    matched_fields: list = field(default_factory=list)
    matched_tags: list = field(default_factory=list)
    ownership_status: SchemeOwnershipStatus = SchemeOwnershipStatus.UNASSIGNED
    associated_scheme_ids: list = field(default_factory=list)
    locked_scheme_id: str = ""

    def to_dict(self):
        return {
            "fragment_id": self.fragment_id,
            "relevance_score": self.relevance_score,
            "matched_fields": self.matched_fields,
            "matched_tags": self.matched_tags,
            "ownership_status": self.ownership_status.value,
            "associated_scheme_ids": self.associated_scheme_ids,
            "locked_scheme_id": self.locked_scheme_id,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            fragment_id=d.get("fragment_id", ""),
            relevance_score=d.get("relevance_score", 0.0),
            matched_fields=d.get("matched_fields", []),
            matched_tags=d.get("matched_tags", []),
            ownership_status=SchemeOwnershipStatus(d.get("ownership_status", "未归属")),
            associated_scheme_ids=d.get("associated_scheme_ids", []),
            locked_scheme_id=d.get("locked_scheme_id", ""),
        )


@dataclass
class ResearchClue:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    clue_type: str = ""
    title: str = ""
    description: str = ""
    fragment_ids: list = field(default_factory=list)
    scheme_ids: list = field(default_factory=list)
    confidence: float = 0.0
    suggestion: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "clue_type": self.clue_type,
            "title": self.title,
            "description": self.description,
            "fragment_ids": self.fragment_ids,
            "scheme_ids": self.scheme_ids,
            "confidence": self.confidence,
            "suggestion": self.suggestion,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            clue_type=d.get("clue_type", ""),
            title=d.get("title", ""),
            description=d.get("description", ""),
            fragment_ids=d.get("fragment_ids", []),
            scheme_ids=d.get("scheme_ids", []),
            confidence=d.get("confidence", 0.0),
            suggestion=d.get("suggestion", ""),
            created_at=d.get("created_at", ""),
        )


class TagRelationType(Enum):
    SYNONYM = "同义关系"
    HYPERNYM = "上位关系"
    HYPONYM = "下位关系"
    RELATED = "相关关系"
    OPPOSITE = "反义关系"
    PART_OF = "部分关系"


@dataclass
class TagRelation:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_tag_id: str = ""
    target_tag_id: str = ""
    relation_type: TagRelationType = TagRelationType.RELATED
    confidence: float = 1.0
    description: str = ""
    created_by: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "source_tag_id": self.source_tag_id,
            "target_tag_id": self.target_tag_id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            source_tag_id=d.get("source_tag_id", ""),
            target_tag_id=d.get("target_tag_id", ""),
            relation_type=TagRelationType(d.get("relation_type", "相关关系")),
            confidence=d.get("confidence", 1.0),
            description=d.get("description", ""),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
        )


class EvidenceType(Enum):
    IMAGE_MATCH = "图像匹配"
    EDGE_FEATURE = "边缘特征"
    SEMANTIC_TAG = "语义标签"
    INSCRIPTION_TEXT = "题跋文字"
    EXPERT_OPINION = "专家意见"
    HISTORICAL_RECORD = "历史记录"
    SCHEME_EVIDENCE = "方案佐证"
    OTHER = "其他证据"


@dataclass
class EvidenceItem:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    evidence_type: EvidenceType = EvidenceType.OTHER
    title: str = ""
    description: str = ""
    source_type: str = ""
    source_id: str = ""
    fragment_ids: list = field(default_factory=list)
    scheme_ids: list = field(default_factory=list)
    tag_ids: list = field(default_factory=list)
    confidence: float = 1.0
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    references: list = field(default_factory=list)

    def to_dict(self):
        return {
            "id": self.id,
            "evidence_type": self.evidence_type.value,
            "title": self.title,
            "description": self.description,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "fragment_ids": self.fragment_ids,
            "scheme_ids": self.scheme_ids,
            "tag_ids": self.tag_ids,
            "confidence": self.confidence,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "references": self.references,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            evidence_type=EvidenceType(d.get("evidence_type", "其他证据")),
            title=d.get("title", ""),
            description=d.get("description", ""),
            source_type=d.get("source_type", ""),
            source_id=d.get("source_id", ""),
            fragment_ids=d.get("fragment_ids", []),
            scheme_ids=d.get("scheme_ids", []),
            tag_ids=d.get("tag_ids", []),
            confidence=d.get("confidence", 1.0),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            references=d.get("references", []),
        )


@dataclass
class EvidenceChain:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = ""
    description: str = ""
    evidence_ids: list = field(default_factory=list)
    conclusion_id: str = ""
    chain_strength: float = 0.0
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "evidence_ids": self.evidence_ids,
            "conclusion_id": self.conclusion_id,
            "chain_strength": self.chain_strength,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            title=d.get("title", ""),
            description=d.get("description", ""),
            evidence_ids=d.get("evidence_ids", []),
            conclusion_id=d.get("conclusion_id", ""),
            chain_strength=d.get("chain_strength", 0.0),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )


class ConclusionStatus(Enum):
    PROPOSED = "待论证"
    EVIDENCE_BUILDING = "证据构建中"
    PENDING_REVIEW = "待审核"
    REVIEWING = "审核中"
    ACCEPTED = "已采纳"
    REJECTED = "已否决"
    REVISED = "已修订"
    CONTROVERSIAL = "有争议"


@dataclass
class ResearchConclusion:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str = ""
    content: str = ""
    conclusion_type: str = ""
    fragment_ids: list = field(default_factory=list)
    scheme_ids: list = field(default_factory=list)
    tag_ids: list = field(default_factory=list)
    evidence_chain_ids: list = field(default_factory=list)
    status: ConclusionStatus = ConclusionStatus.PROPOSED
    confidence: float = 0.0
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    version: int = 1
    version_history: list = field(default_factory=list)
    review_records: list = field(default_factory=list)
    parent_conclusion_id: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "conclusion_type": self.conclusion_type,
            "fragment_ids": self.fragment_ids,
            "scheme_ids": self.scheme_ids,
            "tag_ids": self.tag_ids,
            "evidence_chain_ids": self.evidence_chain_ids,
            "status": self.status.value,
            "confidence": self.confidence,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "version_history": [v.to_dict() if hasattr(v, 'to_dict') else v for v in self.version_history],
            "review_records": [r.to_dict() if hasattr(r, 'to_dict') else r for r in self.review_records],
            "parent_conclusion_id": self.parent_conclusion_id,
        }

    @classmethod
    def from_dict(cls, d):
        version_history = d.get("version_history", [])
        review_records = d.get("review_records", [])
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            title=d.get("title", ""),
            content=d.get("content", ""),
            conclusion_type=d.get("conclusion_type", ""),
            fragment_ids=d.get("fragment_ids", []),
            scheme_ids=d.get("scheme_ids", []),
            tag_ids=d.get("tag_ids", []),
            evidence_chain_ids=d.get("evidence_chain_ids", []),
            status=ConclusionStatus(d.get("status", "待论证")),
            confidence=d.get("confidence", 0.0),
            created_by=d.get("created_by", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            version=d.get("version", 1),
            version_history=[ConclusionVersion.from_dict(v) if isinstance(v, dict) else v for v in version_history],
            review_records=[ReviewRecord.from_dict(r) if isinstance(r, dict) else r for r in review_records],
            parent_conclusion_id=d.get("parent_conclusion_id", ""),
        )


@dataclass
class ConclusionVersion:
    version_number: int = 1
    title: str = ""
    content: str = ""
    status: str = ""
    confidence: float = 0.0
    changed_by: str = ""
    changed_at: str = ""
    change_description: str = ""

    def to_dict(self):
        return {
            "version_number": self.version_number,
            "title": self.title,
            "content": self.content,
            "status": self.status,
            "confidence": self.confidence,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at,
            "change_description": self.change_description,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            version_number=d.get("version_number", 1),
            title=d.get("title", ""),
            content=d.get("content", ""),
            status=d.get("status", ""),
            confidence=d.get("confidence", 0.0),
            changed_by=d.get("changed_by", ""),
            changed_at=d.get("changed_at", ""),
            change_description=d.get("change_description", ""),
        )


class ReviewDecision(Enum):
    APPROVE = "同意"
    REJECT = "否决"
    REVISE = "需修订"
    COMMENT = "仅评论"
    FLAG = "存疑标记"


@dataclass
class ReviewRecord:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    target_type: str = ""
    target_id: str = ""
    decision: ReviewDecision = ReviewDecision.COMMENT
    comment: str = ""
    evidence_refs: list = field(default_factory=list)
    reviewed_by: str = ""
    reviewed_at: str = ""
    is_official: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "decision": self.decision.value,
            "comment": self.comment,
            "evidence_refs": self.evidence_refs,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "is_official": self.is_official,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            target_type=d.get("target_type", ""),
            target_id=d.get("target_id", ""),
            decision=ReviewDecision(d.get("decision", "仅评论")),
            comment=d.get("comment", ""),
            evidence_refs=d.get("evidence_refs", []),
            reviewed_by=d.get("reviewed_by", ""),
            reviewed_at=d.get("reviewed_at", ""),
            is_official=d.get("is_official", False),
        )


class KGNodeType(Enum):
    FRAGMENT = "残片"
    EDGE_FEATURE = "边缘特征"
    TAG = "标签"
    INSCRIPTION = "题跋"
    SCHEME = "方案"
    CONCLUSION = "研究结论"
    EVIDENCE = "证据"
    RESEARCHER = "研究员"


class KGEdgeType(Enum):
    HAS_EDGE = "具有边缘"
    HAS_TAG = "具有标签"
    RELATED_TO = "相关于"
    MATCHES = "匹配"
    BELONGS_TO = "属于"
    SUPPORTS = "支持"
    CONTRADICTS = "反驳"
    DERIVED_FROM = "源自"
    CREATED_BY = "创建者"
    TAG_RELATION = "标签关系"


@dataclass
class KnowledgeGraph:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    description: str = ""
    created_by: str = ""
    created_at: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "nodes": [n.to_dict() if hasattr(n, 'to_dict') else n for n in self.nodes],
            "edges": [e.to_dict() if hasattr(e, 'to_dict') else e for e in self.edges],
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }


@dataclass
class KGNode:
    id: str = ""
    node_type: KGNodeType = KGNodeType.FRAGMENT
    label: str = ""
    description: str = ""
    properties: dict = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0

    def to_dict(self):
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "label": self.label,
            "description": self.description,
            "properties": self.properties,
            "x": self.x,
            "y": self.y,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", ""),
            node_type=KGNodeType(d.get("node_type", "残片")),
            label=d.get("label", ""),
            description=d.get("description", ""),
            properties=d.get("properties", {}),
            x=d.get("x", 0.0),
            y=d.get("y", 0.0),
        )


@dataclass
class KGEdge:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    source_id: str = ""
    target_id: str = ""
    edge_type: KGEdgeType = KGEdgeType.RELATED_TO
    label: str = ""
    weight: float = 1.0
    properties: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "label": self.label,
            "weight": self.weight,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            source_id=d.get("source_id", ""),
            target_id=d.get("target_id", ""),
            edge_type=KGEdgeType(d.get("edge_type", "相关于")),
            label=d.get("label", ""),
            weight=d.get("weight", 1.0),
            properties=d.get("properties", {}),
        )


@dataclass
class InferencePath:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    start_node_id: str = ""
    end_node_id: str = ""
    path_nodes: list = field(default_factory=list)
    path_edges: list = field(default_factory=list)
    path_strength: float = 0.0
    description: str = ""

    def to_dict(self):
        return {
            "id": self.id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "path_nodes": self.path_nodes,
            "path_edges": self.path_edges,
            "path_strength": self.path_strength,
            "description": self.description,
        }


class SchemeCompareDepth(Enum):
    BASIC = "基础对比"
    EVIDENCE = "证据对比"
    CONCLUSION = "结论对比"
    FULL = "全面对比"


@dataclass
class SchemeDeepCompareResult:
    scheme_a_id: str = ""
    scheme_b_id: str = ""
    scheme_a_name: str = ""
    scheme_b_name: str = ""
    compare_depth: SchemeCompareDepth = SchemeCompareDepth.BASIC
    match_similarity: float = 0.0
    fragment_overlap: list = field(default_factory=list)
    edge_overlap: list = field(default_factory=list)
    tag_similarity: float = 0.0
    shared_evidence: list = field(default_factory=list)
    conflicting_evidence: list = field(default_factory=list)
    conclusion_alignment: list = field(default_factory=list)
    conflict_points: list = field(default_factory=list)
    consensus_points: list = field(default_factory=list)
    overall_consistency_score: float = 0.0

    def to_dict(self):
        return {
            "scheme_a_id": self.scheme_a_id,
            "scheme_b_id": self.scheme_b_id,
            "scheme_a_name": self.scheme_a_name,
            "scheme_b_name": self.scheme_b_name,
            "compare_depth": self.compare_depth.value,
            "match_similarity": self.match_similarity,
            "fragment_overlap": self.fragment_overlap,
            "edge_overlap": self.edge_overlap,
            "tag_similarity": self.tag_similarity,
            "shared_evidence": self.shared_evidence,
            "conflicting_evidence": self.conflicting_evidence,
            "conclusion_alignment": self.conclusion_alignment,
            "conflict_points": self.conflict_points,
            "consensus_points": self.consensus_points,
            "overall_consistency_score": self.overall_consistency_score,
        }
