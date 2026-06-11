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


class ConflictType(Enum):
    EDGE_OVERLAP = "边缘重叠冲突"
    FRAGMENT_LOCK = "锁定残片冲突"
    SCHEME_DIRECTION = "方向矛盾冲突"
    REVIEW_DISPUTE = "争议匹配冲突"


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
