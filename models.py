import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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
        )


@dataclass
class EdgeMatch:
    edge_a_id: str = ""
    edge_a_fragment_id: str = ""
    edge_b_id: str = ""
    edge_b_fragment_id: str = ""
    similarity: float = 0.0

    def to_dict(self):
        return {
            "edge_a_id": self.edge_a_id,
            "edge_a_fragment_id": self.edge_a_fragment_id,
            "edge_b_id": self.edge_b_id,
            "edge_b_fragment_id": self.edge_b_fragment_id,
            "similarity": self.similarity,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            edge_a_id=d.get("edge_a_id", ""),
            edge_a_fragment_id=d.get("edge_a_fragment_id", ""),
            edge_b_id=d.get("edge_b_id", ""),
            edge_b_fragment_id=d.get("edge_b_fragment_id", ""),
            similarity=d.get("similarity", 0.0),
        )


@dataclass
class Scheme:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    matches: list = field(default_factory=list)
    is_locked: bool = False

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "matches": [m.to_dict() for m in self.matches],
            "is_locked": self.is_locked,
        }

    @classmethod
    def from_dict(cls, d):
        matches = [EdgeMatch.from_dict(m) for m in d.get("matches", [])]
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            name=d.get("name", ""),
            matches=matches,
            is_locked=d.get("is_locked", False),
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
