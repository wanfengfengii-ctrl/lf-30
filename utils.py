import numpy as np
import networkx as nx
from models import Fragment, EdgeFeature, EdgeMatch, Scheme, EdgeDirection, EdgeType, DefectType


OPPOSITE_DIRECTIONS = {
    EdgeDirection.UP: EdgeDirection.DOWN,
    EdgeDirection.DOWN: EdgeDirection.UP,
    EdgeDirection.LEFT: EdgeDirection.RIGHT,
    EdgeDirection.RIGHT: EdgeDirection.LEFT,
    EdgeDirection.UPPER_LEFT: EdgeDirection.LOWER_RIGHT,
    EdgeDirection.LOWER_RIGHT: EdgeDirection.UPPER_LEFT,
    EdgeDirection.UPPER_RIGHT: EdgeDirection.LOWER_LEFT,
    EdgeDirection.LOWER_LEFT: EdgeDirection.UPPER_RIGHT,
}

COMPATIBLE_EDGE_TYPES = {
    (EdgeType.BROKEN, EdgeType.BROKEN),
    (EdgeType.IRREGULAR, EdgeType.IRREGULAR),
    (EdgeType.IRREGULAR, EdgeType.BROKEN),
    (EdgeType.BROKEN, EdgeType.IRREGULAR),
    (EdgeType.CURVED, EdgeType.CURVED),
    (EdgeType.SERRATED, EdgeType.SERRATED),
    (EdgeType.STRAIGHT, EdgeType.STRAIGHT),
}


def validate_image(image_bytes):
    from PIL import Image
    import io
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        if img.size[0] < 10 or img.size[1] < 10:
            return False, "图像尺寸过小（小于10px），可能无效"
        return True, ""
    except Exception as e:
        return False, f"无效图像文件: {str(e)}"


def check_duplicate_fragment_id(fragments, fragment_id):
    for f in fragments:
        if f.id == fragment_id:
            return True
    return False


def compute_edge_similarity(edge_a, edge_b):
    score = 0.0
    if edge_b.direction == OPPOSITE_DIRECTIONS.get(edge_a.direction):
        score += 0.30
    else:
        score -= 0.20

    if (edge_a.edge_type, edge_b.edge_type) in COMPATIBLE_EDGE_TYPES:
        score += 0.20
    else:
        score -= 0.15

    if edge_a.defect_type == edge_b.defect_type:
        score += 0.10

    length_diff = abs(edge_a.length_px - edge_b.length_px)
    max_len = max(edge_a.length_px, edge_b.length_px, 1)
    length_ratio = 1.0 - (length_diff / max_len)
    score += length_ratio * 0.15

    curvature_diff = abs(edge_a.curvature - edge_b.curvature)
    score += max(0, 0.10 - curvature_diff * 0.05)

    roughness_diff = abs(edge_a.roughness - edge_b.roughness)
    score += max(0, 0.05 - roughness_diff * 0.02)

    if edge_a.descriptor and edge_b.descriptor:
        desc_a = np.array(edge_a.descriptor)
        desc_b = np.array(edge_b.descriptor)
        if desc_a.shape == desc_b.shape:
            norm_a = desc_a / (np.linalg.norm(desc_a) + 1e-8)
            norm_b = desc_b / (np.linalg.norm(desc_b) + 1e-8)
            cos_sim = float(np.dot(norm_a, norm_b))
            score += cos_sim * 0.10

    return max(0.0, min(1.0, score))


def find_candidate_matches(fragments, similarity_threshold=0.40):
    matches = []
    for i, frag_a in enumerate(fragments):
        for j, frag_b in enumerate(fragments):
            if i >= j:
                continue
            for edge_a in frag_a.edges:
                for edge_b in frag_b.edges:
                    sim = compute_edge_similarity(edge_a, edge_b)
                    if sim >= similarity_threshold:
                        matches.append(EdgeMatch(
                            edge_a_id=edge_a.id,
                            edge_a_fragment_id=frag_a.id,
                            edge_b_id=edge_b.id,
                            edge_b_fragment_id=frag_b.id,
                            similarity=sim,
                        ))
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


def build_candidate_graph(fragments, matches):
    G = nx.Graph()
    for f in fragments:
        G.add_node(f.id, label=f.id)
    for m in matches:
        G.add_edge(
            m.edge_a_fragment_id,
            m.edge_b_fragment_id,
            similarity=m.similarity,
            edge_a_id=m.edge_a_id,
            edge_b_id=m.edge_b_id,
        )
    return G


def compute_scheme_coverage(scheme, fragments):
    if not fragments:
        return 0.0, 0, 0
    total_edges = sum(len(f.edges) for f in fragments)
    if total_edges == 0:
        return 0.0, 0, 0
    matched_edge_ids = scheme.get_involved_edge_ids()
    matched_count = 0
    for f in fragments:
        for e in f.edges:
            if e.id in matched_edge_ids:
                matched_count += 1
    coverage = matched_count / total_edges
    return coverage, matched_count, total_edges


def compute_scheme_conflicts(scheme, all_schemes, fragments):
    conflicts = []
    scheme_edge_ids = scheme.get_involved_edge_ids()
    scheme_frag_ids = scheme.get_involved_fragment_ids()

    for other in all_schemes:
        if other.id == scheme.id:
            continue
        other_edge_ids = other.get_involved_edge_ids()
        other_frag_ids = other.get_involved_fragment_ids()

        edge_overlap = scheme_edge_ids & other_edge_ids
        if edge_overlap:
            conflicts.append({
                "type": "边缘冲突",
                "conflicting_scheme_id": other.id,
                "detail": f"边缘 {edge_overlap} 在方案 {scheme.id} 和 {other.id} 中重复使用",
                "overlap_ids": list(edge_overlap),
            })

        for frag_id in scheme_frag_ids & other_frag_ids:
            frag = next((f for f in fragments if f.id == frag_id), None)
            if frag and frag.is_locked and frag.locked_scheme_id not in (scheme.id, other.id):
                conflicts.append({
                    "type": "锁定残片冲突",
                    "conflicting_scheme_id": other.id,
                    "detail": f"残片 {frag_id} 已锁定于方案 {frag.locked_scheme_id}",
                    "fragment_id": frag_id,
                })

    return conflicts


def validate_edge_exclusivity(scheme):
    edge_ids_seen = set()
    for m in scheme.matches:
        for eid in [m.edge_a_id, m.edge_b_id]:
            if eid in edge_ids_seen:
                return False, f"边缘 {eid} 在方案中被重复匹配了多个对象"
            edge_ids_seen.add(eid)
    return True, ""


def validate_locked_fragment(fragment_id, fragments, scheme_id):
    frag = next((f for f in fragments if f.id == fragment_id), None)
    if frag and frag.is_locked and frag.locked_scheme_id != scheme_id:
        return False, f"残片 {fragment_id} 已锁定于方案 {frag.locked_scheme_id}，不能被方案 {scheme_id} 占用"
    return True, ""


def validate_delete_fragment(fragment_id, schemes):
    involved_schemes = []
    for s in schemes:
        if fragment_id in s.get_involved_fragment_ids():
            involved_schemes.append(s.id)
    return involved_schemes


def validate_export_scheme(scheme):
    if not scheme.matches:
        return False, "方案无任何候选匹配关系，导出无效"
    for m in scheme.matches:
        if not m.edge_a_id or not m.edge_b_id:
            return False, "存在不完整的候选关系，缺少边缘ID"
        if not m.edge_a_fragment_id or not m.edge_b_fragment_id:
            return False, "存在不完整的候选关系，缺少残片ID"
    return True, ""


def validate_import_scheme(scheme_dict, existing_schemes, existing_fragments):
    try:
        scheme = Scheme.from_dict(scheme_dict)
    except Exception as e:
        return None, f"方案格式无效: {str(e)}"

    frag_map = {f.id: f for f in existing_fragments}
    for m in scheme.matches:
        if m.edge_a_fragment_id not in frag_map:
            return None, f"候选关系引用了不存在的残片 {m.edge_a_fragment_id}"
        if m.edge_b_fragment_id not in frag_map:
            return None, f"候选关系引用了不存在的残片 {m.edge_b_fragment_id}"
        frag_a = frag_map[m.edge_a_fragment_id]
        frag_b = frag_map[m.edge_b_fragment_id]
        if not any(e.id == m.edge_a_id for e in frag_a.edges):
            return None, f"残片 {m.edge_a_fragment_id} 中不存在边缘 {m.edge_a_id}"
        if not any(e.id == m.edge_b_id for e in frag_b.edges):
            return None, f"残片 {m.edge_b_fragment_id} 中不存在边缘 {m.edge_b_id}"

    for existing in existing_schemes:
        if existing.id == scheme.id:
            return None, f"方案ID {scheme.id} 已存在于当前分析中，不能覆盖"

    ok, msg = validate_edge_exclusivity(scheme)
    if not ok:
        return None, msg

    return scheme, ""


def compute_descriptor_from_image(image_bytes):
    from PIL import Image
    import io
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L")
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32).flatten()
        arr = (arr - arr.mean()) / (arr.std() + 1e-8)
        return arr.tolist()[:32]
    except Exception:
        return []
