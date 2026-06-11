import datetime
import numpy as np
import networkx as nx
from models import (
    Fragment, EdgeFeature, EdgeMatch, Scheme, EdgeDirection, EdgeType, DefectType,
    ReviewStatus, OperationType, OperationLog, SchemeVersion, SchemeDiff,
    ImportValidationReport, ImportValidationIssue, ConflictInfo, ConflictType
)


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


def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_operation_log(op_type, target_type, target_id, description, operator="", details=None):
    log = OperationLog(
        operation_type=op_type,
        target_type=target_type,
        target_id=target_id,
        description=description,
        operator=operator,
        timestamp=now_str(),
        details=details or {},
    )
    return log


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
                            review_status=ReviewStatus.PENDING,
                        ))
    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


def build_candidate_graph(fragments, matches):
    G = nx.MultiGraph()
    for f in fragments:
        G.add_node(f.id, label=f.id)
    for idx, m in enumerate(matches):
        G.add_edge(
            m.edge_a_fragment_id,
            m.edge_b_fragment_id,
            key=idx,
            similarity=m.similarity,
            edge_a_id=m.edge_a_id,
            edge_b_id=m.edge_b_id,
            review_status=m.review_status.value,
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


def detect_all_conflicts(all_schemes, fragments):
    conflict_infos = []

    for i, scheme_a in enumerate(all_schemes):
        for j, scheme_b in enumerate(all_schemes):
            if i >= j:
                continue

            edges_a = scheme_a.get_involved_edge_ids()
            edges_b = scheme_b.get_involved_edge_ids()
            edge_overlap = edges_a & edges_b

            if edge_overlap:
                frags_involved = set()
                for m in scheme_a.matches:
                    if m.edge_a_id in edge_overlap or m.edge_b_id in edge_overlap:
                        frags_involved.add(m.edge_a_fragment_id)
                        frags_involved.add(m.edge_b_fragment_id)

                conflict_infos.append(ConflictInfo(
                    conflict_type=ConflictType.EDGE_OVERLAP,
                    severity="high",
                    description=f"方案 '{scheme_a.name}' 与方案 '{scheme_b.name}' 存在边缘重叠冲突",
                    involved_schemes=[scheme_a.id, scheme_b.id],
                    involved_fragments=list(frags_involved),
                    involved_edges=list(edge_overlap),
                    details=f"重叠边缘数: {len(edge_overlap)}",
                ))

    for scheme in all_schemes:
        frag_ids = scheme.get_involved_fragment_ids()
        for fid in frag_ids:
            frag = next((f for f in fragments if f.id == fid), None)
            if frag and frag.is_locked and frag.locked_scheme_id != scheme.id:
                locked_scheme = next((s for s in all_schemes if s.id == frag.locked_scheme_id), None)
                locked_name = locked_scheme.name if locked_scheme else frag.locked_scheme_id
                conflict_infos.append(ConflictInfo(
                    conflict_type=ConflictType.FRAGMENT_LOCK,
                    severity="high",
                    description=f"残片 '{fid}' 已被方案 '{locked_name}' 锁定，但方案 '{scheme.name}' 也在使用",
                    involved_schemes=[scheme.id, frag.locked_scheme_id],
                    involved_fragments=[fid],
                    involved_edges=[],
                    details=f"锁定方案: {locked_name}, 冲突方案: {scheme.name}",
                ))

    for scheme in all_schemes:
        approved_matches = [m for m in scheme.matches if m.review_status == ReviewStatus.APPROVED]
        flagged_matches = [m for m in scheme.matches if m.review_status == ReviewStatus.FLAGGED]
        rejected_matches = [m for m in scheme.matches if m.review_status == ReviewStatus.REJECTED]

        if flagged_matches and approved_matches:
            for fm in flagged_matches:
                frags = {fm.edge_a_fragment_id, fm.edge_b_fragment_id}
                for am in approved_matches:
                    if {am.edge_a_fragment_id, am.edge_b_fragment_id} & frags:
                        conflict_infos.append(ConflictInfo(
                            conflict_type=ConflictType.REVIEW_DISPUTE,
                            severity="medium",
                            description=f"方案 '{scheme.name}' 中存在存疑匹配与已通过匹配的争议",
                            involved_schemes=[scheme.id],
                            involved_fragments=list(frags),
                            involved_edges=[fm.edge_a_id, fm.edge_b_id],
                            details=f"存疑匹配涉及残片 {frags}",
                        ))
                        break

    return conflict_infos


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


def create_scheme_version(scheme, description="", created_by=""):
    return scheme.create_version(description, created_by)


def restore_scheme_version(scheme, version_number):
    return scheme.restore_version(version_number)


def compare_schemes(scheme_a, scheme_b, fragments=None):
    def match_key(m):
        return tuple(sorted([
            (m.edge_a_fragment_id, m.edge_a_id),
            (m.edge_b_fragment_id, m.edge_b_id),
        ]))

    keys_a = {match_key(m): m for m in scheme_a.matches}
    keys_b = {match_key(m): m for m in scheme_b.matches}

    set_a = set(keys_a.keys())
    set_b = set(keys_b.keys())

    added_keys = set_b - set_a
    removed_keys = set_a - set_b
    common_keys = set_a & set_b

    added_matches = [keys_b[k] for k in added_keys]
    removed_matches = [keys_a[k] for k in removed_keys]
    common_matches = [keys_a[k] for k in common_keys]

    total = len(set_a | set_b)
    similarity = len(common_keys) / total if total > 0 else 1.0

    coverage_diff = 0.0
    if fragments:
        cov_a, _, _ = compute_scheme_coverage(scheme_a, fragments)
        cov_b, _, _ = compute_scheme_coverage(scheme_b, fragments)
        coverage_diff = cov_b - cov_a

    return SchemeDiff(
        scheme_a_name=scheme_a.name,
        scheme_b_name=scheme_b.name,
        added_matches=added_matches,
        removed_matches=removed_matches,
        common_matches=common_matches,
        similarity_score=similarity,
        coverage_diff=coverage_diff,
        conflict_diff=0,
    )


def validate_import_data(data, existing_fragments, existing_schemes):
    report = ImportValidationReport()
    report.total_fragments = len(data.get("fragments", []))
    report.total_schemes = len(data.get("schemes", []))

    existing_frag_ids = {f.id for f in existing_fragments}
    existing_scheme_ids = {s.id for s in existing_schemes}

    valid_frags = 0
    frag_id_map = {}
    for i, frag_dict in enumerate(data.get("fragments", [])):
        fid = frag_dict.get("id", "")
        location = f"fragments[{i}]"

        if not fid:
            report.issues.append(ImportValidationIssue(
                level="error",
                code="FRAG_ID_EMPTY",
                message="残片ID为空",
                location=location,
                suggestion="请为该残片设置有效的ID",
            ))
            continue

        if fid in existing_frag_ids:
            report.issues.append(ImportValidationIssue(
                level="error",
                code="FRAG_ID_DUPLICATE",
                message=f"残片ID '{fid}' 与现有数据冲突",
                location=location,
                suggestion="重命名该残片ID后再导入",
            ))
            continue

        if fid in frag_id_map:
            report.issues.append(ImportValidationIssue(
                level="error",
                code="FRAG_ID_DUPLICATE_IN_FILE",
                message=f"残片ID '{fid}' 在导入文件中重复",
                location=location,
                suggestion="修正文件中的重复ID",
            ))
            continue

        edges = frag_dict.get("edges", [])
        edge_ids = set()
        for j, edge in enumerate(edges):
            eid = edge.get("id", "")
            if not eid:
                report.issues.append(ImportValidationIssue(
                    level="warning",
                    code="EDGE_ID_EMPTY",
                    message=f"边缘 {j} 的ID为空",
                    location=f"{location}.edges[{j}]",
                    suggestion="系统将自动生成边缘ID",
                ))
            elif eid in edge_ids:
                report.issues.append(ImportValidationIssue(
                    level="error",
                    code="EDGE_ID_DUPLICATE",
                    message=f"边缘ID '{eid}' 在同一残片中重复",
                    location=f"{location}.edges[{j}]",
                    suggestion="修正重复的边缘ID",
                ))
            edge_ids.add(eid)

        img_b64 = frag_dict.get("image_data_b64", "")
        if img_b64:
            try:
                import base64
                img_data = base64.b64decode(img_b64)
                valid, msg = validate_image(img_data)
                if not valid:
                    report.issues.append(ImportValidationIssue(
                        level="warning",
                        code="FRAG_IMAGE_INVALID",
                        message=f"残片图像无效: {msg}",
                        location=location,
                        suggestion="检查图像数据是否完整",
                    ))
            except Exception as e:
                report.issues.append(ImportValidationIssue(
                    level="warning",
                    code="FRAG_IMAGE_DECODE_ERROR",
                    message=f"残片图像解码失败: {str(e)}",
                    location=location,
                    suggestion="图像数据可能损坏",
                ))

        frag_id_map[fid] = frag_dict
        valid_frags += 1

    report.valid_fragments = valid_frags

    valid_schemes = 0
    for i, scheme_dict in enumerate(data.get("schemes", [])):
        sid = scheme_dict.get("id", "")
        location = f"schemes[{i}]"
        scheme_valid = True

        if not sid:
            report.issues.append(ImportValidationIssue(
                level="error",
                code="SCHEME_ID_EMPTY",
                message="方案ID为空",
                location=location,
                suggestion="请为该方案设置有效的ID",
            ))
            scheme_valid = False

        if sid in existing_scheme_ids:
            report.issues.append(ImportValidationIssue(
                level="error",
                code="SCHEME_ID_DUPLICATE",
                message=f"方案ID '{sid}' 与现有数据冲突",
                location=location,
                suggestion="重命名该方案ID后再导入",
            ))
            scheme_valid = False

        matches = scheme_dict.get("matches", [])
        if not matches:
            report.issues.append(ImportValidationIssue(
                level="warning",
                code="SCHEME_NO_MATCHES",
                message="方案无任何匹配关系",
                location=location,
                suggestion="空方案可导入但无实际意义",
            ))

        for j, m in enumerate(matches):
            fa_id = m.get("edge_a_fragment_id", "")
            fb_id = m.get("edge_b_fragment_id", "")
            ea_id = m.get("edge_a_id", "")
            eb_id = m.get("edge_b_id", "")
            match_loc = f"{location}.matches[{j}]"

            if fa_id not in frag_id_map and fa_id not in existing_frag_ids:
                report.issues.append(ImportValidationIssue(
                    level="error",
                    code="MATCH_FRAG_NOT_FOUND",
                    message=f"匹配引用了不存在的残片 '{fa_id}'",
                    location=match_loc,
                    suggestion="确保引用的残片存在于导入数据或当前数据中",
                ))
                scheme_valid = False

            if fb_id not in frag_id_map and fb_id not in existing_frag_ids:
                report.issues.append(ImportValidationIssue(
                    level="error",
                    code="MATCH_FRAG_NOT_FOUND",
                    message=f"匹配引用了不存在的残片 '{fb_id}'",
                    location=match_loc,
                    suggestion="确保引用的残片存在于导入数据或当前数据中",
                ))
                scheme_valid = False

        if scheme_valid:
            valid_schemes += 1

    report.valid_schemes = valid_schemes

    error_count = sum(1 for iss in report.issues if iss.level == "error")
    report.is_valid = error_count == 0

    if report.is_valid:
        report.summary = f"校验通过：{report.valid_fragments} 个残片, {report.valid_schemes} 个方案可导入"
    else:
        report.summary = f"校验发现 {error_count} 个错误，{len(report.issues) - error_count} 个警告"

    return report


def generate_analysis_package(scheme, all_schemes, fragments, operation_logs=None):
    package = {
        "package_info": {
            "generated_at": now_str(),
            "scheme_name": scheme.name,
            "scheme_id": scheme.id,
            "version": scheme.current_version,
        },
        "scheme": scheme.to_dict(),
        "fragments": [],
        "candidate_relations": [m.to_dict() for m in scheme.matches],
        "conflict_analysis": [],
        "review_summary": {
            "total": len(scheme.matches),
            "approved": len([m for m in scheme.matches if m.review_status == ReviewStatus.APPROVED]),
            "rejected": len([m for m in scheme.matches if m.review_status == ReviewStatus.REJECTED]),
            "pending": len([m for m in scheme.matches if m.review_status == ReviewStatus.PENDING]),
            "flagged": len([m for m in scheme.matches if m.review_status == ReviewStatus.FLAGGED]),
        },
        "operation_logs": [],
    }

    involved_frag_ids = scheme.get_involved_fragment_ids()
    for frag in fragments:
        if frag.id in involved_frag_ids:
            package["fragments"].append(frag.to_dict())

    conflicts = compute_scheme_conflicts(scheme, all_schemes, fragments)
    package["conflict_analysis"] = conflicts

    if operation_logs:
        scheme_logs = [
            log.to_dict() for log in operation_logs
            if log.target_id == scheme.id or log.details.get("scheme_id") == scheme.id
        ]
        package["operation_logs"] = scheme_logs[-50:]

    return package


def review_match(match, status, comment="", reviewer=""):
    match.review_status = status
    match.review_comment = comment
    match.reviewed_by = reviewer
    match.reviewed_at = now_str()
    return match


def get_locked_fragments_info(fragments, schemes):
    info = []
    for frag in fragments:
        if frag.is_locked:
            scheme = next((s for s in schemes if s.id == frag.locked_scheme_id), None)
            info.append({
                "fragment_id": frag.id,
                "locked_scheme_id": frag.locked_scheme_id,
                "locked_scheme_name": scheme.name if scheme else "未知",
            })
    return info


def filter_operation_logs(logs, op_type=None, target_type=None, target_id=None, operator=None, limit=100):
    filtered = logs
    if op_type:
        filtered = [l for l in filtered if l.operation_type == op_type]
    if target_type:
        filtered = [l for l in filtered if l.target_type == target_type]
    if target_id:
        filtered = [l for l in filtered if l.target_id == target_id]
    if operator:
        filtered = [l for l in filtered if operator in l.operator]
    return filtered[-limit:]
