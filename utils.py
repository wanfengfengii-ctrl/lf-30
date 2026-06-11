import datetime
import numpy as np
import networkx as nx
from models import (
    Fragment, EdgeFeature, EdgeMatch, Scheme, EdgeDirection, EdgeType, DefectType,
    ReviewStatus, OperationType, OperationLog, SchemeVersion, SchemeDiff,
    ImportValidationReport, ImportValidationIssue, ConflictInfo, ConflictType,
    TagCategory, SchemeOwnershipStatus, CustomTag, SemanticAnnotation,
    FragmentCluster, SearchResult, ResearchClue
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


def build_default_tags():
    defaults = []
    glyph_tags = [
        ("偏旁部首", ["亻", "氵", "扌", "艹", "木", "口", "日", "月", "土", "火"]),
        ("结构类型", ["左右结构", "上下结构", "包围结构", "独体字", "半包围"]),
        ("笔画类型", ["横", "竖", "撇", "捺", "点", "折", "钩"]),
    ]
    for group, tags in glyph_tags:
        for t in tags:
            defaults.append(CustomTag(name=f"{group}:{t}", category=TagCategory.GLYPH_COMPONENT,
                                      description=f"字形部件-{group}-{t}", color="#3498DB",
                                      created_at=now_str(), updated_at=now_str()))

    knife_tags = [
        "单刀直入", "双刀刻制", "浅刻", "深刻", "冲刀", "切刀", "涩刀", "游刀", "崩口", "毛刺"
    ]
    for t in knife_tags:
        defaults.append(CustomTag(name=t, category=TagCategory.KNIFE_MARK,
                                  description=f"刀痕形态-{t}", color="#E74C3C",
                                  created_at=now_str(), updated_at=now_str()))

    weathering_tags = [
        "自然风化", "水渍侵蚀", "土蚀", "磨损严重", "轻度风化",
        "裂纹密布", "表面剥落", "钙质沉积", "霉斑", "铁锈污染"
    ]
    for t in weathering_tags:
        defaults.append(CustomTag(name=t, category=TagCategory.WEATHERING,
                                  description=f"风化特征-{t}", color="#F39C12",
                                  created_at=now_str(), updated_at=now_str()))

    edge_texture_tags = [
        "毛边", "齐边", "锯齿边", "不规则断裂", "斜切边",
        "圆润边", "破损毛边", "刀切边", "自然断裂", "磨蚀边"
    ]
    for t in edge_texture_tags:
        defaults.append(CustomTag(name=t, category=TagCategory.EDGE_TEXTURE,
                                  description=f"边缘纹理-{t}", color="#2ECC71",
                                  created_at=now_str(), updated_at=now_str()))

    inscription_tags = [
        "楷书", "行书", "草书", "隶书", "篆书", "魏碑",
        "题记", "落款", "印章", "纪年", "人名", "地名", "官职"
    ]
    for t in inscription_tags:
        defaults.append(CustomTag(name=t, category=TagCategory.INSCRIPTION_TAG,
                                  description=f"题跋标签-{t}", color="#9B59B6",
                                  created_at=now_str(), updated_at=now_str()))

    research_tags = [
        "待考释", "重要文献", "版本差异", "待拼接", "已释读",
        "重复残片", "参考样本", "特殊形制", "需要专家鉴定", "已归档"
    ]
    for t in research_tags:
        defaults.append(CustomTag(name=t, category=TagCategory.RESEARCH_TAG,
                                  description=f"研究标签-{t}", color="#1ABC9C",
                                  created_at=now_str(), updated_at=now_str()))
    return defaults


def get_tags_by_category(tags, category):
    if isinstance(category, TagCategory):
        return [t for t in tags if t.category == category]
    return [t for t in tags if t.category.value == category]


def get_annotation_for_fragment(annotations, fragment_id):
    for ann in annotations:
        if ann.fragment_id == fragment_id:
            return ann
    return None


def get_fragment_ownership_status(fragment, schemes):
    if fragment.is_locked and fragment.locked_scheme_id:
        in_conflict = False
        for s in schemes:
            if s.id != fragment.locked_scheme_id and fragment.id in s.get_involved_fragment_ids():
                in_conflict = True
                break
        return SchemeOwnershipStatus.CONFLICTED if in_conflict else SchemeOwnershipStatus.LOCKED
    involved = [s for s in schemes if fragment.id in s.get_involved_fragment_ids()]
    if len(involved) > 1:
        return SchemeOwnershipStatus.CONFLICTED
    elif len(involved) == 1:
        return SchemeOwnershipStatus.CANDIDATE
    return SchemeOwnershipStatus.UNASSIGNED


def get_associated_schemes(fragment_id, schemes):
    return [s for s in schemes if fragment_id in s.get_involved_fragment_ids()]


def compute_texture_similarity(frag_a, frag_b):
    score = 0.0
    if frag_a.image_data and frag_b.image_data:
        desc_a = np.array(compute_descriptor_from_image(frag_a.image_data))
        desc_b = np.array(compute_descriptor_from_image(frag_b.image_data))
        if desc_a.size > 0 and desc_b.size > 0 and desc_a.shape == desc_b.shape:
            norm_a = desc_a / (np.linalg.norm(desc_a) + 1e-8)
            norm_b = desc_b / (np.linalg.norm(desc_b) + 1e-8)
            score = float(np.dot(norm_a, norm_b))
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def compute_annotation_similarity(ann_a, ann_b):
    if not ann_a or not ann_b:
        return 0.0
    score = 0.0
    field_weights = {
        "glyph_components": 0.20,
        "knife_marks": 0.15,
        "weathering_features": 0.15,
        "edge_textures": 0.15,
        "inscription_tags": 0.15,
        "research_tags": 0.10,
        "custom_tag_ids": 0.10,
    }
    for field, weight in field_weights.items():
        set_a = set(getattr(ann_a, field, []))
        set_b = set(getattr(ann_b, field, []))
        if set_a or set_b:
            intersection = set_a & set_b
            union = set_a | set_b
            jaccard = len(intersection) / len(union) if union else 0.0
            score += jaccard * weight
    return score


def compute_overall_similarity(frag_a, frag_b, ann_a, ann_b):
    tex_sim = compute_texture_similarity(frag_a, frag_b)
    ann_sim = compute_annotation_similarity(ann_a, ann_b)
    edge_sim = 0.0
    if frag_a.edges and frag_b.edges:
        sims = []
        for ea in frag_a.edges:
            for eb in frag_b.edges:
                sims.append(compute_edge_similarity(ea, eb))
        if sims:
            edge_sim = max(sims)
    overall = tex_sim * 0.35 + ann_sim * 0.40 + edge_sim * 0.25
    return overall, {"texture": tex_sim, "annotation": ann_sim, "edge": edge_sim}


def search_fragments(
    fragments, annotations, schemes, tags,
    fragment_id_keyword=None,
    inscription_keyword=None,
    research_keyword=None,
    selected_tag_ids=None,
    edge_direction=None,
    edge_type=None,
    defect_type=None,
    ownership_status=None,
    texture_ref_frag_id=None,
    texture_similarity_min=0.0,
    sort_by="relevance",
):
    results = []
    selected_tag_ids = selected_tag_ids or []

    for frag in fragments:
        score = 0.0
        matched_fields = []
        matched_tags = []

        if fragment_id_keyword and fragment_id_keyword.strip():
            kw = fragment_id_keyword.strip().lower()
            if kw in frag.id.lower():
                score += 0.30
                matched_fields.append("残片编号")
            else:
                continue

        ann = get_annotation_for_fragment(annotations, frag.id)

        if inscription_keyword and inscription_keyword.strip():
            kw = inscription_keyword.strip().lower()
            in_inscriptions = frag.inscriptions and kw in frag.inscriptions.lower()
            in_ann = ann and ann.inscription_content and kw in ann.inscription_content.lower()
            if in_inscriptions or in_ann:
                score += 0.25
                matched_fields.append("题跋内容")
            else:
                continue

        if research_keyword and research_keyword.strip():
            kw = research_keyword.strip().lower()
            in_notes = frag.notes and kw in frag.notes.lower()
            in_ann = ann and ann.research_notes and kw in ann.research_notes.lower()
            if in_notes or in_ann:
                score += 0.20
                matched_fields.append("研究备注")
            else:
                continue

        if selected_tag_ids:
            if ann:
                ann_tags = ann.get_all_tag_ids()
                matched = ann_tags & set(selected_tag_ids)
                if matched:
                    matched_tags = list(matched)
                    score += 0.15 * len(matched) / len(selected_tag_ids)
                    matched_fields.append("标签匹配")
                else:
                    continue
            else:
                continue

        edge_match = True
        if edge_direction or edge_type or defect_type:
            edge_match = False
            for e in frag.edges:
                d_ok = (edge_direction is None) or (e.direction == edge_direction)
                t_ok = (edge_type is None) or (e.edge_type == edge_type)
                def_ok = (defect_type is None) or (e.defect_type == defect_type)
                if d_ok and t_ok and def_ok:
                    edge_match = True
                    matched_fields.append("边缘特征")
                    break
            if not edge_match:
                continue

        frag_status = get_fragment_ownership_status(frag, schemes)
        if ownership_status:
            if isinstance(ownership_status, list):
                if frag_status not in ownership_status:
                    continue
            elif frag_status != ownership_status:
                continue

        if texture_ref_frag_id:
            ref_frag = next((f for f in fragments if f.id == texture_ref_frag_id), None)
            if ref_frag and ref_frag.id != frag.id:
                tex_sim = compute_texture_similarity(ref_frag, frag)
                if tex_sim >= texture_similarity_min:
                    score += tex_sim * 0.40
                    matched_fields.append(f"纹理相似({tex_sim:.2f})")
                else:
                    continue
            elif ref_frag and ref_frag.id == frag.id:
                pass

        associated_schemes = get_associated_schemes(frag.id, schemes)
        results.append(SearchResult(
            fragment_id=frag.id,
            relevance_score=score,
            matched_fields=matched_fields,
            matched_tags=matched_tags,
            ownership_status=frag_status,
            associated_scheme_ids=[s.id for s in associated_schemes],
            locked_scheme_id=frag.locked_scheme_id or "",
        ))

    if sort_by == "relevance":
        results.sort(key=lambda r: r.relevance_score, reverse=True)
    elif sort_by == "fragment_id":
        results.sort(key=lambda r: r.fragment_id)
    elif sort_by == "ownership":
        order = {
            SchemeOwnershipStatus.LOCKED: 0,
            SchemeOwnershipStatus.CANDIDATE: 1,
            SchemeOwnershipStatus.CONFLICTED: 2,
            SchemeOwnershipStatus.UNASSIGNED: 3,
        }
        results.sort(key=lambda r: order.get(r.ownership_status, 99))

    return results


def cluster_fragments(fragments, annotations, method="tag_based", n_clusters=3):
    clusters = []
    if not fragments:
        return clusters

    if method == "tag_based":
        frag_tags = {}
        for frag in fragments:
            ann = get_annotation_for_fragment(annotations, frag.id)
            if ann:
                frag_tags[frag.id] = ann.get_all_tag_ids()
            else:
                frag_tags[frag.id] = set()

        unassigned = set(f.id for f in fragments)
        cluster_id = 0
        while unassigned:
            seed_id = None
            max_tags = -1
            for fid in unassigned:
                if len(frag_tags[fid]) > max_tags:
                    max_tags = len(frag_tags[fid])
                    seed_id = fid
                    break
            if seed_id is None:
                seed_id = next(iter(unassigned))

            seed_tags = frag_tags[seed_id]
            cluster_members = {seed_id}
            unassigned.remove(seed_id)

            for fid in list(unassigned):
                f_tags = frag_tags[fid]
                if seed_tags and f_tags:
                    sim = len(seed_tags & f_tags) / max(len(seed_tags | f_tags), 1)
                    if sim >= 0.3:
                        cluster_members.add(fid)
                        unassigned.remove(fid)
                elif not seed_tags and not f_tags:
                    cluster_members.add(fid)
                    unassigned.remove(fid)

            cluster_id += 1
            clusters.append(FragmentCluster(
                name=f"标签聚类组-{cluster_id}",
                fragment_ids=list(cluster_members),
                cluster_method="tag_based",
                cluster_params={"threshold": 0.3},
                representative_fragment_id=seed_id,
                description=f"基于标签相似度聚类，核心残片 {seed_id}",
                created_at=now_str(),
            ))

    elif method == "texture_based":
        sim_matrix = {}
        frag_ids = [f.id for f in fragments]
        frag_map = {f.id: f for f in fragments}
        for i, fa in enumerate(fragments):
            for j, fb in enumerate(fragments):
                if i < j:
                    sim = compute_texture_similarity(fa, fb)
                    sim_matrix[(fa.id, fb.id)] = sim
                    sim_matrix[(fb.id, fa.id)] = sim

        unassigned = set(frag_ids)
        cluster_id = 0
        while unassigned:
            seed_id = next(iter(unassigned))
            cluster_members = {seed_id}
            unassigned.remove(seed_id)
            for fid in list(unassigned):
                sim = sim_matrix.get((seed_id, fid), 0.0)
                if sim >= 0.6:
                    cluster_members.add(fid)
                    unassigned.remove(fid)
            cluster_id += 1
            clusters.append(FragmentCluster(
                name=f"纹理聚类组-{cluster_id}",
                fragment_ids=list(cluster_members),
                cluster_method="texture_based",
                cluster_params={"threshold": 0.6},
                representative_fragment_id=seed_id,
                description=f"基于纹理特征相似度聚类，核心残片 {seed_id}",
                created_at=now_str(),
            ))

    elif method == "ownership_based":
        groups = {}
        for frag in fragments:
            status = get_fragment_ownership_status(frag, [])
            key = status.value
            if frag.locked_scheme_id:
                key = f"LOCKED:{frag.locked_scheme_id}"
            if key not in groups:
                groups[key] = []
            groups[key].append(frag.id)
        cluster_id = 0
        for key, members in groups.items():
            cluster_id += 1
            clusters.append(FragmentCluster(
                name=f"归属组-{key}",
                fragment_ids=members,
                cluster_method="ownership_based",
                cluster_params={"group_key": key},
                representative_fragment_id=members[0] if members else "",
                description=f"按归属状态分组: {key}",
                created_at=now_str(),
            ))

    return clusters


def find_similar_fragments(target_frag, fragments, annotations, top_k=5):
    results = []
    target_ann = get_annotation_for_fragment(annotations, target_frag.id)
    for frag in fragments:
        if frag.id == target_frag.id:
            continue
        ann = get_annotation_for_fragment(annotations, frag.id)
        overall, components = compute_overall_similarity(target_frag, frag, target_ann, ann)
        results.append({
            "fragment_id": frag.id,
            "similarity": overall,
            "components": components,
        })
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def discover_research_clues(fragments, annotations, schemes, matches):
    clues = []

    tag_groups = {}
    for frag in fragments:
        ann = get_annotation_for_fragment(annotations, frag.id)
        if ann:
            for tid in ann.glyph_components:
                if tid not in tag_groups:
                    tag_groups[tid] = []
                tag_groups[tid].append(frag.id)

    for tag_id, frag_ids in tag_groups.items():
        if len(frag_ids) >= 3:
            clues.append(ResearchClue(
                clue_type="同字形部件",
                title=f"字形部件关联组: {tag_id}",
                description=f"发现 {len(frag_ids)} 个残片具有相同字形部件 '{tag_id}'，可能属于同一组文字",
                fragment_ids=frag_ids,
                confidence=min(0.9, 0.5 + 0.05 * len(frag_ids)),
                suggestion="建议比对这些残片的文字位置关系，尝试拼接成完整文字",
                created_at=now_str(),
            ))

    for scheme in schemes:
        for other in schemes:
            if scheme.id >= other.id:
                continue
            s_frags = scheme.get_involved_fragment_ids()
            o_frags = other.get_involved_fragment_ids()
            overlap = s_frags & o_frags
            if overlap:
                clues.append(ResearchClue(
                    clue_type="方案交叉",
                    title=f"方案交叉重叠: {scheme.name} ↔ {other.name}",
                    description=f"方案 '{scheme.name}' 与 '{other.name}' 共享残片 {list(overlap)}，存在潜在归属冲突",
                    fragment_ids=list(overlap),
                    scheme_ids=[scheme.id, other.id],
                    confidence=0.85,
                    suggestion="建议审核两个方案的匹配关系，确认残片的正确归属",
                    created_at=now_str(),
                ))

    for frag in fragments:
        ann = get_annotation_for_fragment(annotations, frag.id)
        if ann and ann.inscription_content and not frag.is_locked:
            for other in fragments:
                if other.id == frag.id:
                    continue
                other_ann = get_annotation_for_fragment(annotations, other.id)
                if other_ann and other_ann.inscription_content:
                    words_a = set(ann.inscription_content)
                    words_b = set(other_ann.inscription_content)
                    common = words_a & words_b
                    if len(common) >= 3:
                        clues.append(ResearchClue(
                            clue_type="题跋关联",
                            title=f"题跋内容关联: {frag.id} ↔ {other.id}",
                            description=f"两个残片的题跋共享文字 '{''.join(common)}'，可能为同一段文字",
                            fragment_ids=[frag.id, other.id],
                            confidence=0.75,
                            suggestion="建议详细比对题跋内容，确认是否为连续文本",
                            created_at=now_str(),
                        ))
                        break

    for scheme in schemes:
        locked = [fid for fid in scheme.get_involved_fragment_ids()
                  if next((f for f in fragments if f.id == fid), None) and not next(f for f in fragments if f.id == fid).is_locked]
        if len(locked) < len(scheme.get_involved_fragment_ids()) * 0.5 and len(scheme.matches) >= 2:
            clues.append(ResearchClue(
                clue_type="待锁定方案",
                title=f"方案建议锁定: {scheme.name}",
                description=f"方案 '{scheme.name}' 包含 {len(scheme.matches)} 个匹配关系，但锁定率较低，建议确认后锁定",
                scheme_ids=[scheme.id],
                fragment_ids=list(scheme.get_involved_fragment_ids()),
                confidence=0.6,
                suggestion="审核方案匹配关系，确认无误后锁定方案以避免冲突",
                created_at=now_str(),
            ))

    unannotated = [f.id for f in fragments if not get_annotation_for_fragment(annotations, f.id)]
    if unannotated:
        clues.append(ResearchClue(
            clue_type="待标注",
            title=f"待语义标注残片 ({len(unannotated)} 个)",
            description=f"发现 {len(unannotated)} 个残片尚未进行语义标注，建议补全以增强检索效果",
            fragment_ids=unannotated,
            confidence=0.95,
            suggestion="为未标注残片添加字形部件、刀痕、风化等维度的语义标注",
            created_at=now_str(),
        ))

    clues.sort(key=lambda c: c.confidence, reverse=True)
    return clues


def build_tag_color_map(tags):
    return {t.id: t.color for t in tags}


def get_tag_name_by_id(tags, tag_id):
    for t in tags:
        if t.id == tag_id:
            return t.name
    return tag_id


def get_tag_category_name(tag_id, tags, annotations, fragment_id):
    ann = get_annotation_for_fragment(annotations, fragment_id)
    if not ann:
        return "未分类"
    if tag_id in ann.glyph_components:
        return TagCategory.GLYPH_COMPONENT.value
    if tag_id in ann.knife_marks:
        return TagCategory.KNIFE_MARK.value
    if tag_id in ann.weathering_features:
        return TagCategory.WEATHERING.value
    if tag_id in ann.edge_textures:
        return TagCategory.EDGE_TEXTURE.value
    if tag_id in ann.inscription_tags:
        return TagCategory.INSCRIPTION_TAG.value
    if tag_id in ann.research_tags:
        return TagCategory.RESEARCH_TAG.value
    if tag_id in ann.custom_tag_ids:
        for t in tags:
            if t.id == tag_id:
                return t.category.value
    return TagCategory.CUSTOM.value
