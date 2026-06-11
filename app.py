import json
import io
import copy
import datetime

import streamlit as st
import numpy as np
from PIL import Image
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import (
    Fragment, EdgeFeature, EdgeMatch, Scheme, EdgeDirection, EdgeType, DefectType,
    ReviewStatus, OperationType, OperationLog, SchemeVersion, SchemeDiff,
    ImportValidationReport, ImportValidationIssue, ConflictInfo, ConflictType
)
from utils import (
    validate_image,
    check_duplicate_fragment_id,
    compute_edge_similarity,
    find_candidate_matches,
    build_candidate_graph,
    compute_scheme_coverage,
    compute_scheme_conflicts,
    validate_edge_exclusivity,
    validate_locked_fragment,
    validate_delete_fragment,
    validate_export_scheme,
    validate_import_scheme,
    compute_descriptor_from_image,
    create_operation_log,
    now_str,
    create_scheme_version,
    restore_scheme_version,
    compare_schemes,
    validate_import_data,
    generate_analysis_package,
    review_match,
    get_locked_fragments_info,
    filter_operation_logs,
    detect_all_conflicts,
)


def init_session_state():
    if "fragments" not in st.session_state:
        st.session_state.fragments = []
    if "schemes" not in st.session_state:
        st.session_state.schemes = []
    if "candidate_matches" not in st.session_state:
        st.session_state.candidate_matches = []
    if "delete_confirm" not in st.session_state:
        st.session_state.delete_confirm = None
    if "operation_logs" not in st.session_state:
        st.session_state.operation_logs = []
    if "import_report" not in st.session_state:
        st.session_state.import_report = None
    if "import_pending_data" not in st.session_state:
        st.session_state.import_pending_data = None
    if "current_operator" not in st.session_state:
        st.session_state.current_operator = "研究员"


def add_log(op_type, target_type, target_id, description, details=None):
    log = create_operation_log(
        op_type, target_type, target_id, description,
        st.session_state.current_operator, details
    )
    st.session_state.operation_logs.append(log)


def render_fragment_image(frag, width=200):
    if frag.image_data:
        try:
            img = Image.open(io.BytesIO(frag.image_data))
            return img
        except Exception:
            return None
    return None


def get_review_status_color(status):
    if status == ReviewStatus.APPROVED:
        return "#2ECC71"
    elif status == ReviewStatus.REJECTED:
        return "#E74C3C"
    elif status == ReviewStatus.FLAGGED:
        return "#F39C12"
    else:
        return "#95A5A6"


def get_review_status_emoji(status):
    if status == ReviewStatus.APPROVED:
        return "✅"
    elif status == ReviewStatus.REJECTED:
        return "❌"
    elif status == ReviewStatus.FLAGGED:
        return "⚠️"
    else:
        return "⏳"


def page_fragments():
    st.header("残片管理")

    locked_info = get_locked_fragments_info(st.session_state.fragments, st.session_state.schemes)
    if locked_info:
        st.info(f"🔒 当前有 {len(locked_info)} 个残片已被锁定，归属不同方案")

    with st.expander("上传新残片", expanded=True):
        frag_id = st.text_input("残片编号", key="new_frag_id")
        uploaded_file = st.file_uploader("上传残片图像", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="frag_upload")
        inscriptions = st.text_area("题跋信息", key="new_frag_inscriptions")
        notes = st.text_area("备注", key="new_frag_notes")

        if st.button("添加残片", key="add_frag_btn"):
            if not frag_id.strip():
                st.error("残片编号不能为空")
            elif check_duplicate_fragment_id(st.session_state.fragments, frag_id.strip()):
                st.error(f"残片编号 '{frag_id.strip()}' 已存在，不能重复")
            elif not uploaded_file:
                st.error("请上传残片图像")
            else:
                img_bytes = uploaded_file.read()
                valid, msg = validate_image(img_bytes)
                if not valid:
                    st.error(f"图像无效，不能进入候选分析: {msg}")
                else:
                    img = Image.open(io.BytesIO(img_bytes))
                    descriptor = compute_descriptor_from_image(img_bytes)
                    ts = now_str()
                    frag = Fragment(
                        id=frag_id.strip(),
                        image_data=img_bytes,
                        image_format=uploaded_file.type or "image/png",
                        image_width=img.size[0],
                        image_height=img.size[1],
                        inscriptions=inscriptions.strip(),
                        notes=notes.strip(),
                        created_at=ts,
                        updated_at=ts,
                    )
                    st.session_state.fragments.append(frag)
                    add_log(OperationType.FRAGMENT_ADD, "fragment", frag.id,
                            f"添加残片 {frag.id} ({frag.image_width}x{frag.image_height})")
                    st.success(f"残片 '{frag.id}' 添加成功 ({frag.image_width}x{frag.image_height})")

    st.divider()
    st.subheader("已有残片列表")

    if not st.session_state.fragments:
        st.info("暂无残片数据，请先上传")
        return

    for frag in st.session_state.fragments:
        col_img, col_info, col_act = st.columns([2, 3, 1])
        with col_img:
            img = render_fragment_image(frag)
            if img:
                st.image(img, width=150)
            else:
                st.warning("无图像")

            if frag.is_locked:
                locked_scheme = next((s for s in st.session_state.schemes if s.id == frag.locked_scheme_id), None)
                scheme_name = locked_scheme.name if locked_scheme else frag.locked_scheme_id
                st.markdown(
                    f"<div style='background-color:#FFF3CD;padding:6px;border-radius:4px;"
                    f"border-left:4px solid #FFC107;font-size:12px;'>"
                    f"🔒 锁定于方案: <b>{scheme_name}</b></div>",
                    unsafe_allow_html=True,
                )

        with col_info:
            lock_status = f" 🔒锁定于方案 {frag.locked_scheme_id}" if frag.is_locked else " 未锁定"
            st.markdown(f"**编号:** {frag.id}  |  **尺寸:** {frag.image_width}x{frag.image_height}{lock_status}")
            if frag.inscriptions:
                st.markdown(f"**题跋:** {frag.inscriptions}")
            if frag.notes:
                st.markdown(f"**备注:** {frag.notes}")
            if frag.created_at:
                st.markdown(f"**创建时间:** {frag.created_at}")
            st.markdown(f"**边缘数量:** {len(frag.edges)}")
            for e in frag.edges:
                st.markdown(f"&nbsp;&nbsp;- {e.direction.value} | {e.edge_type.value} | {e.defect_type.value} | 长度{e.length_px}px")

        with col_act:
            if st.button("删除", key=f"del_{frag.id}"):
                involved = validate_delete_fragment(frag.id, st.session_state.schemes)
                if involved:
                    st.session_state.delete_confirm = frag.id
                    st.warning(f"残片 '{frag.id}' 参与了方案 {involved}，请二次确认删除")
                else:
                    st.session_state.fragments = [f for f in st.session_state.fragments if f.id != frag.id]
                    add_log(OperationType.FRAGMENT_DELETE, "fragment", frag.id,
                            f"删除残片 {frag.id}")
                    st.success(f"残片 '{frag.id}' 已删除")

    if st.session_state.delete_confirm:
        frag_id = st.session_state.delete_confirm
        st.warning(f"⚠️ 残片 '{frag_id}' 已参与方案，确认删除将移除相关方案中的匹配关系")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("确认删除", key="confirm_del"):
                st.session_state.fragments = [f for f in st.session_state.fragments if f.id != frag_id]
                for scheme in st.session_state.schemes:
                    scheme.matches = [
                        m for m in scheme.matches
                        if m.edge_a_fragment_id != frag_id and m.edge_b_fragment_id != frag_id
                    ]
                st.session_state.delete_confirm = None
                add_log(OperationType.FRAGMENT_DELETE, "fragment", frag_id,
                        f"删除残片 {frag_id} 及其在所有方案中的匹配关系")
                st.success(f"残片 '{frag_id}' 及相关匹配已删除")
        with c2:
            if st.button("取消删除", key="cancel_del"):
                st.session_state.delete_confirm = None


def page_edges():
    st.header("边缘特征标注")

    if not st.session_state.fragments:
        st.info("暂无残片，请先在「残片管理」中上传")
        return

    frag_options = {f.id: f for f in st.session_state.fragments}
    selected_frag_id = st.selectbox("选择残片", options=list(frag_options.keys()), key="edge_frag_select")

    if selected_frag_id:
        frag = frag_options[selected_frag_id]
        img = render_fragment_image(frag)
        if img:
            st.image(img, width=250, caption=f"残片 {frag.id}")

        st.subheader(f"残片 {frag.id} 的边缘列表")
        if frag.edges:
            for e in frag.edges:
                st.markdown(f"- **{e.id}**: 方向={e.direction.value}, 类型={e.edge_type.value}, 缺损={e.defect_type.value}, 长度={e.length_px}px")

        with st.expander("添加边缘特征", expanded=True):
            direction = st.selectbox("边缘方向", [d.value for d in EdgeDirection], key="edge_dir")
            edge_type = st.selectbox("边缘类型", [t.value for t in EdgeType], key="edge_type")
            defect_type = st.selectbox("缺损类型", [d.value for d in DefectType], key="edge_defect")
            length_px = st.number_input("长度 (像素)", min_value=0, max_value=10000, value=100, key="edge_len")
            curvature = st.number_input("曲率", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="edge_curv")
            roughness = st.number_input("粗糙度", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="edge_rough")

            if st.button("添加边缘", key="add_edge_btn"):
                new_edge = EdgeFeature(
                    fragment_id=selected_frag_id,
                    direction=EdgeDirection(direction),
                    edge_type=EdgeType(edge_type),
                    defect_type=DefectType(defect_type),
                    length_px=int(length_px),
                    curvature=float(curvature),
                    roughness=float(roughness),
                    descriptor=compute_descriptor_from_image(frag.image_data) if frag.image_data else [],
                )
                frag.edges.append(new_edge)
                frag.updated_at = now_str()
                add_log(OperationType.EDGE_ADD, "edge", new_edge.id,
                        f"为残片 {frag.id} 添加边缘 {new_edge.id} ({direction}, {length_px}px)")
                st.success(f"边缘 {new_edge.id} 已添加到残片 {frag.id}")

        if frag.edges:
            with st.expander("删除边缘"):
                edge_to_del = st.selectbox("选择要删除的边缘", options=[e.id for e in frag.edges], key="edge_del_select")
                if st.button("删除边缘", key="del_edge_btn"):
                    frag.edges = [e for e in frag.edges if e.id != edge_to_del]
                    frag.updated_at = now_str()
                    add_log(OperationType.EDGE_DELETE, "edge", edge_to_del,
                            f"从残片 {frag.id} 删除边缘 {edge_to_del}")
                    st.success(f"边缘 {edge_to_del} 已删除")


def page_analysis():
    st.header("候选分析与人工审核")

    if len(st.session_state.fragments) < 2:
        st.info("至少需要2个残片才能进行候选分析")
        return

    threshold = st.slider("相似度阈值", min_value=0.0, max_value=1.0, value=0.40, step=0.05, key="sim_threshold")

    if st.button("执行候选分析", key="run_analysis"):
        valid_frags = [f for f in st.session_state.fragments if f.edges]
        if len(valid_frags) < 2:
            st.warning("至少需要2个带有边缘标注的残片")
            return
        st.session_state.candidate_matches = find_candidate_matches(valid_frags, threshold)
        add_log(OperationType.ANALYSIS_RUN, "analysis", "candidate",
                f"执行候选分析，阈值={threshold:.2f}，找到 {len(st.session_state.candidate_matches)} 对匹配",
                {"threshold": threshold, "match_count": len(st.session_state.candidate_matches)})
        if st.session_state.candidate_matches:
            st.success(f"找到 {len(st.session_state.candidate_matches)} 对候选匹配")
        else:
            st.warning("未找到满足阈值的候选匹配")

    if not st.session_state.candidate_matches:
        return

    st.subheader("候选匹配列表与人工审核")

    review_tab, graph_tab = st.tabs(["候选匹配与审核", "拼接候选图谱"])

    with review_tab:
        status_filter = st.selectbox(
            "按审核状态筛选",
            ["全部", "待审核", "已通过", "已拒绝", "存疑"],
            key="review_filter"
        )

        display_matches = st.session_state.candidate_matches
        if status_filter != "全部":
            status_map = {
                "待审核": ReviewStatus.PENDING,
                "已通过": ReviewStatus.APPROVED,
                "已拒绝": ReviewStatus.REJECTED,
                "存疑": ReviewStatus.FLAGGED,
            }
            display_matches = [m for m in display_matches if m.review_status == status_map[status_filter]]

        st.info(f"共 {len(display_matches)} 条匹配 (筛选后)")

        for i, m in enumerate(display_matches):
            status_color = get_review_status_color(m.review_status)
            status_emoji = get_review_status_emoji(m.review_status)

            with st.container():
                col_info, col_actions = st.columns([3, 2])

                with col_info:
                    st.markdown(
                        f"{status_emoji} **{m.edge_a_fragment_id}** 的边缘 `{m.edge_a_id}` ↔ "
                        f"**{m.edge_b_fragment_id}** 的边缘 `{m.edge_b_id}` — "
                        f"相似度 <span style='color:{status_color};font-weight:bold;'>{m.similarity:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    if m.review_comment:
                        st.caption(f"审核意见: {m.review_comment}")
                    if m.reviewed_at:
                        st.caption(f"审核时间: {m.reviewed_at} | 审核人: {m.reviewed_by}")

                with col_actions:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("✅ 通过", key=f"approve_{i}_{m.edge_a_id}_{m.edge_b_id}"):
                            review_match(m, ReviewStatus.APPROVED, "", st.session_state.current_operator)
                            add_log(OperationType.REVIEW_APPROVE, "match", f"{m.edge_a_id}_{m.edge_b_id}",
                                    f"审核通过匹配: {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}]")
                            st.rerun()
                    with c2:
                        if st.button("❌ 拒绝", key=f"reject_{i}_{m.edge_a_id}_{m.edge_b_id}"):
                            review_match(m, ReviewStatus.REJECTED, "", st.session_state.current_operator)
                            add_log(OperationType.REVIEW_REJECT, "match", f"{m.edge_a_id}_{m.edge_b_id}",
                                    f"审核拒绝匹配: {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}]")
                            st.rerun()
                    with c3:
                        if st.button("⚠️ 存疑", key=f"flag_{i}_{m.edge_a_id}_{m.edge_b_id}"):
                            review_match(m, ReviewStatus.FLAGGED, "", st.session_state.current_operator)
                            add_log(OperationType.REVIEW_FLAG, "match", f"{m.edge_a_id}_{m.edge_b_id}",
                                    f"审核存疑匹配: {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}]")
                            st.rerun()

                    comment = st.text_input("审核意见 (可选)", key=f"comment_{i}_{m.edge_a_id}_{m.edge_b_id}",
                                            label_visibility="collapsed", placeholder="填写审核意见...")
                    if st.button("提交意见", key=f"submit_comment_{i}_{m.edge_a_id}_{m.edge_b_id}"):
                        m.review_comment = comment
                        m.reviewed_by = st.session_state.current_operator
                        m.reviewed_at = now_str()
                        st.success("意见已提交")

                st.divider()

    with graph_tab:
        st.subheader("拼接候选图谱")
        G = build_candidate_graph(st.session_state.fragments, st.session_state.candidate_matches)

        fig, ax = plt.subplots(figsize=(10, 8))
        if len(G.nodes) > 0:
            pos = nx.spring_layout(G, seed=42)
            edge_colors = []
            for u, v in G.edges:
                rs = G[u][v].get("review_status", "待审核")
                if rs == "已通过":
                    edge_colors.append("#2ECC71")
                elif rs == "已拒绝":
                    edge_colors.append("#E74C3C")
                elif rs == "存疑":
                    edge_colors.append("#F39C12")
                else:
                    edge_colors.append("#95A5A6")

            edge_widths = [G[u][v].get("similarity", 0.5) * 5 for u, v in G.edges]
            nx.draw(
                G, pos, ax=ax, with_labels=True, node_color="#4ECDC4",
                node_size=1200, font_size=10, font_weight="bold",
                width=edge_widths, edge_color=edge_colors, alpha=0.85,
            )
            edge_labels = {(u, v): f"{G[u][v]['similarity']:.2f}" for u, v in G.edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color="#2ECC71", lw=2, label="已通过"),
                Line2D([0], [0], color="#95A5A6", lw=2, label="待审核"),
                Line2D([0], [0], color="#F39C12", lw=2, label="存疑"),
                Line2D([0], [0], color="#E74C3C", lw=2, label="已拒绝"),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

        ax.set_title("拼接候选图谱（颜色表示审核状态）")
        st.pyplot(fig)
        plt.close(fig)


def page_schemes():
    st.header("方案管理与版本控制")

    with st.expander("创建新方案", expanded=True):
        scheme_name = st.text_input("方案名称", key="new_scheme_name")
        scheme_desc = st.text_area("方案描述", key="new_scheme_desc", height=60)
        available_matches = [m for m in st.session_state.candidate_matches
                             if m.review_status in (ReviewStatus.APPROVED, ReviewStatus.PENDING, ReviewStatus.FLAGGED)]

        if not available_matches:
            st.info("暂无可用的候选匹配，请先在「候选分析」中运行分析")
        else:
            match_labels = [
                f"{get_review_status_emoji(m.review_status)} {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}] (相似度={m.similarity:.3f})"
                for m in available_matches
            ]
            selected_indices = st.multiselect(
                "选择候选匹配（可多选）",
                options=list(range(len(match_labels))),
                format_func=lambda i: match_labels[i],
                key="scheme_match_select"
            )

            if st.button("创建方案", key="create_scheme_btn"):
                if not scheme_name.strip():
                    st.error("方案名称不能为空")
                elif not selected_indices:
                    st.error("请至少选择一个候选匹配")
                else:
                    selected_matches = [available_matches[i] for i in selected_indices]
                    ts = now_str()
                    new_scheme = Scheme(
                        name=scheme_name.strip(),
                        matches=selected_matches,
                        description=scheme_desc.strip(),
                        created_at=ts,
                        updated_at=ts,
                        created_by=st.session_state.current_operator,
                    )

                    ok, msg = validate_edge_exclusivity(new_scheme)
                    if not ok:
                        st.error(f"方案创建失败: {msg}")
                    else:
                        all_frag_ids = new_scheme.get_involved_fragment_ids()
                        locked_error = None
                        for fid in all_frag_ids:
                            ok2, msg2 = validate_locked_fragment(fid, st.session_state.fragments, new_scheme.id)
                            if not ok2:
                                locked_error = msg2
                                break
                        if locked_error:
                            st.error(f"方案创建失败: {locked_error}")
                        else:
                            new_scheme.create_version("初始版本", st.session_state.current_operator)
                            st.session_state.schemes.append(new_scheme)
                            add_log(OperationType.SCHEME_CREATE, "scheme", new_scheme.id,
                                    f"创建方案 '{new_scheme.name}'，包含 {len(selected_matches)} 个匹配",
                                    {"match_count": len(selected_matches)})
                            st.success(f"方案 '{new_scheme.name}' 创建成功 (ID: {new_scheme.id})")

    st.divider()

    all_conflicts = detect_all_conflicts(st.session_state.schemes, st.session_state.fragments)
    if all_conflicts:
        high_count = sum(1 for c in all_conflicts if c.severity == "high")
        med_count = sum(1 for c in all_conflicts if c.severity == "medium")
        st.warning(
            f"⚠️ 检测到 {len(all_conflicts)} 个冲突 "
            f"(高危: {high_count}, 中危: {med_count})，请在方案详情中查看"
        )

    st.subheader("已有方案")

    if not st.session_state.schemes:
        st.info("暂无方案")
        return

    for scheme in st.session_state.schemes:
        with st.expander(f"📋 {scheme.name} (v{scheme.current_version})", expanded=False):
            coverage, matched, total = compute_scheme_coverage(scheme, st.session_state.fragments)
            conflicts = compute_scheme_conflicts(scheme, st.session_state.schemes, st.session_state.fragments)

            lock_icon = " 🔒" if scheme.is_locked else ""
            st.markdown(f"### 方案: {scheme.name}{lock_icon}")
            st.markdown(f"**ID:** {scheme.id}  |  **版本:** v{scheme.current_version}  |  **历史版本数:** {len(scheme.versions)}")
            if scheme.description:
                st.markdown(f"**描述:** {scheme.description}")
            st.markdown(f"**覆盖率:** {coverage:.1%} ({matched}/{total} 边缘)")
            st.markdown(f"**匹配数:** {len(scheme.matches)}")
            st.markdown(f"**涉及残片:** {', '.join(scheme.get_involved_fragment_ids())}")
            if scheme.created_at:
                st.markdown(f"**创建时间:** {scheme.created_at}")
            if scheme.updated_at:
                st.markdown(f"**更新时间:** {scheme.updated_at}")

            review_summary = {
                "已通过": len([m for m in scheme.matches if m.review_status == ReviewStatus.APPROVED]),
                "待审核": len([m for m in scheme.matches if m.review_status == ReviewStatus.PENDING]),
                "存疑": len([m for m in scheme.matches if m.review_status == ReviewStatus.FLAGGED]),
                "已拒绝": len([m for m in scheme.matches if m.review_status == ReviewStatus.REJECTED]),
            }
            st.markdown(
                f"**审核状态:** ✅ {review_summary['已通过']} | ⏳ {review_summary['待审核']} | "
                f"⚠️ {review_summary['存疑']} | ❌ {review_summary['已拒绝']}"
            )

            with st.expander("查看匹配详情", expanded=False):
                for m in scheme.matches:
                    status_emoji = get_review_status_emoji(m.review_status)
                    st.markdown(
                        f"&nbsp;&nbsp;{status_emoji} {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ "
                        f"{m.edge_b_fragment_id}[{m.edge_b_id}] (相似度={m.similarity:.3f})"
                    )

            if conflicts:
                st.warning(f"**冲突点 ({len(conflicts)}):**")
                for c in conflicts:
                    st.markdown(f"&nbsp;&nbsp;⚠️ [{c['type']}] {c['detail']}")
            else:
                st.success("无冲突")

            st.divider()

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if not scheme.is_locked:
                    if st.button("🔒 锁定", key=f"lock_scheme_{scheme.id}"):
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag:
                                frag.is_locked = True
                                frag.locked_scheme_id = scheme.id
                        scheme.is_locked = True
                        add_log(OperationType.SCHEME_LOCK, "scheme", scheme.id,
                                f"锁定方案 '{scheme.name}'")
                        st.success(f"方案 '{scheme.name}' 已锁定，涉及残片已被占用")
                        st.rerun()
                else:
                    if st.button("🔓 解锁", key=f"unlock_scheme_{scheme.id}"):
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag and frag.locked_scheme_id == scheme.id:
                                frag.is_locked = False
                                frag.locked_scheme_id = None
                        scheme.is_locked = False
                        add_log(OperationType.SCHEME_UNLOCK, "scheme", scheme.id,
                                f"解锁方案 '{scheme.name}'")
                        st.success(f"方案 '{scheme.name}' 已解锁")
                        st.rerun()

            with c2:
                if st.button("📊 导出分析包", key=f"export_pkg_{scheme.id}"):
                    pkg = generate_analysis_package(scheme, st.session_state.fragments, st.session_state.operation_logs)
                    pkg_json = json.dumps(pkg, ensure_ascii=False, indent=2)
                    st.download_button(
                        "⬇️ 下载完整分析包",
                        data=pkg_json,
                        file_name=f"analysis_package_{scheme.id}_v{scheme.current_version}.json",
                        mime="application/json",
                        key=f"dl_pkg_{scheme.id}",
                    )
                    add_log(OperationType.DATA_EXPORT, "scheme", scheme.id,
                            f"导出方案 '{scheme.name}' 的完整分析包 (v{scheme.current_version})")

            with c3:
                if st.button("📝 新建版本", key=f"new_ver_{scheme.id}"):
                    st.session_state[f"new_ver_desc_{scheme.id}"] = True

            with c4:
                if st.button("🗑️ 删除", key=f"del_scheme_{scheme.id}"):
                    if scheme.is_locked:
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag and frag.locked_scheme_id == scheme.id:
                                frag.is_locked = False
                                frag.locked_scheme_id = None
                    st.session_state.schemes = [s for s in st.session_state.schemes if s.id != scheme.id]
                    add_log(OperationType.SCHEME_DELETE, "scheme", scheme.id,
                            f"删除方案 '{scheme.name}'")
                    st.success(f"方案 '{scheme.name}' 已删除")
                    st.rerun()

            if st.session_state.get(f"new_ver_desc_{scheme.id}", False):
                ver_desc = st.text_input("版本描述", key=f"ver_desc_input_{scheme.id}")
                if st.button("确认创建版本", key=f"confirm_ver_{scheme.id}"):
                    scheme.create_version(ver_desc, st.session_state.current_operator)
                    scheme.updated_at = now_str()
                    st.session_state[f"new_ver_desc_{scheme.id}"] = False
                    add_log(OperationType.SCHEME_VERSION_CREATE, "scheme", scheme.id,
                            f"为方案 '{scheme.name}' 创建版本 v{scheme.current_version}",
                            {"version": scheme.current_version, "description": ver_desc})
                    st.success(f"版本 v{scheme.current_version} 创建成功")
                    st.rerun()

            with st.expander("版本历史", expanded=False):
                if scheme.versions:
                    for v in reversed(scheme.versions):
                        col_ver, col_info, col_act = st.columns([1, 3, 1])
                        with col_ver:
                            st.markdown(f"**v{v.version_number}**")
                        with col_info:
                            st.markdown(f"**描述:** {v.description or '(无)'}")
                            st.caption(f"创建时间: {v.created_at} | 创建人: {v.created_by}")
                            st.caption(f"匹配数: {len(v.matches)} | 锁定: {'是' if v.is_locked else '否'}")
                        with col_act:
                            if v.version_number != scheme.current_version:
                                if st.button("恢复", key=f"restore_{scheme.id}_{v.version_number}"):
                                    if scheme.is_locked:
                                        st.error("方案已锁定，不能恢复版本")
                                    else:
                                        scheme.restore_version(v.version_number)
                                        scheme.updated_at = now_str()
                                        add_log(OperationType.SCHEME_VERSION_RESTORE, "scheme", scheme.id,
                                                f"将方案 '{scheme.name}' 恢复到版本 v{v.version_number}",
                                                {"target_version": v.version_number})
                                        st.success(f"已恢复到版本 v{v.version_number}")
                                        st.rerun()
                            else:
                                st.caption("(当前版本)")
                else:
                    st.caption("暂无历史版本")


def page_compare():
    st.header("方案差异对比")

    if len(st.session_state.schemes) < 2:
        st.info("至少需要2个方案才能进行对比")
        return

    scheme_map = {s.name: s for s in st.session_state.schemes}
    scheme_names = list(scheme_map.keys())

    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox("方案 A", scheme_names, key="cmp_scheme_a")
    with col_b:
        name_b = st.selectbox("方案 B", [n for n in scheme_names if n != name_a] or scheme_names, key="cmp_scheme_b")

    scheme_a = scheme_map[name_a]
    scheme_b = scheme_map[name_b]

    if st.button("执行对比", key="run_compare"):
        diff = compare_schemes(scheme_a, scheme_b, st.session_state.fragments)

        st.divider()
        st.subheader("对比结果")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("相似度", f"{diff.similarity_score:.1%}")
        with col2:
            st.metric("覆盖率差异", f"{diff.coverage_diff:+.1%}")
        with col3:
            st.metric("匹配数变化", f"{len(diff.added_matches) - len(diff.removed_matches):+d}")

        st.divider()

        tab_added, tab_removed, tab_common = st.tabs([
            f"新增匹配 ({len(diff.added_matches)})",
            f"移除匹配 ({len(diff.removed_matches)})",
            f"共有匹配 ({len(diff.common_matches)})",
        ])

        with tab_added:
            if diff.added_matches:
                for m in diff.added_matches:
                    status_emoji = get_review_status_emoji(m.review_status)
                    st.markdown(
                        f"{status_emoji} **{m.edge_a_fragment_id}**[{m.edge_a_id}] ↔ "
                        f"**{m.edge_b_fragment_id}**[{m.edge_b_id}] — 相似度 {m.similarity:.3f}"
                    )
            else:
                st.info("无新增匹配")

        with tab_removed:
            if diff.removed_matches:
                for m in diff.removed_matches:
                    status_emoji = get_review_status_emoji(m.review_status)
                    st.markdown(
                        f"{status_emoji} **{m.edge_a_fragment_id}**[{m.edge_a_id}] ↔ "
                        f"**{m.edge_b_fragment_id}**[{m.edge_b_id}] — 相似度 {m.similarity:.3f}"
                    )
            else:
                st.info("无移除匹配")

        with tab_common:
            if diff.common_matches:
                for m in diff.common_matches:
                    status_emoji = get_review_status_emoji(m.review_status)
                    st.markdown(
                        f"{status_emoji} **{m.edge_a_fragment_id}**[{m.edge_a_id}] ↔ "
                        f"**{m.edge_b_fragment_id}**[{m.edge_b_id}] — 相似度 {m.similarity:.3f}"
                    )
            else:
                st.info("无共有匹配")

        st.divider()
        st.subheader("覆盖率对比图")

        cov_a, matched_a, total_a = compute_scheme_coverage(scheme_a, st.session_state.fragments)
        cov_b, matched_b, total_b = compute_scheme_coverage(scheme_b, st.session_state.fragments)

        fig, ax = plt.subplots(figsize=(8, 4))
        x = [scheme_a.name, scheme_b.name]
        y = [cov_a * 100, cov_b * 100]
        bars = ax.bar(x, y, color=["#4ECDC4", "#FF6B6B"], alpha=0.8)
        ax.set_ylabel("覆盖率 (%)")
        ax.set_ylim(0, 100)
        ax.set_title("方案覆盖率对比")

        for bar, val in zip(bars, y):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f"{val:.1f}%", ha='center', va='bottom')

        st.pyplot(fig)
        plt.close(fig)


def page_logs():
    st.header("操作日志追踪")

    col1, col2, col3 = st.columns(3)
    with col1:
        op_type_filter = st.selectbox(
            "操作类型",
            ["全部"] + [ot.value for ot in OperationType],
            key="log_op_type"
        )
    with col2:
        target_type_filter = st.selectbox(
            "目标类型",
            ["全部", "fragment", "edge", "scheme", "analysis", "match"],
            key="log_target_type"
        )
    with col3:
        operator_filter = st.text_input("操作人", key="log_operator")

    log_limit = st.slider("显示条数", min_value=10, max_value=500, value=100, step=10, key="log_limit")

    op_type = None
    if op_type_filter != "全部":
        for ot in OperationType:
            if ot.value == op_type_filter:
                op_type = ot
                break

    target_type = None if target_type_filter == "全部" else target_type_filter
    operator = None if not operator_filter.strip() else operator_filter.strip()

    filtered_logs = filter_operation_logs(
        st.session_state.operation_logs,
        op_type=op_type,
        target_type=target_type,
        operator=operator,
        limit=log_limit,
    )

    st.info(f"共 {len(filtered_logs)} 条记录 (筛选后) / 总计 {len(st.session_state.operation_logs)} 条")

    if not filtered_logs:
        st.info("暂无操作日志")
        return

    for log in reversed(filtered_logs):
        with st.container():
            col_time, col_op, col_target, col_desc = st.columns([2, 2, 2, 4])
            with col_time:
                st.caption(log.timestamp)
            with col_op:
                st.markdown(f"**{log.operation_type.value}**")
            with col_target:
                st.caption(f"{log.target_type}: {log.target_id[:12]}...")
            with col_desc:
                st.markdown(log.description)
            if log.operator:
                st.caption(f"操作人: {log.operator}")
            st.divider()

    if st.button("导出操作日志", key="export_logs_btn"):
        logs_data = [l.to_dict() for l in st.session_state.operation_logs]
        logs_json = json.dumps(logs_data, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ 下载日志JSON",
            data=logs_json,
            file_name=f"operation_logs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key="dl_logs_btn",
        )


def page_import_export():
    st.header("导入 / 导出")

    tab_export, tab_import = st.tabs(["导出", "导入"])

    with tab_export:
        st.subheader("导出全部数据")
        export = {
            "fragments": [f.to_dict() for f in st.session_state.fragments],
            "schemes": [s.to_dict() for s in st.session_state.schemes],
            "operation_logs": [l.to_dict() for l in st.session_state.operation_logs],
            "exported_at": now_str(),
            "exported_by": st.session_state.current_operator,
        }
        json_str = json.dumps(export, ensure_ascii=False, indent=2)
        st.download_button(
            "📦 导出当前所有残片、方案与日志",
            data=json_str,
            file_name="rubbing_analysis_full.json",
            mime="application/json",
            key="export_all_btn",
        )
        st.caption(f"当前包含 {len(st.session_state.fragments)} 个残片, {len(st.session_state.schemes)} 个方案, {len(st.session_state.operation_logs)} 条操作日志")

    with tab_import:
        st.subheader("导入完整数据")
        uploaded_all = st.file_uploader("上传完整数据JSON文件", type=["json"], key="import_all_upload")

        if uploaded_all is not None:
            try:
                content = uploaded_all.read().decode("utf-8")
                data = json.loads(content)

                if "fragments" not in data or "schemes" not in data:
                    st.error("JSON缺少 fragments 或 schemes 字段")
                else:
                    report = validate_import_data(data, st.session_state.fragments, st.session_state.schemes)
                    st.session_state.import_report = report
                    st.session_state.import_pending_data = data

                    st.subheader("导入校验报告")

                    if report.is_valid:
                        st.success(f"✅ {report.summary}")
                    else:
                        st.error(f"❌ {report.summary}")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("残片总数", report.total_fragments)
                    with col2:
                        st.metric("有效残片", report.valid_fragments)
                    with col3:
                        st.metric("方案总数", report.total_schemes)
                    with col4:
                        st.metric("有效方案", report.valid_schemes)

                    if report.issues:
                        st.subheader("问题详情")
                        errors = [i for i in report.issues if i.level == "error"]
                        warnings = [i for i in report.issues if i.level == "warning"]

                        if errors:
                            st.error(f"错误 ({len(errors)} 项)")
                            for iss in errors:
                                with st.expander(f"❌ [{iss.code}] {iss.message}"):
                                    st.markdown(f"**位置:** `{iss.location}`")
                                    st.markdown(f"**建议:** {iss.suggestion}")

                        if warnings:
                            st.warning(f"警告 ({len(warnings)} 项)")
                            for iss in warnings:
                                with st.expander(f"⚠️ [{iss.code}] {iss.message}"):
                                    st.markdown(f"**位置:** `{iss.location}`")
                                    st.markdown(f"**建议:** {iss.suggestion}")

                    if report.is_valid:
                        if st.button("确认导入", key="confirm_import_btn", type="primary"):
                            new_fragments = [Fragment.from_dict(d) for d in data["fragments"]]
                            new_schemes = [Scheme.from_dict(d) for d in data["schemes"]]
                            new_logs = []
                            if "operation_logs" in data:
                                new_logs = [OperationLog.from_dict(l) for l in data["operation_logs"]]

                            st.session_state.fragments.extend(new_fragments)
                            st.session_state.schemes.extend(new_schemes)
                            st.session_state.operation_logs.extend(new_logs)

                            add_log(OperationType.DATA_IMPORT, "data", "import_all",
                                    f"导入 {len(new_fragments)} 个残片, {len(new_schemes)} 个方案, {len(new_logs)} 条日志",
                                    {"fragment_count": len(new_fragments),
                                     "scheme_count": len(new_schemes),
                                     "log_count": len(new_logs)})

                            st.session_state.import_report = None
                            st.session_state.import_pending_data = None
                            st.success(f"导入成功: {len(new_fragments)} 个残片, {len(new_schemes)} 个方案")
                    else:
                        st.info("请修正数据后重新导入，或选择仅导入有效部分")

            except json.JSONDecodeError:
                st.error("无效的JSON文件")
            except Exception as e:
                st.error(f"导入异常: {str(e)}（不会覆盖当前分析结果）")

        st.divider()
        st.subheader("导入单个方案")
        uploaded_scheme = st.file_uploader("上传方案JSON文件", type=["json"], key="import_scheme_upload")

        if uploaded_scheme is not None:
            try:
                content = uploaded_scheme.read().decode("utf-8")
                scheme_dict = json.loads(content)

                is_single = "matches" in scheme_dict and "name" in scheme_dict
                if is_single:
                    scheme_obj, err = validate_import_scheme(scheme_dict, st.session_state.schemes, st.session_state.fragments)
                    if err:
                        st.error(f"导入失败: {err}（不会覆盖当前分析结果）")
                    else:
                        st.session_state.schemes.append(scheme_obj)
                        add_log(OperationType.DATA_IMPORT, "scheme", scheme_obj.id,
                                f"导入方案 '{scheme_obj.name}'")
                        st.success(f"方案 '{scheme_obj.name}' 导入成功")
                else:
                    st.error("JSON格式不正确，需要包含 name 和 matches 字段")
            except json.JSONDecodeError:
                st.error("无效的JSON文件")
            except Exception as e:
                st.error(f"导入异常: {str(e)}（不会覆盖当前分析结果）")


def main():
    init_session_state()

    st.set_page_config(page_title="多方案可追溯拼接分析平台", layout="wide", page_icon="🪨")
    st.title("🪨 多方案可追溯拼接分析平台")

    with st.sidebar:
        st.subheader("研究员信息")
        st.session_state.current_operator = st.text_input("操作人", value=st.session_state.current_operator)
        st.divider()
        st.caption("碑刻拓片残片拼接分析系统 v2.0")
        st.caption("支持方案版本管理、人工审核、冲突预警、操作追踪")

    tab_fragments, tab_edges, tab_analysis, tab_schemes, tab_compare, tab_logs, tab_io = st.tabs([
        "残片管理", "边缘标注", "候选分析", "方案管理", "方案对比", "操作日志", "导入/导出",
    ])

    with tab_fragments:
        page_fragments()
    with tab_edges:
        page_edges()
    with tab_analysis:
        page_analysis()
    with tab_schemes:
        page_schemes()
    with tab_compare:
        page_compare()
    with tab_logs:
        page_logs()
    with tab_io:
        page_import_export()


if __name__ == "__main__":
    main()
