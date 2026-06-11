import json
import io
import copy

import streamlit as st
import numpy as np
from PIL import Image
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import Fragment, EdgeFeature, EdgeMatch, Scheme, EdgeDirection, EdgeType, DefectType
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


def render_fragment_image(frag, width=200):
    if frag.image_data:
        try:
            img = Image.open(io.BytesIO(frag.image_data))
            return img
        except Exception:
            return None
    return None


def page_fragments():
    st.header("残片管理")

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
                    frag = Fragment(
                        id=frag_id.strip(),
                        image_data=img_bytes,
                        image_format=uploaded_file.type or "image/png",
                        image_width=img.size[0],
                        image_height=img.size[1],
                        inscriptions=inscriptions.strip(),
                        notes=notes.strip(),
                    )
                    st.session_state.fragments.append(frag)
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
        with col_info:
            lock_status = f" 🔒锁定于方案 {frag.locked_scheme_id}" if frag.is_locked else " 未锁定"
            st.markdown(f"**编号:** {frag.id}  |  **尺寸:** {frag.image_width}x{frag.image_height}{lock_status}")
            if frag.inscriptions:
                st.markdown(f"**题跋:** {frag.inscriptions}")
            if frag.notes:
                st.markdown(f"**备注:** {frag.notes}")
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
                st.success(f"边缘 {new_edge.id} 已添加到残片 {frag.id}")

        if frag.edges:
            with st.expander("删除边缘"):
                edge_to_del = st.selectbox("选择要删除的边缘", options=[e.id for e in frag.edges], key="edge_del_select")
                if st.button("删除边缘", key="del_edge_btn"):
                    frag.edges = [e for e in frag.edges if e.id != edge_to_del]
                    st.success(f"边缘 {edge_to_del} 已删除")


def page_analysis():
    st.header("候选分析与拼接图谱")

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
        if st.session_state.candidate_matches:
            st.success(f"找到 {len(st.session_state.candidate_matches)} 对候选匹配")
        else:
            st.warning("未找到满足阈值的候选匹配")

    if not st.session_state.candidate_matches:
        return

    st.subheader("候选匹配列表")
    for i, m in enumerate(st.session_state.candidate_matches[:50]):
        st.markdown(
            f"{i+1}. **{m.edge_a_fragment_id}** 的边缘 `{m.edge_a_id}` ↔ "
            f"**{m.edge_b_fragment_id}** 的边缘 `{m.edge_b_id}` — 相似度 **{m.similarity:.3f}**"
        )

    st.subheader("拼接候选图谱")
    G = build_candidate_graph(st.session_state.fragments, st.session_state.candidate_matches)

    fig, ax = plt.subplots(figsize=(10, 8))
    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, seed=42)
        edge_widths = [G[u][v].get("similarity", 0.5) * 5 for u, v in G.edges]
        nx.draw(
            G, pos, ax=ax, with_labels=True, node_color="#4ECDC4",
            node_size=1200, font_size=10, font_weight="bold",
            width=edge_widths, edge_color="#FF6B6B", alpha=0.85,
        )
        edge_labels = {(u, v): f"{G[u][v]['similarity']:.2f}" for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title("拼接候选图谱")
    st.pyplot(fig)
    plt.close(fig)


def page_schemes():
    st.header("方案管理")

    if not st.session_state.candidate_matches:
        st.info("请先在「候选分析」中运行分析，生成候选匹配")
        return

    with st.expander("创建新方案", expanded=True):
        scheme_name = st.text_input("方案名称", key="new_scheme_name")
        available_matches = st.session_state.candidate_matches

        match_labels = [
            f"{m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}] (相似度={m.similarity:.3f})"
            for m in available_matches
        ]
        selected_indices = st.multiselect("选择候选匹配（可多选）", options=list(range(len(match_labels))), format_func=lambda i: match_labels[i], key="scheme_match_select")

        if st.button("创建方案", key="create_scheme_btn"):
            if not scheme_name.strip():
                st.error("方案名称不能为空")
            elif not selected_indices:
                st.error("请至少选择一个候选匹配")
            else:
                selected_matches = [available_matches[i] for i in selected_indices]
                new_scheme = Scheme(name=scheme_name.strip(), matches=selected_matches)

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
                        st.session_state.schemes.append(new_scheme)
                        st.success(f"方案 '{new_scheme.name}' 创建成功 (ID: {new_scheme.id})")

    st.divider()
    st.subheader("已有方案")

    if not st.session_state.schemes:
        st.info("暂无方案")
        return

    for scheme in st.session_state.schemes:
        with st.container():
            coverage, matched, total = compute_scheme_coverage(scheme, st.session_state.fragments)
            conflicts = compute_scheme_conflicts(scheme, st.session_state.schemes, st.session_state.fragments)

            lock_icon = " 🔒" if scheme.is_locked else ""
            st.markdown(f"### 方案: {scheme.name} (ID: {scheme.id}){lock_icon}")
            st.markdown(f"**覆盖率:** {coverage:.1%} ({matched}/{total} 边缘)")
            st.markdown(f"**匹配数:** {len(scheme.matches)}")
            st.markdown(f"**涉及残片:** {', '.join(scheme.get_involved_fragment_ids())}")

            for m in scheme.matches:
                st.markdown(f"&nbsp;&nbsp;- {m.edge_a_fragment_id}[{m.edge_a_id}] ↔ {m.edge_b_fragment_id}[{m.edge_b_id}] (相似度={m.similarity:.3f})")

            if conflicts:
                st.warning(f"**冲突点 ({len(conflicts)}):**")
                for c in conflicts:
                    st.markdown(f"&nbsp;&nbsp;⚠️ [{c['type']}] {c['detail']}")
            else:
                st.success("无冲突")

            c1, c2, c3 = st.columns(3)
            with c1:
                if not scheme.is_locked:
                    if st.button("锁定方案", key=f"lock_scheme_{scheme.id}"):
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag:
                                frag.is_locked = True
                                frag.locked_scheme_id = scheme.id
                        scheme.is_locked = True
                        st.success(f"方案 '{scheme.name}' 已锁定，涉及残片已被占用")
                else:
                    if st.button("解锁方案", key=f"unlock_scheme_{scheme.id}"):
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag and frag.locked_scheme_id == scheme.id:
                                frag.is_locked = False
                                frag.locked_scheme_id = None
                        scheme.is_locked = False
                        st.success(f"方案 '{scheme.name}' 已解锁")
            with c2:
                ok_export, msg_export = validate_export_scheme(scheme)
                if st.button("导出方案", key=f"export_scheme_{scheme.id}", disabled=not ok_export):
                    if not ok_export:
                        st.error(f"导出失败: {msg_export}")
                    else:
                        export_data = scheme.to_dict()
                        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
                        st.download_button(
                            "下载方案JSON", data=json_str,
                            file_name=f"scheme_{scheme.id}.json", mime="application/json",
                            key=f"dl_scheme_{scheme.id}",
                        )
                elif not ok_export:
                    st.caption(f"⚠️ {msg_export}")
            with c3:
                if st.button("删除方案", key=f"del_scheme_{scheme.id}"):
                    if scheme.is_locked:
                        frag_ids = scheme.get_involved_fragment_ids()
                        for fid in frag_ids:
                            frag = next((f for f in st.session_state.fragments if f.id == fid), None)
                            if frag and frag.locked_scheme_id == scheme.id:
                                frag.is_locked = False
                                frag.locked_scheme_id = None
                    st.session_state.schemes = [s for s in st.session_state.schemes if s.id != scheme.id]
                    st.success(f"方案 '{scheme.name}' 已删除")

    st.divider()
    st.subheader("方案覆盖率对比")

    scheme_names = []
    coverages = []
    conflict_counts = []
    for s in st.session_state.schemes:
        cov, _, _ = compute_scheme_coverage(s, st.session_state.fragments)
        confs = compute_scheme_conflicts(s, st.session_state.schemes, st.session_state.fragments)
        scheme_names.append(s.name)
        coverages.append(cov * 100)
        conflict_counts.append(len(confs))

    if scheme_names:
        fig, ax1 = plt.subplots(figsize=(max(6, len(scheme_names) * 1.5), 5))
        x = np.arange(len(scheme_names))
        bars = ax1.bar(x - 0.2, coverages, 0.4, label="覆盖率 (%)", color="#4ECDC4")
        ax1.set_ylabel("覆盖率 (%)")
        ax1.set_ylim(0, 105)
        ax1.set_xticks(x)
        ax1.set_xticklabels(scheme_names, rotation=30, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, conflict_counts, 0.4, label="冲突数", color="#FF6B6B")
        ax2.set_ylabel("冲突数")
        ax2.set_ylim(bottom=0)

        ax1.set_title("方案覆盖率与冲突点对比")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        st.pyplot(fig)
        plt.close(fig)


def page_import_export():
    st.header("导入 / 导出")

    st.subheader("导出全部数据")
    if st.button("导出当前所有残片与方案", key="export_all_btn"):
        export = {
            "fragments": [f.to_dict() for f in st.session_state.fragments],
            "schemes": [s.to_dict() for s in st.session_state.schemes],
        }
        json_str = json.dumps(export, ensure_ascii=False, indent=2)
        st.download_button(
            "下载完整数据JSON", data=json_str,
            file_name="rubbing_analysis_all.json", mime="application/json",
            key="dl_all",
        )

    st.divider()
    st.subheader("导入方案")
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
                    st.success(f"方案 '{scheme_obj.name}' 导入成功")
            else:
                st.error("JSON格式不正确，需要包含 name 和 matches 字段")
        except json.JSONDecodeError:
            st.error("无效的JSON文件")
        except Exception as e:
            st.error(f"导入异常: {str(e)}（不会覆盖当前分析结果）")

    st.divider()
    st.subheader("导入完整数据")
    uploaded_all = st.file_uploader("上传完整数据JSON文件", type=["json"], key="import_all_upload")

    if uploaded_all is not None:
        try:
            content = uploaded_all.read().decode("utf-8")
            data = json.loads(content)

            if "fragments" not in data or "schemes" not in data:
                st.error("JSON缺少 fragments 或 schemes 字段")
            else:
                new_fragments = [Fragment.from_dict(d) for d in data["fragments"]]
                new_schemes = [Scheme.from_dict(d) for d in data["schemes"]]

                existing_ids = {f.id for f in st.session_state.fragments}
                existing_scheme_ids = {s.id for s in st.session_state.schemes}

                frag_conflicts = [f.id for f in new_fragments if f.id in existing_ids]
                scheme_conflicts = [s.id for s in new_schemes if s.id in existing_scheme_ids]

                if frag_conflicts or scheme_conflicts:
                    st.error(
                        f"导入数据与当前数据冲突 — 残片ID冲突: {frag_conflicts}, 方案ID冲突: {scheme_conflicts}。"
                        f"不能覆盖当前分析结果，请先处理冲突"
                    )
                else:
                    st.session_state.fragments.extend(new_fragments)
                    st.session_state.schemes.extend(new_schemes)
                    st.success(f"导入成功: {len(new_fragments)} 个残片, {len(new_schemes)} 个方案")
        except json.JSONDecodeError:
            st.error("无效的JSON文件")
        except Exception as e:
            st.error(f"导入异常: {str(e)}（不会覆盖当前分析结果）")


def main():
    init_session_state()

    st.set_page_config(page_title="碑刻拓片残片拼接分析系统", layout="wide", page_icon="🪨")
    st.title("🪨 碑刻拓片残片拼接分析系统")

    tab_fragments, tab_edges, tab_analysis, tab_schemes, tab_io = st.tabs([
        "残片管理", "边缘标注", "候选分析", "方案管理", "导入/导出",
    ])

    with tab_fragments:
        page_fragments()
    with tab_edges:
        page_edges()
    with tab_analysis:
        page_analysis()
    with tab_schemes:
        page_schemes()
    with tab_io:
        page_import_export()


if __name__ == "__main__":
    main()
