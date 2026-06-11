import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from models import (
    Fragment, EdgeFeature, Scheme, EdgeMatch,
    CustomTag, TagCategory, SemanticAnnotation,
    TagRelationType, TagRelation,
    EvidenceType, EvidenceItem, EvidenceChain,
    ConclusionStatus, ResearchConclusion, ConclusionVersion,
    ReviewDecision, ReviewRecord,
    KGNodeType, KGEdgeType, KnowledgeGraph, KGNode, KGEdge,
    InferencePath, SchemeCompareDepth, SchemeDeepCompareResult,
    OperationType,
)
from utils import (
    add_tag_relation, get_tag_relations, find_related_tags,
    create_evidence, get_evidence_for_fragment, get_evidence_for_scheme,
    create_evidence_chain, add_evidence_to_chain, compute_evidence_chain_strength,
    create_conclusion, update_conclusion,
    create_review_record, add_review_to_conclusion,
    get_conclusions_for_fragment,
    build_knowledge_graph, build_knowledge_graph_nx,
    find_inference_paths, find_relation_path_between_fragments,
    get_node_type_color, deep_compare_schemes,
    generate_auto_evidence, get_conclusion_statistics,
    now_str, generate_id,
)


def test_tag_relations():
    print("\n=== 测试标签关系 ===")
    
    tag1 = CustomTag(name="隶书", category=TagCategory.STYLE, description="隶书字体", color="#3498DB")
    tag2 = CustomTag(name="篆书", category=TagCategory.STYLE, description="篆书字体", color="#2ECC71")
    tag3 = CustomTag(name="古文字", category=TagCategory.STYLE, description="古代文字", color="#E74C3C")
    
    relations = []
    
    r1 = add_tag_relation(relations, tag1.id, tag2.id, TagRelationType.SYNONYMOUS, 
                          "都是古代字体", 0.6, "研究员A")
    print(f"创建同义关系: {r1.tag_a_id} <-> {r1.tag_b_id}, 类型={r1.relation_type.value}")
    
    r2 = add_tag_relation(relations, tag3.id, tag1.id, TagRelationType.SUPERCLASS,
                          "隶书属于古文字", 0.9, "研究员A")
    print(f"创建上位关系: {r2.tag_a_id} -> {r2.tag_b_id}, 类型={r2.relation_type.value}")
    
    related = find_related_tags(relations, tag1.id, max_depth=2)
    print(f"与tag1相关的标签数: {len(related)}")
    
    print("✅ 标签关系测试通过")


def test_evidence_system():
    print("\n=== 测试证据系统 ===")
    
    frag1 = Fragment(id="f1", name="残片A", description="测试残片")
    frag2 = Fragment(id="f2", name="残片B", description="测试残片")
    
    scheme = Scheme(id="s1", name="测试方案", fragments=["f1", "f2"], matches=[])
    
    e1 = create_evidence(
        fragment_ids=["f1"],
        scheme_ids=["s1"],
        evidence_type=EvidenceType.EDGE_MATCH,
        title="边缘形态匹配",
        description="两残片边缘纹理高度吻合",
        confidence=0.85,
        operator="研究员A",
    )
    print(f"创建证据: {e1.title}, 置信度={e1.confidence}")
    
    e2 = create_evidence(
        fragment_ids=["f1", "f2"],
        scheme_ids=["s1"],
        evidence_type=EvidenceType.SEMANTIC_SIMILARITY,
        title="文字语义关联",
        description="两残片文字内容连续",
        confidence=0.72,
        operator="研究员B",
    )
    print(f"创建证据: {e2.title}, 置信度={e2.confidence}")
    
    ev_chain = create_evidence_chain(
        name="残片A-B关联证据链",
        description="支持残片A和B拼接的证据集合",
        evidence_ids=[e1.id, e2.id],
        creator="研究员A",
    )
    print(f"创建证据链: {ev_chain.name}, 证据数={len(ev_chain.evidence_ids)}")
    
    strength = compute_evidence_chain_strength(ev_chain, [e1, e2])
    print(f"证据链强度: {strength:.3f}")
    
    print("✅ 证据系统测试通过")


def test_research_conclusions():
    print("\n=== 测试研究结论 ===")
    
    frag1 = Fragment(id="f1", name="残片A", description="测试残片")
    
    ev = create_evidence(
        fragment_ids=["f1"],
        evidence_type=EvidenceType.HISTORICAL_RECORD,
        title="文献记载",
        description="据《金石录》记载，此碑刻立于东汉",
        confidence=0.9,
        operator="研究员A",
    )
    
    conc = create_conclusion(
        title="残片A的年代考证",
        content="根据文献记载和字体风格分析，此残片应为东汉时期所刻",
        fragment_ids=["f1"],
        evidence_ids=[ev.id],
        conclusion_type="年代考证",
        operator="研究员A",
    )
    print(f"创建结论: {conc.title}, 版本数={len(conc.versions)}")
    
    conc2 = update_conclusion(
        conc,
        title="残片A的年代考证（修订）",
        content="根据文献记载和字体风格分析，此残片应为东汉末年所刻，约公元180年",
        operator="研究员A",
        change_reason="补充具体年代",
    )
    print(f"更新结论后版本数: {len(conc2.versions)}")
    
    review = create_review_record(
        conclusion_id=conc2.id,
        reviewer="研究员B",
        decision=ReviewDecision.COMMENT,
        comment="论证充分，建议补充更多出土文物佐证",
        is_formal=True,
    )
    
    conc3 = add_review_to_conclusion(conc2, review)
    print(f"添加审核后记录数: {len(conc3.review_records)}")
    print(f"结论状态: {conc3.status.value}")
    
    stats = get_conclusion_statistics([conc3])
    print(f"结论统计: 总数={stats['total']}, 草稿={stats['draft']}")
    
    print("✅ 研究结论测试通过")


def test_knowledge_graph():
    print("\n=== 测试知识图谱 ===")
    
    frag1 = Fragment(id="f1", name="残片A", description="测试残片")
    frag2 = Fragment(id="f2", name="残片B", description="测试残片")
    
    edge1 = EdgeFeature(id="e1", fragment_id="f1", edge_position="left", 
                        description="左侧边缘", texture_features={"pattern": "锯齿形"})
    edge2 = EdgeFeature(id="e2", fragment_id="f2", edge_position="right",
                        description="右侧边缘", texture_features={"pattern": "锯齿形"})
    
    match = EdgeMatch(id="m1", fragment_a_id="f1", fragment_b_id="f2",
                      edge_a_id="e1", edge_b_id="e2", similarity_score=0.82)
    
    scheme = Scheme(id="s1", name="测试方案", fragments=["f1", "f2"], 
                    matches=[match], status="approved")
    
    tag1 = CustomTag(id="t1", name="隶书", category=TagCategory.STYLE)
    tag2 = CustomTag(id="t2", name="东汉", category=TagCategory.PERIOD)
    
    tag_rel = TagRelation(id="tr1", tag_a_id="t1", tag_b_id="t2",
                          relation_type=TagRelationType.RELATED, weight=0.6)
    
    anno1 = SemanticAnnotation(id="a1", fragment_id="f1", tag_id="t1",
                               confidence=0.9, created_by="研究员A")
    anno2 = SemanticAnnotation(id="a2", fragment_id="f2", tag_id="t1",
                               confidence=0.85, created_by="研究员A")
    anno3 = SemanticAnnotation(id="a3", fragment_id="f1", tag_id="t2",
                               confidence=0.7, created_by="研究员B")
    
    ev = create_evidence(fragment_ids=["f1", "f2"], scheme_ids=["s1"],
                         evidence_type=EvidenceType.EDGE_MATCH, title="边缘匹配",
                         description="测试证据", confidence=0.82, operator="研究员A")
    
    conc = create_conclusion(title="测试结论", content="测试内容",
                             fragment_ids=["f1"], evidence_ids=[ev.id],
                             operator="研究员A")
    
    fragments = [frag1, frag2]
    edges = [edge1, edge2]
    schemes = [scheme]
    tags = [tag1, tag2]
    annotations = [anno1, anno2, anno3]
    evidences = [ev]
    conclusions = [conc]
    tag_relations = [tag_rel]
    
    kg = build_knowledge_graph(fragments, edges, schemes, tags, annotations, 
                               evidences, conclusions, tag_relations)
    print(f"知识图谱节点数: {len(kg.nodes)}")
    print(f"知识图谱边数: {len(kg.edges)}")
    
    node_types = {}
    for n in kg.nodes.values():
        node_types[n.node_type.value] = node_types.get(n.node_type.value, 0) + 1
    print(f"节点类型分布: {node_types}")
    
    try:
        import networkx as nx
        G = build_knowledge_graph_nx(kg)
        print(f"NetworkX图节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
        
        paths = find_inference_paths(kg, "f1", "f2", max_depth=5, max_paths=3)
        print(f"推理路径数: {len(paths)}")
        if paths:
            print(f"最强路径强度: {paths[0].strength:.3f}")
            print(f"路径节点: {[n.name for n in paths[0].nodes]}")
    except ImportError:
        print("⚠️ 未安装networkx，跳过NetworkX相关测试")
    
    print("✅ 知识图谱测试通过")


def test_deep_compare():
    print("\n=== 测试深度比对 ===")
    
    frag1 = Fragment(id="f1", name="残片A", description="测试残片")
    frag2 = Fragment(id="f2", name="残片B", description="测试残片")
    frag3 = Fragment(id="f3", name="残片C", description="测试残片")
    
    match1 = EdgeMatch(id="m1", fragment_a_id="f1", fragment_b_id="f2",
                       edge_a_id="e1", edge_b_id="e2", similarity_score=0.85)
    match2 = EdgeMatch(id="m2", fragment_a_id="f1", fragment_b_id="f2",
                       edge_a_id="e1", edge_b_id="e2", similarity_score=0.78)
    
    scheme1 = Scheme(id="s1", name="方案甲", fragments=["f1", "f2"], matches=[match1])
    scheme2 = Scheme(id="s2", name="方案乙", fragments=["f1", "f2", "f3"], matches=[match2])
    
    result = deep_compare_schemes(scheme1, scheme2, [], [], [], [])
    print(f"总体相似度: {result.overall_similarity:.3f}")
    print(f"匹配相似度: {result.match_similarity:.3f}")
    print(f"残片相似度: {result.fragment_similarity:.3f}")
    print(f"差异项数: {len(result.differences)}")
    
    print("✅ 深度比对测试通过")


def test_serialization():
    print("\n=== 测试序列化/反序列化 ===")
    
    ev = EvidenceItem(
        id="ev1",
        fragment_ids=["f1", "f2"],
        scheme_ids=["s1"],
        evidence_type=EvidenceType.EDGE_MATCH,
        title="测试证据",
        description="测试描述",
        confidence=0.85,
        source_url="http://example.com",
        created_by="研究员A",
        created_at="2024-01-01 12:00:00",
    )
    
    ev_dict = ev.to_dict()
    ev2 = EvidenceItem.from_dict(ev_dict)
    assert ev2.id == ev.id
    assert ev2.title == ev.title
    assert ev2.evidence_type == ev.evidence_type
    print(f"EvidenceItem 序列化 OK")
    
    ec = EvidenceChain(
        id="ec1",
        name="测试证据链",
        description="测试描述",
        evidence_ids=["ev1", "ev2"],
        overall_strength=0.8,
        created_by="研究员A",
        created_at="2024-01-01 12:00:00",
        updated_at="2024-01-01 12:00:00",
    )
    
    ec_dict = ec.to_dict()
    ec2 = EvidenceChain.from_dict(ec_dict)
    assert ec2.id == ec.id
    assert ec2.name == ec.name
    print(f"EvidenceChain 序列化 OK")
    
    conc = ResearchConclusion(
        id="c1",
        title="测试结论",
        content="测试内容",
        fragment_ids=["f1"],
        scheme_ids=["s1"],
        evidence_ids=["ev1"],
        tags=["tag1"],
        conclusion_type="类型",
        status=ConclusionStatus.DRAFT,
        created_by="研究员A",
        created_at="2024-01-01 12:00:00",
        updated_at="2024-01-01 12:00:00",
        versions=[],
        review_records=[],
    )
    
    conc_dict = conc.to_dict()
    conc2 = ResearchConclusion.from_dict(conc_dict)
    assert conc2.id == conc.id
    assert conc2.title == conc.title
    assert conc2.status == conc.status
    print(f"ResearchConclusion 序列化 OK")
    
    tr = TagRelation(
        id="tr1",
        tag_a_id="t1",
        tag_b_id="t2",
        relation_type=TagRelationType.SYNONYMOUS,
        description="同义",
        weight=0.9,
        created_by="研究员A",
        created_at="2024-01-01 12:00:00",
    )
    
    tr_dict = tr.to_dict()
    tr2 = TagRelation.from_dict(tr_dict)
    assert tr2.id == tr.id
    assert tr2.relation_type == tr.relation_type
    print(f"TagRelation 序列化 OK")
    
    print("✅ 序列化测试全部通过")


def test_kg_serialization():
    print("\n=== 测试知识图谱序列化 ===")
    
    node = KGNode(
        id="n1",
        node_type=KGNodeType.FRAGMENT,
        name="残片A",
        properties={"size": "large"},
    )
    
    edge = KGEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type=KGEdgeType.HAS_TAG,
        weight=0.8,
        properties={"confidence": 0.9},
    )
    
    kg = KnowledgeGraph(
        nodes={"n1": node, "n2": KGNode(id="n2", node_type=KGNodeType.TAG, name="标签A")},
        edges=[edge],
    )
    
    kg_dict = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(kg_dict)
    
    assert len(kg2.nodes) == 2
    assert len(kg2.edges) == 1
    assert kg2.nodes["n1"].name == "残片A"
    assert kg2.edges[0].edge_type == KGEdgeType.HAS_TAG
    
    print("✅ 知识图谱序列化测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("残片知识图谱与协同考释平台 - 功能测试")
    print("=" * 50)
    
    try:
        test_tag_relations()
        test_evidence_system()
        test_research_conclusions()
        test_knowledge_graph()
        test_deep_compare()
        test_serialization()
        test_kg_serialization()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过！")
        print("=" * 50)
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)
