import sys
import os

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
from utils import now_str


def test_basic_models():
    print("\n=== 测试基础模型 ===")
    
    tag_rel = TagRelation(
        source_tag_id="t1",
        target_tag_id="t2",
        relation_type=TagRelationType.SYNONYM,
        confidence=0.8,
        description="测试同义关系",
        created_by="研究员A",
    )
    print(f"标签关系: {tag_rel.source_tag_id} -> {tag_rel.target_tag_id}, 类型={tag_rel.relation_type.value}")
    
    ev = EvidenceItem(
        evidence_type=EvidenceType.EDGE_FEATURE,
        title="边缘特征匹配",
        description="测试描述",
        fragment_ids=["f1", "f2"],
        confidence=0.85,
        created_by="研究员A",
    )
    print(f"证据: {ev.title}, 类型={ev.evidence_type.value}, 置信度={ev.confidence}")
    
    chain = EvidenceChain(
        title="测试证据链",
        description="测试证据链描述",
        evidence_ids=[ev.id],
        created_by="研究员A",
    )
    print(f"证据链: {chain.title}, 证据数={len(chain.evidence_ids)}")
    
    conc = ResearchConclusion(
        title="测试结论",
        content="测试内容",
        conclusion_type="年代考证",
        fragment_ids=["f1"],
        status=ConclusionStatus.PROPOSED,
        created_by="研究员A",
    )
    print(f"研究结论: {conc.title}, 状态={conc.status.value}, 版本={conc.version}")
    
    review = ReviewRecord(
        conclusion_id=conc.id,
        reviewer="研究员B",
        decision=ReviewDecision.COMMENT,
        comment="测试评论",
    )
    print(f"审核记录: 决定={review.decision.value}, 审核人={review.reviewer}")
    
    print("✅ 基础模型测试通过")


def test_serialization():
    print("\n=== 测试序列化 ===")
    
    tag_rel = TagRelation(
        id="tr1",
        source_tag_id="t1",
        target_tag_id="t2",
        relation_type=TagRelationType.SYNONYM,
        confidence=0.9,
        description="同义",
        created_by="研究员A",
        created_at="2024-01-01",
    )
    
    d = tag_rel.to_dict()
    tag_rel2 = TagRelation.from_dict(d)
    assert tag_rel2.id == tag_rel.id
    assert tag_rel2.relation_type == tag_rel.relation_type
    assert tag_rel2.confidence == tag_rel.confidence
    print("TagRelation 序列化 OK")
    
    ev = EvidenceItem(
        id="ev1",
        evidence_type=EvidenceType.EDGE_FEATURE,
        title="测试证据",
        description="测试描述",
        fragment_ids=["f1", "f2"],
        scheme_ids=["s1"],
        confidence=0.85,
        created_by="研究员A",
        created_at="2024-01-01",
        updated_at="2024-01-02",
    )
    
    d = ev.to_dict()
    ev2 = EvidenceItem.from_dict(d)
    assert ev2.id == ev.id
    assert ev2.evidence_type == ev.evidence_type
    assert ev2.fragment_ids == ev.fragment_ids
    print("EvidenceItem 序列化 OK")
    
    chain = EvidenceChain(
        id="ec1",
        title="测试链",
        description="测试描述",
        evidence_ids=["ev1", "ev2"],
        conclusion_id="c1",
        chain_strength=0.8,
        created_by="研究员A",
        created_at="2024-01-01",
        updated_at="2024-01-02",
    )
    
    d = chain.to_dict()
    chain2 = EvidenceChain.from_dict(d)
    assert chain2.id == chain.id
    assert chain2.evidence_ids == chain.evidence_ids
    assert chain2.chain_strength == chain.chain_strength
    print("EvidenceChain 序列化 OK")
    
    conc = ResearchConclusion(
        id="c1",
        title="测试结论",
        content="测试内容",
        conclusion_type="类型",
        fragment_ids=["f1"],
        scheme_ids=["s1"],
        tag_ids=["t1"],
        evidence_chain_ids=["ec1"],
        status=ConclusionStatus.PROPOSED,
        confidence=0.7,
        created_by="研究员A",
        created_at="2024-01-01",
        updated_at="2024-01-02",
        version=2,
        version_history=[],
        review_records=[],
        parent_conclusion_id="",
    )
    
    d = conc.to_dict()
    conc2 = ResearchConclusion.from_dict(d)
    assert conc2.id == conc.id
    assert conc2.status == conc.status
    assert conc2.version == conc.version
    print("ResearchConclusion 序列化 OK")
    
    kg_node = KGNode(
        id="n1",
        node_type=KGNodeType.FRAGMENT,
        label="残片A",
        properties={"size": "大"},
    )
    
    kg_edge = KGEdge(
        id="e1",
        source_id="n1",
        target_id="n2",
        edge_type=KGEdgeType.HAS_TAG,
        weight=0.8,
        properties={"confidence": 0.9},
    )
    
    kg = KnowledgeGraph(
        nodes=[
            kg_node,
            KGNode(id="n2", node_type=KGNodeType.TAG, label="标签A"),
        ],
        edges=[kg_edge],
    )
    
    d = kg.to_dict()
    kg2 = KnowledgeGraph.from_dict(d)
    assert len(kg2.nodes) == 2
    assert len(kg2.edges) == 1
    assert kg2.nodes[0].label == "残片A"
    assert kg2.edges[0].edge_type == KGEdgeType.HAS_TAG
    print("KnowledgeGraph 序列化 OK")
    
    print("✅ 所有序列化测试通过")


def test_kg_structure():
    print("\n=== 测试知识图谱结构 ===")
    
    kg = KnowledgeGraph(name="测试图谱")
    
    n1 = KGNode(id="f1", node_type=KGNodeType.FRAGMENT, label="残片A")
    n2 = KGNode(id="f2", node_type=KGNodeType.FRAGMENT, label="残片B")
    n3 = KGNode(id="t1", node_type=KGNodeType.TAG, label="隶书")
    n4 = KGNode(id="s1", node_type=KGNodeType.SCHEME, label="方案甲")
    
    kg.nodes.extend([n1, n2, n3, n4])
    
    e1 = KGEdge(source_id="f1", target_id="t1", edge_type=KGEdgeType.HAS_TAG, weight=0.9)
    e2 = KGEdge(source_id="f2", target_id="t1", edge_type=KGEdgeType.HAS_TAG, weight=0.85)
    e3 = KGEdge(source_id="f1", target_id="f2", edge_type=KGEdgeType.MATCHES, weight=0.82)
    e4 = KGEdge(source_id="s1", target_id="f1", edge_type=KGEdgeType.CONTAINS, weight=1.0)
    e5 = KGEdge(source_id="s1", target_id="f2", edge_type=KGEdgeType.CONTAINS, weight=1.0)
    
    kg.edges.extend([e1, e2, e3, e4, e5])
    
    print(f"节点数: {len(kg.nodes)}")
    print(f"边数: {len(kg.edges)}")
    
    has_tag_edges = [e for e in kg.edges if e.edge_type == KGEdgeType.HAS_TAG]
    print(f"具有标签的边数: {len(has_tag_edges)}")
    
    print("✅ 知识图谱结构测试通过")


if __name__ == "__main__":
    print("=" * 50)
    print("残片知识图谱与协同考释平台 - 模型测试")
    print("=" * 50)
    
    try:
        test_basic_models()
        test_serialization()
        test_kg_structure()
        
        print("\n" + "=" * 50)
        print("🎉 所有模型测试通过！")
        print("=" * 50)
    except Exception as e:
        import traceback
        print(f"\n❌ 测试失败: {e}")
        traceback.print_exc()
        sys.exit(1)
