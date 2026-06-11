import sys
sys.path.insert(0, '.')

from models import *
from utils import *

print('核心模型和工具函数导入成功')

functions_to_check = [
    'add_tag_relation', 'get_tag_relations', 'find_related_tags',
    'create_evidence', 'get_evidence_for_fragment', 'get_evidence_for_scheme',
    'create_evidence_chain', 'add_evidence_to_chain', 'compute_evidence_chain_strength',
    'create_conclusion', 'update_conclusion',
    'create_review_record', 'add_review_to_conclusion',
    'get_conclusions_for_fragment',
    'build_knowledge_graph', 'build_knowledge_graph_nx',
    'find_inference_paths', 'find_relation_path_between_fragments',
    'get_node_type_color', 'deep_compare_schemes',
    'generate_auto_evidence', 'get_conclusion_statistics',
]

missing = []
for func_name in functions_to_check:
    if func_name not in dir():
        missing.append(func_name)

if missing:
    print(f'缺少的函数: {missing}')
else:
    print('所有工具函数都已存在')

print()
print('数据模型检查:')
model_classes = [
    'TagRelationType', 'TagRelation',
    'EvidenceType', 'EvidenceItem', 'EvidenceChain',
    'ConclusionStatus', 'ResearchConclusion', 'ConclusionVersion',
    'ReviewDecision', 'ReviewRecord',
    'KGNodeType', 'KGEdgeType', 'KnowledgeGraph', 'KGNode', 'KGEdge',
    'InferencePath', 'SchemeCompareDepth', 'SchemeDeepCompareResult',
]

missing_models = []
for cls_name in model_classes:
    if cls_name not in dir():
        missing_models.append(cls_name)

if missing_models:
    print(f'缺少的模型: {missing_models}')
else:
    print('所有数据模型都已存在')

print()
print('=== 快速功能测试 ===')

frag1 = Fragment(id='f1', name='残片A', description='测试')
frag2 = Fragment(id='f2', name='残片B', description='测试')

tag1 = CustomTag(id='t1', name='隶书', category=TagCategory.STYLE)
tag2 = CustomTag(id='t2', name='篆书', category=TagCategory.STYLE)

print('创建标签关系...')
relations = []
r = add_tag_relation(relations, tag1.id, tag2.id, TagRelationType.RELATED, 
                     '都是古代字体', 0.6, '研究员A')
print(f'  标签关系: {r.source_tag_id} <-> {r.target_tag_id}')

print('创建证据...')
ev = create_evidence('边缘匹配', EvidenceType.EDGE_FEATURE, '测试描述', 
                     fragment_ids=['f1', 'f2'], confidence=0.85, created_by='研究员A')
print(f'  证据: {ev.title}, 置信度={ev.confidence}')

print('创建证据链...')
chain = create_evidence_chain('测试证据链', '测试描述', [ev.id], created_by='研究员A')
print(f'  证据链: {chain.title}')

print('计算证据链强度...')
strength = compute_evidence_chain_strength(chain, [ev])
print(f'  证据链强度: {strength:.3f}')

print('创建研究结论...')
conc = create_conclusion('测试结论', '测试内容', fragment_ids=['f1'], 
                         evidence_chain_ids=[chain.id], created_by='研究员A')
print(f'  结论: {conc.title}, 状态={conc.status.value}')

print('更新结论...')
conc2 = update_conclusion(conc, title='测试结论（修订）', content='修订内容', 
                          operator='研究员A', change_reason='测试更新')
print(f'  更新后版本: {conc2.version}')
print(f'  历史版本数: {len(conc2.version_history)}')

print('添加审核记录...')
review = create_review_record('conclusion', conc.id, '研究员B', ReviewDecision.COMMENT, 
                              '测试评论', is_official=True)
conc3 = add_review_to_conclusion(conc2, review)
print(f'  审核记录数: {len(conc3.review_records)}')

print()
print('🎉 所有核心功能测试通过！')
