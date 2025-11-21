import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def dict_to_triples(data:dict, relation="IsA"):
    triples = []
    
    # 2. Triple 생성 (Child, IsA, Parent), 기본 relation이 IsA
    # Box Embedding에서는 보통 (Head, Relation, Tail)을 사용
    # 여기서는 "Child는 Parent의 부분집합이다"를 표현
    for parent, children in data.items():
        for child in children:
            # (Subject, Relation, Object)
            triples.append((child, relation, parent))
            
            # Transitive Data (선택 사항): 
            # 학습을 돕기 위해 'Dog IsA Animal' 같은 건너뛰는 관계도 추가할 수 있음
            # 하지만 Box Embedding의 강력함을 보려면, 직접적인 관계만 주고
            # 건너뛰는 관계를 잘 추론하는지 보는 것이 좋음. (일단은 생략)

    return triples

def triples_to_list(data):
    entities = sorted(list(set([t[0] for t in data] + [t[2] for t in data])))
    entity2id = {e:i for i, e in enumerate(entities)}
    return entities, entity2id

# 4. (참고용) 데이터 구조 시각화 (트리 구조 확인)
# 실제 Box 학습 전, 데이터가 어떻게 생겼는지 확인하는 용도
def draw_by_triple(triples, name="Test dataset"):
    plt.figure(figsize=(8, 6))
    G = nx.DiGraph()
    G.add_edges_from([(t[0], t[2]) for t in triples])
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True)
    plt.title("Graph of {}".format(name))
    plt.show()


# triple을 받아서 {entity(stock) : level(1)} 형태의 dict반환
def get_entity_levels(triples):
    """
    Triples를 분석하여 각 Entity의 깊이(Level)를 계산합니다.
    Root(부모가 없는 노드) = Level 0
    """
    # 1. 그래프 구성 (Adjacency List) 및 모든 노드 파악
    adj_list = defaultdict(list) # Parent -> Children
    all_entities = set()
    children = set() # 누군가의 자식인 노드들
    
    for child, relation, parent in triples:
        adj_list[parent].append(child)
        children.add(child)
        all_entities.add(child)
        all_entities.add(parent)
    
    # 2. Root 노드 찾기 (전체 집합 - 자식 집합)
    # 누군가의 자식이 아닌 노드가 곧 Root입니다.
    roots = list(all_entities - children)
    
    # 3. BFS(너비 우선 탐색)로 레벨 할당
    levels = {}
    queue = deque()
    
    # Root부터 시작 (Level 0)
    for root in roots:
        queue.append((root, 0))
        levels[root] = 0
    
    while queue:
        current_node, current_level = queue.popleft()
        
        # 자식들은 현재 레벨 + 1
        for child in adj_list[current_node]:
            # (주의) 이미 레벨이 매겨진 경우 건너뜀 (Cycle 방지)
            if child not in levels:
                levels[child] = current_level + 1
                queue.append((child, current_level + 1))
                
    return levels

class data_dealer:
    def __init__(self, data:dict):
        self.data = data
        self.triples = dict_to_triples(data)
        self.entities, self.entity2id = \
        triples_to_list(self.triples)
        self.level_dict = get_entity_levels(self.triples)
        
        #not use in console environment
        #self.draw = draw_by_triple(self.triples)
