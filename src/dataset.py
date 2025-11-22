import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque

def dict_to_triples(data:dict, relation="IsA"):
    # 1. 역방향 매핑 (자식 -> 부모) 생성
    # 트리를 거슬러 올라가기 위해 필요합니다.
    child_to_parent = {}
    all_entities = set()

    for parent, children in data.items():
        all_entities.add(parent)
        for child in children:
            all_entities.add(child)
            child_to_parent[child] = parent  # 자식의 부모는 누구인가?

    triples = []

    # 2. 모든 Entity에 대해 족보 탐색 (Transitive Closure)
    for entity in all_entities:
        current_node = entity
        
        # 내 위로 부모가 없을 때까지(Root에 도달할 때까지) 계속 올라감
        while current_node in child_to_parent:
            parent = child_to_parent[current_node]
            
            # (나, IsA, 조상님) 관계 추가
            # entity는 고정(나), parent는 계속 위로 올라감(아버지 -> 할아버지...)
            triples.append((entity, 'IsA', parent))
            
            # 한 칸 위로 이동
            current_node = parent

    # 3. 중복 제거 (집합 변환) 및 정렬
    triples = list(set(triples))

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
    """ 할아버지-손자 triple이 없는 경우의 코드
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
                """
    
    # 할아버지-손자 triple이 추가된 경우 코드
    levels = dict()
    for child, rel, parent in triples:
        if(child in levels.keys()):
            levels[child] += 1
        else:
            levels[child] = 1
    for _,__,parent in triples:
        if(parent not in levels.keys()):
            levels[parent] = 0
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
