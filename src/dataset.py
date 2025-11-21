import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
    entity2id = {e: i for i, e in enumerate(entities)}
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

class data_dealer:
    def __init__(self, data:dict):
        self.data = data
        self.triples = dict_to_triples(data)
        self.entities, self.entity2id = \
        triples_to_list(self.triples)
        
        #not use in console environment
        #self.draw = draw_by_triple(self.triples)
