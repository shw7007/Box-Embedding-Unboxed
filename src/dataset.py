import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def generate_toy_hierarchy():
    """
    3단계 깊이의 계층 구조 데이터를 생성합니다.
    Level 0: Living Thing (생물)
    Level 1: Animal, Plant (동물, 식물)
    Level 2: Mammal, Reptile, Flower, Tree (포유류, 파충류 등)
    Level 3: Dog, Cat, Snake, Rose, Pine (구체적 대상)
    """
    
    # 1. 계층 구조 정의 (Parent -> Children)
    hierarchy = {
        'Living Thing': ['Animal', 'Plant'],
        
        'Animal': ['Mammal', 'Reptile'],
        'Plant': ['Flower', 'Tree'],
        
        'Mammal': ['Dog', 'Cat', 'Human'],
        'Reptile': ['Snake', 'Lizard'],
        'Flower': ['Rose', 'Tulip'],
        'Tree': ['Pine', 'Oak']
    }

    triples = []
    
    # 2. Triple 생성 (Child, IsA, Parent)
    # Box Embedding에서는 보통 (Head, Relation, Tail)을 사용
    # 여기서는 "Child는 Parent의 부분집합이다"를 표현
    for parent, children in hierarchy.items():
        for child in children:
            # (Subject, Relation, Object)
            triples.append((child, 'IsA', parent))
            
            # Transitive Data (선택 사항): 
            # 학습을 돕기 위해 'Dog IsA Animal' 같은 건너뛰는 관계도 추가할 수 있음
            # 하지만 Box Embedding의 강력함을 보려면, 직접적인 관계만 주고
            # 건너뛰는 관계를 잘 추론하는지 보는 것이 좋음. (일단은 생략)

    return triples, hierarchy

# 데이터 생성 실행
toy_triples, toy_adj = generate_toy_hierarchy()

# 3. 데이터 확인 및 ID 매핑 (문자열 -> 숫자)
entities = sorted(list(set([t[0] for t in toy_triples] + [t[2] for t in toy_triples])))
entity2id = {e: i for i, e in enumerate(entities)}

print(f"=== 총 Entity 개수: {len(entities)} ===")
print(entity2id)
print("\n=== 생성된 학습 데이터 (Triples) ===")
df = pd.DataFrame(toy_triples, columns=['Head', 'Relation', 'Tail'])
print(df)



# 4. (참고용) 데이터 구조 시각화 (트리 구조 확인)
# 실제 Box 학습 전, 데이터가 어떻게 생겼는지 확인하는 용도
plt.figure(figsize=(8, 6))
G = nx.DiGraph()
G.add_edges_from([(t[0], t[2]) for t in toy_triples])
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, arrows=True)
plt.title("Structure of Toy Dataset")
plt.show()