from src.dataset import data_dealer
from src.model import BoxEmbeddingModel
from src.train import train_and_visualize
from hyperparamters.HPs import hyperparamters as HPs

hierarchy = {
        # --- Group 1: Science ---
        'Science': ['Physics', 'CS'],
        'Physics': ['Quantum', 'Gravity'],
        'CS': ['AI', 'Security'],
        
        # --- Group 2: Art ---
        'Art': ['Music', 'Visual Art'],
        'Music': ['Jazz', 'Rock'],
        'Visual Art': ['Painting', 'Sketch'],
        
        # --- Group 3: Business ---
        'Business': ['Finance', 'Marketing'],
        'Finance': ['Stock', 'Bond'],
        'Marketing': ['Ads', 'Branding']
}

k = data_dealer(hierarchy)

# --- 실행 코드 ---
# 1. 모델 생성 (Entity 개수, 차원=2)
model = BoxEmbeddingModel(num_entities=len(k.entities), HPs=HPs, embedding_dim=2)

# 2. 학습 및 시각화 실행
# 학습률(lr)이나 에포크(epochs)는 상황에 따라 조절
filename = "living things++"
train_and_visualize(model, k.triples, k.entity2id, HPs, epochs=HPs["epochs"], lr=HPs["lr"], snapshot_interval=1, filename=filename, level_dict=k.level_dict)

