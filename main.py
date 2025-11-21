from src.dataset import data_dealer
from src.model import BoxEmbeddingModel
from src.train import train_and_visualize

hierarchy = {
    'Living Thing': ['Animal', 'Plant'],
    
    'Animal': ['Mammal', 'Reptile'],
    'Plant': ['Flower', 'Tree'],
    
    'Mammal': ['Dog', 'Cat', 'Human'],
    'Reptile': ['Snake', 'Lizard'],
    'Flower': ['Rose', 'Tulip'],
    'Tree': ['Pine', 'Oak'],

    'My class' : ['Math', 'Engineering', 'Economics'],
    
    'Math' : ['Linear Algebra'],
    'Engineering' : ['CO, eletromagnetic'],
    'Economics' : ['Macro Econ','Mirco Econ']
}

k = data_dealer(hierarchy)

# --- 실행 코드 ---
# 1. 모델 생성 (Entity 개수, 차원=2)
model = BoxEmbeddingModel(num_entities=len(k.entities), embedding_dim=2)

# 2. 학습 및 시각화 실행
# 학습률(lr)이나 에포크(epochs)는 상황에 따라 조절
filename = "living things++"
train_and_visualize(model, k.triples, k.entity2id, epochs=300, lr=0.1, snapshot_interval=1,filename=filename)