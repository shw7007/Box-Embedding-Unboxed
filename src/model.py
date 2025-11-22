import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxEmbeddingModel(nn.Module):
    def __init__(self, num_entities, HPs:dict, embedding_dim=2, volume_temp=1.0):
        super().__init__()

        if(HPs["fix_random_seed"] is True) : torch.manual_seed(1)

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.volume_temp = volume_temp  # Gumbel-Softmax 등에 쓰이는 온도 상수 (여기선 단순화)

        # 1. Center Embedding (위치)
        # Center를 -3.0 ~ 3.0 정도로 넓게 퍼뜨림 (기존 -1~1)
        self.center_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.center_embeddings.weight, -HPs["initial_center_range"], HPs["initial_center_range"])

        # 2. Offset Embedding (크기/너비)
        # 중요: 박스 크기는 무조건 양수여야 함.
        # 초기화: 너무 작으면(0에 수렴) 점이 되고, 너무 크면 학습이 안 됨.
        self.offset_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.offset_embeddings.weight, HPs["initial_offset_min"], HPs["initial_offset_max"])

    def get_boxes(self, indices):
        """
        Entity ID를 받아 [Min_coord, Max_coord]를 반환
        Min = Center - Offset
        Max = Center + Offset
        """
        centers = self.center_embeddings(indices)
        
        # Offset은 항상 양수여야 하므로 Softplus (Log(1+exp(x))) 통과
        # 그냥 abs()를 쓰면 0에서 미분 불가능 점이 생겨 학습이 불안정할 수 있음
        offsets = F.softplus(self.offset_embeddings(indices))
        
        min_coords = centers - offsets
        max_coords = centers + offsets
        
        return min_coords, max_coords, centers, offsets

    def forward(self, child_indices, parent_indices, lambda_vol=0.001): # lambda 추가
        c_min, c_max, _, c_offsets = self.get_boxes(child_indices)
        p_min, p_max, _, p_offsets = self.get_boxes(parent_indices)
        
        # --- Intersection Loss (L1 Norm 추천) ---
        # L2보다 힘이 일정해서 '교착 상태' 방지에 더 유리함
        violation_min = F.relu(p_min - c_min)
        violation_max = F.relu(c_max - p_max)
        
        # dim=-1은 그대로 두고, p=1로 변경
        distance = torch.norm(violation_min, p=1, dim=-1) + \
                            torch.norm(violation_max, p=1, dim=-1)
        
        # --- Volume Loss ---
        # 박스 크기(Offset)의 합
        volume = torch.sum(c_offsets, dim=-1) + torch.sum(p_offsets, dim=-1)
        
        # 최종 Loss
        return distance, volume, c_offsets, p_offsets

    def get_all_boxes_for_visualization(self):
        """시각화를 위해 모든 박스 좌표 반환 (Detach)"""
        with torch.no_grad():
            centers = self.center_embeddings.weight.data
            offsets = F.softplus(self.offset_embeddings.weight.data)
            
            min_coords = centers - offsets
            max_coords = centers + offsets
            
            return min_coords.cpu().numpy(), max_coords.cpu().numpy()