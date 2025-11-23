import torch
import torch.nn as nn
import torch.nn.functional as F

class BoxEmbeddingModel(nn.Module):
    def __init__(self, num_entities, HPs:dict, embedding_dim=2, volume_temp=1.0):
        super().__init__()

        self.num_entities = num_entities
        self.embedding_dim = embedding_dim
        self.volume_temp = volume_temp  # temperature constant used in Gumbel-Softmax 

        # 1. Center Embedding 
        self.center_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.center_embeddings.weight, -HPs["initial_center_range"], HPs["initial_center_range"])

        # 2. Offset Embedding (length/width)
        # importatnt : box offset should be positive, but embedding doesn't need to be positive due to Gumbel-Softmax
        self.offset_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.offset_embeddings.weight, HPs["initial_offset_min"], HPs["initial_offset_max"])

    def get_boxes(self, indices):
        # return Min coord and Max coord of box using entity ID

        centers = self.center_embeddings(indices)
        
        # Offset should always be positive, thus use Softplus : (Log(1+exp(x)))
        # Just using abs() to make offset positive makes non-diffrential point which makes train unstable
        offsets = F.softplus(self.offset_embeddings(indices))
        
        min_coords = centers - offsets
        max_coords = centers + offsets
        
        return min_coords, max_coords, centers, offsets

    def forward(self, child_indices, parent_indices, lambda_vol=0.001): # lambda 추가
        c_min, c_max, _, c_offsets = self.get_boxes(child_indices)
        p_min, p_max, _, p_offsets = self.get_boxes(parent_indices)
        
        # --- Intersection Loss (L1 Norm) ---
        violation_min = F.relu(p_min - c_min)
        violation_max = F.relu(c_max - p_max)
        distance = torch.norm(violation_min, p=1, dim=-1) + \
                            torch.norm(violation_max, p=1, dim=-1)
        
        # --- Volume Loss ---
        # sum off box offset
        volume = torch.sum(c_offsets, dim=-1) + torch.sum(p_offsets, dim=-1)
        
        # Final Loss
        return distance, volume, c_offsets, p_offsets

    def get_all_boxes_for_visualization(self):
        # return every box coordinate for visualization
        with torch.no_grad():
            centers = self.center_embeddings.weight.data
            offsets = F.softplus(self.offset_embeddings.weight.data)
            
            min_coords = centers - offsets
            max_coords = centers + offsets
            
            return min_coords.cpu().numpy(), max_coords.cpu().numpy()