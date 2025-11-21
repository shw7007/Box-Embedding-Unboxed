import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio # GIF 생성을 위한 라이브러리
import os
import shutil
import gc
import numpy as np

def train_and_visualize(model, triples, entity2id, epochs=500, lr=0.01, snapshot_interval=20, filename="Test", color_dict=None):
    # 1. 데이터 준비 (Text -> Index 변환)
    # 학습 데이터는 (Child, Parent) 쌍으로 구성됩니다.
    # triples: [(Child, IsA, Parent), ...]
    child_indices = torch.tensor([entity2id[t[0]] for t in triples], dtype=torch.long)
    parent_indices = torch.tensor([entity2id[t[2]] for t in triples], dtype=torch.long)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 임시 풀더 생성
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    # GIF 생성을 위한 이미지들을 저장할 주소 리스트
    image_paths = []
    
    print("=== 학습 시작 ===")
    
    margin = 3
    vol_weight = 0.001
    color_list = ["black", "green", "blue", "red"]

    for epoch in range(1, epochs + 1):
        model.train() # 학습 모드
        optimizer.zero_grad() # 기울기 초기화
        
        # positive sample 계산
        pos_dist, pos_vol, c_offsets, p_offsets = model(child_indices, parent_indices)

        # negative sample 생성 및 계산
        num_samples = child_indices.size(0)
        random_parents = torch.randint(0, len(entity2id), (num_samples,))
        # 더 정교하게 하려면, 우연히 정답히 뽑히지 않게 필터링
        neg_dist, neg_vol, _, _ = model(child_indices, random_parents)

        # (1) margin loss = positive sample + negative sample
        margin_loss = F.relu(margin+pos_dist-neg_dist).mean()
        # (2) volume loss
        volume_loss = (pos_vol+neg_vol).mean()
        
        # (3) Orthogonal Overlap(Cross problem) 해결을 위한 loss추가
        c_diff = (c_offsets[:, 0] - c_offsets[:, 1]) ** 2
        p_diff = (p_offsets[:, 0] - p_offsets[:, 1]) ** 2
        aspect_loss = torch.mean(c_diff + p_diff)
        
        aspect_weight = 0.2  # 가중치

        loss = margin_loss + (vol_weight * volume_loss) + (aspect_weight*aspect_loss)
        
        # Backward Pass: 파라미터 업데이트
        loss.backward()
        optimizer.step()
        
        # --- 시각화 및 스냅샷 저장 로직 ---
        if epoch % snapshot_interval == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
            
            # 1. 현재 박스 좌표 가져오기 (GPU -> CPU)
            min_coords, max_coords = model.get_all_boxes_for_visualization()
            
            # 2. 그림 그리기 (Matplotlib)
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # 축 범위 고정 (박스가 움직이는 걸 잘 보려면 배경이 고정돼야 함)
            ax.set_xlim(-12.0, 12.0) # 2.0 -> 5.0으로 확대
            ax.set_ylim(-12.0, 12.0)
            ax.set_title(f"Box Embedding Training (Epoch {epoch})")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # 모든 Entity에 대해 박스 그리기
            for entity_name, idx in entity2id.items():
                # 해당 Entity의 좌표
                x_min, y_min = min_coords[idx]
                x_max, y_max = max_coords[idx]
                
                # 너비와 높이 계산
                width = x_max - x_min
                height = y_max - y_min
                
                # 그룹별 색상 다르게 (시각적 디버깅용)
                # Living Thing 계열은 파랑, My Class 계열은 빨강으로 표시되면 좋음
                if color_dict is not None:
                    color = color_list[color_dict[entity_name]]
                else:
                    color = "black"
                # 사각형 객체 생성
                rect = patches.Rectangle(
                    (x_min, y_min), width, height, 
                    linewidth=2, edgecolor=color, facecolor='none', alpha=0.7
                )
                ax.add_patch(rect)
                
                # 텍스트도 겹치지 않게 작게
                ax.text((x_min+x_max)/2, (y_min+y_max)/2, entity_name, 
                        fontsize=10, ha='center', va='center', alpha=0.7)
            
            # [메모리 최적화 3] 파일로 저장하고 메모리 즉시 해제
            save_path = os.path.join(temp_dir, f"frame_{epoch:05d}.png")
            
            # [용량 최적화 4] dpi=72 (웹용 표준)으로 낮춤. (기본 100)
            plt.savefig(save_path, dpi=72) 
            plt.close(fig) # Canvas 닫기 (필수)
            gc.collect()   # 가비지 컬렉션 (필수)
            
            image_paths.append(save_path)

    print("=== 학습 완료 ===")
    
    print("=== GIF 변환 중... ===")
    frames = [imageio.imread(path) for path in image_paths]
    imageio.mimsave('{}.gif'.format(filename), frames, fps=20)
    shutil.rmtree(temp_dir) # 청소
    print("=== 완료 ===")


