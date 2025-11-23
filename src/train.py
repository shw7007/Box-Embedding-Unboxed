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
import copy
from tqdm import tqdm


def train_and_visualize(model, triples, entity2id, HPs, epochs=500, lr=0.01, snapshot_interval=20, filename="Test", level_dict=None):
    # prepare data : Text -> Index
    # train data is composed of  (Child, Parent) pair.
    # triples: [(Child, IsA, Parent), ...]
    child_indices = torch.tensor([entity2id[t[0]] for t in triples], dtype=torch.long)
    parent_indices = torch.tensor([entity2id[t[2]] for t in triples], dtype=torch.long)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # list for saving coordinate data
    history = []
    print("=== Train Start ===")
    
    # take loss hyperparameter from HPs
    margin = HPs["margin"]
    vol_loss_weight = HPs["vol_loss_weight"]
    aspect_ratio_loss_weight = HPs["aspect_ratio_loss_weight"]  

    # list to color entity 
    color_list = HPs["color_list"]

    for epoch in tqdm(range(1, epochs + 1), desc="Training", unit="epoch"):
        model.train() # train mode since now
        optimizer.zero_grad() 
        
        # Caculate positive sample
        pos_dist, pos_vol, c_offsets, p_offsets = model(child_indices, parent_indices)

        # Generate negative sample and calculate
        if HPs["negative_sample"] is True:
            num_samples = child_indices.size(0)
            random_parents = torch.randint(0, len(entity2id), (num_samples,))
            neg_dist, neg_vol, _, _ = model(child_indices, random_parents)
        else:
            neg_dist, neg_vol = 0,0

        # (1) margin loss = positive sample + negative sample
        margin_loss = F.relu(margin+pos_dist-neg_dist).mean()

        # (2) volume loss
        volume_loss = (pos_vol+neg_vol).mean()
        
        # (3) Orthogonal Overlap(Cross problem) loss
        c_diff = (c_offsets[:, 0] - c_offsets[:, 1]) ** 2
        p_diff = (p_offsets[:, 0] - p_offsets[:, 1]) ** 2
        aspect_loss = torch.mean(c_diff + p_diff)
        
        loss = margin_loss + (vol_loss_weight * volume_loss) + (aspect_ratio_loss_weight*aspect_loss)
        
        # Backward Pass: update parameter
        loss.backward()
        optimizer.step()
        
        # --- Visualization and save snapshot ---
        # just save the box data during train process
        if epoch % snapshot_interval == 0 or epoch == 1:
            
            with torch.no_grad():
                min_coords, max_coords = model.get_all_boxes_for_visualization()
                history.append({
                    'epoch' : epoch,
                    'min' : min_coords,
                    'max' : max_coords,
                    'loss' : loss.item()
                })

    print("=== Train Finish ===")
    print("==Image Generation Start==")
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    # list to save images of each epoch
    image_paths = []

    for snap in tqdm(history, desc="Rendering GIF"):
        epoch = snap['epoch']
        min_coords = snap['min']
        max_coords = snap['max']
        loss_val = snap['loss']

        fig, ax = plt.subplots(figsize=(HPs["figure_size"], HPs["figure_size"]))
            
        # fix the axis of the image
        ax.set_xlim(-HPs["screen_size_x"], HPs["screen_size_x"]) 
        ax.set_ylim(-HPs["screen_size_y"], HPs["screen_size_y"])
        ax.set_title(f"Box Embedding Training (Epoch {epoch})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Draw box for every entity
        for entity_name, idx in entity2id.items():
            # Coordinate of the entity
            x_min, y_min = min_coords[idx]
            x_max, y_max = max_coords[idx]
            
            # Calculate width and height of the entitiy's box
            width = x_max - x_min
            height = y_max - y_min
            
            # Assign different colors to boxes with different levels
            if level_dict is not None and level_dict[entity_name] < len(color_list):
                color = color_list[level_dict[entity_name]]
            else:
                color = "black"
            # 사각형 객체 생성
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.7
            )
            ax.add_patch(rect)
            
            # text visualize setting
            ax.text((x_min+x_max)/2, (y_min+y_max)/2, entity_name, 
                    fontsize=10, ha='center', va='center', alpha=0.7)
        
        # save image as file and release memory
        save_path = os.path.join(temp_dir, f"frame_{epoch:05d}.png")
        
        # use dpi from HPs (the higher dpi is, the better image quality)
        plt.savefig(save_path, dpi=HPs["dpi"]) 
        plt.close(fig) # Canvas close
        gc.collect()   # Garbage collection
        
        image_paths.append(save_path)
        
    print("==Image Generation Done==")

    print("=== GIF Converting... ===")
    frames = [imageio.imread(path) for path in image_paths]
    imageio.mimsave('./{}.gif'.format(filename), frames, fps=HPs["fps"])
    shutil.rmtree(temp_dir) # 청소
    print("=== GIF Converting Done ===")


