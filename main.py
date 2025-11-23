from src.dataset import data_dealer
from src.model import BoxEmbeddingModel
from src.train import train_and_visualize
from hyperparamters.HPs import hyperparamters as HPs
from src.utils import set_seed
import sys
import argparse

hierarchy = {
        #'Domain' : ['Science', 'Art', 'Business'],

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

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Box embedding visualization")

    parser.add_argument('--mode', type=str, default="fine", 
                        help="Execution mode: [fine, anisotropy, collapse]")
    
    # filename - default : box embedding
    parser.add_argument('--filename', type=str, default="box_embedding",
                        help="Output GIF filename (without extension)")
    
    # seed - default : None(not using seed)
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed (optional)")

    args = parser.parse_args()

    if args.mode == "fine":
        pass
    elif args.mode == "anisotropy":
        HPs["aspect_ratio_loss_weight"] = 0
        HPs["on_grandparent"] = False
    elif args.mode == "collapse":
        HPs["negative_sample"] = False

    HPs["filename"] = args.filename

    if args.seed is not None:
        HPs["fix_random_seed"] = True
        HPs["seed"] = args.seed
        set_seed(HPs["seed"])
    else:
        HPs["fix_random_seed"] = False


# parse above data
k = data_dealer(hierarchy, on_grandparent=HPs["on_grandparent"])

# generate model
model = BoxEmbeddingModel(num_entities=len(k.entities), HPs=HPs, embedding_dim=2)

# start train and visualization
train_and_visualize(model, k.triples, k.entity2id, HPs, epochs=HPs["epochs"], lr=HPs["lr"], snapshot_interval=1, filename=HPs["filename"], level_dict=k.level_dict)

