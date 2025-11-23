hyperparamters = {

    # main.py hyperparamters
    "epochs" : 1000, # epochs
    "lr" : 0.08, # learning rate
    "fix_random_seed" : True, # True if you use fixed random seed, False makes different output every time you execute
    "seed" : 11, # Seed is used only if "fix_random_seed" is True

    # modele.py hyperparameters
    "initial_center_range" : 4.0, # Initial box center embedding's random distribution range
    "initial_offset_min" : -2.0, # Initial box offset's range
    "initial_offset_max" : -1.0, # Initial box offset's range, now -2.0 ~ -1.0

    # dataset.py
    "on_grandparent" : True, # True if you make grandparent-grandchild relations, False makes only parent-child realtion
    "negative_sample" : True, # True if you use negative sample, False otherwise

    # train.py -> train hyperparameters
    "margin" : 3, # Higher margin makes boxes more sensitive to positive/negative learning
    "vol_loss_weight" : 0.02, # Higher vol_loss_weight makes boxes much smaller
    "aspect_ratio_loss_weight" : 0.2, # Higher aspect_ratio_loss_weight makes boxes more square-like

    # train.py -> visualization hyperparameters
    "screen_size_x" : 10.0, # x axis screen size of image
    "screen_size_y" : 10.0, # y axis screen size of image
    "fps" : 40, # image frame used per sec in GIF
    "color_list" : ["black", "green", "red", "blue"], # color list for painting boxes, now root box will be "black"
    "figure_size" : 8, # the size of image
    "dpi" : 72, # 72 recommended for decent quality

    "filename" : "Have a Good day", # Name of the file to be saved

}


