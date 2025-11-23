hyperparamters = {

    # main.py hyperparamters
    "epochs" : 1000,
    "lr" : 0.08,
    "fix_random_seed" : True,
    "seed" : 11,

    # modele.py hyperparameters
    "initial_center_range" : 4.0, # =4 ~ +4
    "initial_offset_min" : -2.0,
    "initial_offset_max" : -1.0,

    # dataset.py
    "on_grandparent" : True, # True or False
    "negative_sample" : True,

    # train.py -> train hyperparameters
    "margin" : 3,
    "vol_loss_weight" : 0.02,
    "aspect_ratio_loss_weight" : 0.2,
    #"pos_neg_sample_ratio" : 1,

    # train.py -> visualization hyperparameters
    "screen_size_x" : 10.0,
    "screen_size_y" : 10.0,
    "fps" : 40,
    "color_list" : ["black", "green", "red", "blue"],
    "figure_size" : 8,
    "dpi" : 72, # 72 recommended for decent quality

    "filename" : "no negative",

}


