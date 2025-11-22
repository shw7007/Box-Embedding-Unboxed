hyperparamters = {

    # main.py hyperparamters
    "epochs" : 500,
    "lr" : 0.1,

    # modele.py hyperparameters
    "fix_random_seed" : True,
    "initial_center_range" : 4.0,
    "initial_offset_min" : -2.0,
    "initial_offset_max" : -1.0,

    # dataset.py
    "on_grandparent" : False, # True or False
    "topological_trap" : False,

    # train.py -> train hyperparameters
    "margin" : 3,
    "vol_loss_weight" : 0.001,
    "aspect_ratio_loss_weight" : 0,

    # train.py -> visualization hyperparameters
    "screen_size_x" : 10.0,
    "screen_size_y" : 10.0,
    "fps" : 40,
    "color_list" : ["black", "green", "red", "blue"],

    "filename" : "topological trap 0.1",
    "figure_size" : 8,
    "dpi" : 72, # 72 recommended for decent quality


}
