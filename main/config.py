import os
import sys
import yaml

class Base_Config():
    # common
    dev_mode = False
    name = "train01"
    msg = ""
    continue_train = False
    load_model_path = None

    # dataset
    action_dataset = "BABEL"
    body_model = "SMPLH"

    # for BABEL dataset
    max_input_len = 120
    min_input_len = 20
    sample_fps = 20  # dataset fps: 60, 100, 120, 150, 250 -> sample fps
    sampling = "adjacent_two"
    filter_label = "twenty"
    S1_end_len = 4
    train_per_label = 20
    transition_rate = 0.1
    max_transition_len = 5
    S2_extra_train_frames = 3
    S2_extra_val_frames = 3
    extra_S2_frames = {"train": S2_extra_train_frames, "val": S2_extra_val_frames}

    # actions
    action_profiles = action_dataset # loaded in set_env

    # model
    model = "CVAE"
    loss_type = "L1"
    loss_dim = "mmm"

    encoder = "Transformer"
    decoder = "Transformer"

    # model spec
    Transformer_spec = {
        "embed_dim": 512,
        "nhead": 4,
        "enc_layers": 8,
        "dec_layers": 8,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "activation": "gelu"
    }

    # Model
    input_rotation_format = "axis"  # axis, euler or 6dim
    output_rotation_format = "6dim"  # axis, euler or 6dim
    
    postprocessor = "convolution"  # hourglass or convolution or none
    post_inner_dim = "6dim"  # euler or 6dim

    layered_pos_enc = True
    model_fill_last_frame = True

    # train
    start_epoch = 0
    end_epoch = 1000
    batch_size = 24
    train_lr = 0.0001
    kl_weight = 0.00001
    mesh_weight = 1
    accel_weight = 1

    print_epoch = 10
    vis_epoch = 100
    shuffle = True

    # generate
    gen_model_path = None  # must be defined in env yaml file
    output_duration = 60
    gen_batch_size = 10
    vis_num = 5

    resume_generation = False
    resume_exp = 0
    resume_step = 0

    target_labels = ["stand", "tpose", "stand", "tpose", "stand"]

    # test
    test_per_label = 20
    test_batch_size = 10
    test_num_rep = 5


# set environment
def set_env(env_file, arg_gpu_ids):
    cfg = Base_Config

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from utils.dir import mkdir
    from data.datautils.babel_label import label as BABEL_label

    # read yaml
    with open(env_file, "r") as f:
        env = yaml.load(f, Loader=yaml.FullLoader)

    for key, value in env.items():
        setattr(cfg, key, value)

    if hasattr(cfg, "embed_dim"):
        cfg.Transformer_spec["embed_dim"] = cfg.embed_dim

    # change to setattr
    # cfg.human_model_path = env["human_model_dir"]
    # cfg.data_dir = env["data_dir"]
    # cfg.num_workers = env["num_workers"]

    if cfg.dev_mode:
        cfg.shuffle = False

    cfg.home = os.path.join(cfg.base, cfg.name)
    cfg.vis_dir = os.path.join(cfg.home, "vis")
    cfg.log_dir = os.path.join(cfg.home, "log")
    cfg.model_out_dir = os.path.join(cfg.home, "model")

    cfg.gpu_ids = arg_gpu_ids
    cfg.gpu_list = [int(i) for i in cfg.gpu_ids.split(",")]
    cfg.num_gpus = len(cfg.gpu_list)

    mkdir(cfg.home)
    mkdir(cfg.vis_dir)
    mkdir(cfg.log_dir)
    mkdir(cfg.model_out_dir)

    # load action profile
    if cfg.action_profiles == "BABEL":
        cfg.action_profiles = BABEL_label
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    import torch

    print(">>> OS SETTING: using gpu ", os.environ["CUDA_VISIBLE_DEVICES"])
    print(">>> HOME DIRECTORY: ", cfg.home)

    print(">>> Num GPUs: ", torch.cuda.device_count())
    assert cfg.num_gpus == torch.cuda.device_count(), "gpu mismatch "

    print(">>> Config file: ", env_file)

    return cfg
