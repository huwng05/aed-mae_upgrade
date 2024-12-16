import ml_collections


def get_configs_avenue():
    config = ml_collections.ConfigDict()
    config.batch_size = 8
    config.epochs = 200
    config.mask_ratio = 0.5
    config.start_TS_epoch = 78
    config.masking_method = "random_masking"
    config.output_dir = "../experiments/ConViT"  # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L2'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (640, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = 'train'
    config.resume = True
    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "avenue"
    config.avenue_path = "../datasets/avenue"
    config.avenue_gt_path = "../datasets/avenue/gt_avenue"
    config.percent_abnormal = 0.25
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 3
    config.pin_mem = False

    return config


def get_configs_shanghai():
    config = ml_collections.ConfigDict()
    config.batch_size = 100
    config.epochs = 50
    config.mask_ratio = 0.5
    config.start_TS_epoch = 100
    config.masking_method = "random_masking"
    config.output_dir = "github_ckpt/shanghai" # the checkpoints will be loaded from here
    config.abnormal_score_func = 'L1'
    config.grad_weighted_rec_loss = True
    config.model = "mae_cvt"
    config.input_size = (320, 640)
    config.norm_pix_loss = False
    config.use_only_masked_tokens_ab = False
    config.run_type = "inference"
    config.resume=False

    # Optimizer parameters
    config.weight_decay = 0.05
    config.lr = 1e-4

    # Dataset parameters
    config.dataset = "shanghai"
    config.shanghai_path = "/home/alin/datasets/SanhaiTech"
    config.shanghai_gt_path = "/media/alin/hdd/Transformer_Labels/Shanghai_gt"
    config.percent_abnormal = 0.5
    config.input_3d = True
    config.device = "cuda"

    config.start_epoch = 0
    config.print_freq = 10
    config.num_workers = 10
    config.pin_mem = False

    return config
