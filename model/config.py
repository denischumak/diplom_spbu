from trimmer.load_trimmer_cfg import load_trim_cfg

SEED = 42

DATA_ROOT = r"C:\Users\User\Desktop\diplom\dataset_collection"
TRIMMER_CFG_PATH = r"C:\Users\User\Desktop\diplom\model\trimmer\trimmer_cfg.json"
ARTIFACTS_PATH = r"C:\Users\User\Desktop\diplom\model\artifacts"
VAL_SIZE = 0.2
TEST_ON = ["vugar"]

SENS_CFG = {
    "n_hall": 3,
    "n_acc": 3,
    "n_gyro": 3,
    "n_quat": 4,
}

PREPROCESSING_CFG = {
    "add_hall_diff": True,
    "min_target_len": 128,
    "static_target_len": 768,
    "use_dynamic_target_len": True,  # if False than pad every sample to static_target_len
    "dynamic_target_len_quantile": 0.98,
}

TCN_CFG = {
    "in_channels": SENS_CFG["n_hall"]
    + (PREPROCESSING_CFG["add_hall_diff"] * SENS_CFG["n_hall"])
    + SENS_CFG["n_gyro"]
    + SENS_CFG["n_acc"]
    + SENS_CFG["n_quat"],
    "hidden": 32,
    "dropout": 0.2,
    "num_blocks": 3,
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "max_epochs": 100,
    "patience": 30,
}
AUGMENT_CFG = {
    "augment_train": True,
    "hall_offset": 0.1,  # after normalization
    "jitter_offset": 0.03,
    "max_crop_ratio": 0.10,
    "p_offset": 0.7,
    "p_jitter": 0.5,
    "p_warp": 0.5,
}

TRIMMER_CFG = load_trim_cfg(TRIMMER_CFG_PATH)
