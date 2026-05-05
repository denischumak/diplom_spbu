from autotrimmer_find_params import *


if __name__ == "__main__":
    dataset = load_trim_dataset(r"C:\Users\User\Desktop\diplom\dataset_collection")
    cfg = load_trim_cfg(r"C:\Users\User\Desktop\diplom\model\trimmer\trimmer_cfg.json")
    metrics = evaluate_config(dataset, cfg)
    print(metrics, "total score over all samples:", getScore(metrics))