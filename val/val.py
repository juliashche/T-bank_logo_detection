from pathlib import Path
from ultralytics import YOLO

def validate(weights: str, data_yaml: str, save_dir: str = "runs/val_results"):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)

    results = model.val(
        data=str(data_yaml),
        save=True,             # сохранение изображений с bbox
        save_txt=False,
        project=save_dir,
        name="val_results",
        exist_ok=True
    )

    print("\nМетрики на валидационной выборке")
    results.summary()
    metrics_dict = results.results_dict
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Валидация YOLO на размеченной выборке")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, default="runs/val_results")
    args = parser.parse_args()

    validate(args.weights, args.data, args.out)
