"""Compare three classifiers on the scikit-learn handwritten digits dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
DIGITS = list(range(10))


def classifiers():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1500)
        ),
        "random_forest": RandomForestClassifier(n_estimators=150, random_state=SEED),
        "rbf_svm": make_pipeline(StandardScaler(), SVC(C=8, gamma="scale")),
    }


def dataset_split():
    features, targets = load_digits(return_X_y=True)
    return train_test_split(
        features, targets, test_size=0.25, random_state=SEED, stratify=targets
    )


def benchmark(split, output_dir):
    train_x, test_x, train_y, test_y = split
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = classifiers()
    cv_scores = {
        name: cross_val_score(model, train_x, train_y, cv=folds, scoring="f1_macro")
        for name, model in models.items()
    }
    winner = max(cv_scores, key=lambda name: cv_scores[name].mean())

    results = {}
    for name, model in models.items():
        predicted = model.fit(train_x, train_y).predict(test_x)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_y, predicted, average="macro", zero_division=0
        )
        results[name] = {
            "cv_macro_f1_mean": round(float(cv_scores[name].mean()), 4),
            "cv_macro_f1_sd": round(float(cv_scores[name].std()), 4),
            "test_accuracy": round(accuracy_score(test_y, predicted), 4),
            "test_macro_precision": round(float(precision), 4),
            "test_macro_recall": round(float(recall), 4),
            "test_macro_f1": round(float(f1), 4),
            "confusion_matrix": confusion_matrix(
                test_y, predicted, labels=DIGITS
            ).tolist(),
        }

    report = {
        "seed": SEED,
        "training_images": len(train_y),
        "test_images": len(test_y),
        "selected_by_training_cv": winner,
        "models": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    save_confusion_plot(
        results[winner]["confusion_matrix"], winner, output_dir / "confusion_matrix.png"
    )
    return report


def save_confusion_plot(matrix, name, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(figsize=(7, 6), layout="constrained")
    display = ConfusionMatrixDisplay(np.asarray(matrix), display_labels=DIGITS)
    display.plot(ax=axes, cmap="Blues", colorbar=False)
    axes.set_title(f"{name.replace('_', ' ').upper()} · Held-out test images")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def read_pixels(path):
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    try:
        pixels = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("Pixel input must contain integers") from exc
    if len(pixels) != 64 or any(pixel < 0 or pixel > 16 for pixel in pixels):
        raise ValueError("Supply 64 pixel values from 0 to 16 in row-major order")
    return np.asarray(pixels)


def predict(split, model_name, index, pixel_file):
    train_x, test_x, train_y, test_y = split
    result = {"model": model_name}
    if index is not None:
        if not 0 <= index < len(test_y):
            raise ValueError(f"Test index must be between 0 and {len(test_y) - 1}")
        pixels = test_x[index]
        result.update(test_index=index, actual_digit=int(test_y[index]))
    else:
        pixels = read_pixels(pixel_file)
    model = classifiers()[model_name].fit(train_x, train_y)
    result["predicted_digit"] = int(model.predict([pixels])[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("benchmark", help="Evaluate models and save reports")
    run.add_argument("--out", type=Path, default=Path("artifacts"))
    classify = commands.add_parser("predict", help="Classify an 8 by 8 image")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", type=int, help="Index in the held-out split")
    source.add_argument("--pixels", type=Path, help="Text file of 64 integers")
    classify.add_argument("--model", choices=list(classifiers()), default="rbf_svm")
    args = parser.parse_args()

    try:
        split = dataset_split()
        if args.command == "benchmark":
            result = benchmark(split, args.out)
            print(f"Saved report.json and confusion_matrix.png to {args.out}")
            print(f"Selected by cross-validation: {result['selected_by_training_cv']}")
            for name, metrics in result["models"].items():
                print(f"{name:22} test accuracy {metrics['test_accuracy']:.2%}")
        else:
            print(json.dumps(predict(split, args.model, args.index, args.pixels), indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
