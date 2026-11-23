"""build stage 4 report figures and metrics from saved runs"""

import argparse
import csv
import json
import matplotlib
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
BASE = PROJECT / "local_code" / "stage_4_code"

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

COLORS = {"rnn": "#84b2d2", "lstm": "#305577", "gru": "#ad493c"}
UNITS = ("rnn", "lstm", "gru")
TASKS = ("classification", "generation")
VARIANTS = ("baseline", "hidden_256", "two_layers", "learning_rate_0005", "frozen_embeddings")
VARIANT_LABELS = {"baseline": "Baseline", "hidden_256": "Hidden\n256",
                  "two_layers": "Two\nlayers", "learning_rate_0005": "Learning rate\n0.0005",
                  "frozen_embeddings": "Frozen\nembeddings"}
METRIC_COLUMNS = ("variant", "task", "unit", "hidden_size", "num_layers", "dropout",
                  "learning_rate", "freeze_embeddings", "max_epoch", "batch_size", "seed",
                  "best_epoch", "validation_loss", "validation_accuracy", "training_seconds",
                  "test_loss", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "precision_weighted", "recall_weighted", "f1_weighted", "token_accuracy",
                  "perplexity", "evaluated_tokens")


def read_runs(root):
    # read the saved metrics and training history for each model
    runs = []
    for filename in sorted(Path(root).rglob("summary.json")):
        folder = filename.parent
        if not (folder / "history.csv").is_file():
            continue
        summary = json.loads(filename.read_text())
        config = summary["config"]
        if config.get("task") not in TASKS or config.get("unit") not in UNITS:
            continue
        # find the setting used for this run from its folder name
        parts = filename.relative_to(root).parts[:-1]
        variant = "baseline"
        for part in parts:
            if part in VARIANTS:
                variant = part
                break
        with (folder / "history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        # convert saved text values into numbers for plotting
        for row in history:
            row["epoch"] = int(row["epoch"])
            for key in ("train_loss", "validation_loss", "validation_accuracy", "seconds"):
                row[key] = float(row[key])
        best = None
        for row in history:
            if row["epoch"] == summary["best_epoch"]:
                best = row
                break
        # keep each model configuration metrics and history together
        runs.append(dict(variant=variant, task=config["task"], unit=config["unit"],
                         config=config, metrics=summary["metrics"],
                         history=history, best=best))
    return runs


def save_figure(figure, folder, name):
    figure.tight_layout()
    figure.savefig(folder / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def baseline_runs(runs, task):
    selected = []
    for run in runs:
        if run["task"] == task and run["variant"] == "baseline":
            selected.append(run)
    return sorted(selected, key=lambda run: UNITS.index(run["unit"]))


def tasks_with_baselines(runs):
    tasks = []
    for task in TASKS:
        if baseline_runs(runs, task):
            tasks.append(task)
    return tasks


def convergence(runs, folder):
    # compare training and validation loss across the baseline models
    available = tasks_with_baselines(runs)
    if not available:
        return False
    figure, axes = plt.subplots(1, len(available), figsize=(4.8 * len(available), 3.4), squeeze=False)
    # make one panel for each task with a line for every model and data split
    for axis, task in zip(axes[0], available):
        selected = baseline_runs(runs, task)
        for run in selected:
            epochs = []
            for row in run["history"]:
                epochs.append(row["epoch"])
            for split, style in (("train", "-"), ("validation", "--")):
                losses = []
                for row in run["history"]:
                    losses.append(row[f"{split}_loss"])
                axis.plot(epochs, losses,
                          color=COLORS[run["unit"]], linestyle=style,
                          marker="o" if len(epochs) == 1 else None,
                          label=f"{run['unit'].upper()} {split}")
        axis.set(title=task.capitalize(), xlabel="Epoch", ylabel="Cross-entropy loss")
        epoch_numbers = set()
        for run in selected:
            for row in run["history"]:
                epoch_numbers.add(row["epoch"])
        all_epochs = sorted(epoch_numbers)
        if len(all_epochs) == 1:
            axis.set_xticks(all_epochs)
        else:
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    save_figure(figure, folder, "training_convergence")
    return True


def percentage_axis(axis, label):
    axis.set_ylabel(label)
    axis.set_ylim(0, 1.12)
    axis.yaxis.set_major_formatter(PercentFormatter(1))


def bars(axis, names, values, colors):
    plotted = axis.bar(names, values, color=colors)
    labels = []
    for value in values:
        labels.append(f"{value:.1%}")
    axis.bar_label(plotted, labels=labels, fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)


def performance(runs, folder):
    # compare test metrics across the baseline models
    available = tasks_with_baselines(runs)
    if not available:
        return False
    metrics = (("accuracy", "Accuracy"), ("precision_macro", "Macro precision"),
               ("recall_macro", "Macro recall"), ("f1_macro", "Macro F1"))
    figure, axes = plt.subplots(len(available), 4, figsize=(12, 3.2 * len(available)), squeeze=False)
    # give each task a row and each metric a column
    for row, task in enumerate(available):
        selected = baseline_runs(runs, task)
        for axis, (metric, label) in zip(axes[row], metrics):
            values = []
            names = []
            colors = []
            for run in selected:
                values.append(run["metrics"][metric])
                names.append(run["unit"].upper())
                colors.append(COLORS[run["unit"]])
            plotted = axis.bar(names, values, color=colors)
            labels = []
            for value in values:
                labels.append(f"{value:.2%}")
            axis.bar_label(plotted, labels=labels, fontsize=8)
            axis.set_ylim(0, min(1.12, max(values) * 1.3 or 1))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.set_title(f"{task.capitalize()}\n{label}", fontsize=10)
            axis.set_ylabel("Held-out score")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
    save_figure(figure, folder, "model_performance")
    return True


def ablations(runs, folder, unit):
    # compare each setting with the baseline for the selected model type
    groups = {}
    for task in TASKS:
        selected = []
        for run in runs:
            if run["task"] == task and run["unit"] == unit:
                selected.append(run)
        groups[task] = sorted(selected, key=lambda run: VARIANTS.index(run["variant"]))
    available = []
    for task, group in groups.items():
        has_baseline = False
        for run in group:
            if run["variant"] == "baseline":
                has_baseline = True
                break
        if has_baseline and len(group) > 1:
            available.append(task)
    if not available:
        return False
    # plot the best validation accuracy for each setting
    figure, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 3.5), squeeze=False)
    for axis, task in zip(axes[0], available):
        selected = groups[task]
        values = []
        names = []
        colors = []
        for run in selected:
            values.append(run["best"]["validation_accuracy"])
            names.append(VARIANT_LABELS[run["variant"]])
            colors.append("#84b2d2" if run["variant"] == "baseline" else "#305577")
        bars(axis, names, values, colors)
        percentage_axis(axis, "Validation accuracy" if task == "classification" else "Validation next-token accuracy")
        axis.set_title(f"{task.capitalize()}: {unit.upper()} settings")
        axis.tick_params(axis="x", labelsize=8)
    save_figure(figure, folder, "ablation_comparison")
    return True


def save_metrics(runs, folder):
    # combine model settings validation results and test results in one table
    with (folder / "all_experiments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for run in runs:
            row = {"variant": run["variant"], "task": run["task"], "unit": run["unit"],
                   "best_epoch": run["best"]["epoch"],
                   "validation_loss": run["best"]["validation_loss"],
                   "validation_accuracy": run["best"]["validation_accuracy"]}
            seconds = 0.0
            for epoch in run["history"]:
                seconds += epoch["seconds"]
            row["training_seconds"] = seconds
            for key in METRIC_COLUMNS:
                if key in run["config"]:
                    row[key] = run["config"][key]
                if key in run["metrics"]:
                    row[key] = run["metrics"][key]
            writer.writerow(row)


def make_report_figures(results, output=None, ablation_unit="gru"):
    # write the metrics table and the three report figures
    results = Path(results).resolve()
    output = Path(output or results / "report_figures").resolve()
    runs = read_runs(results)
    output.mkdir(parents=True, exist_ok=True)
    save_metrics(runs, output)
    # create each figure only when the required runs are available
    count = 0
    if convergence(runs, output):
        count += 1
    if performance(runs, output):
        count += 1
    if ablations(runs, output, ablation_unit):
        count += 1
    print(f"Saved metrics for {len(runs)} runs and {count} figures in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BASE / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ablation-unit", choices=UNITS, default="gru")
    args = parser.parse_args()
    make_report_figures(args.results, args.output, args.ablation_unit)


if __name__ == "__main__":
    main()
