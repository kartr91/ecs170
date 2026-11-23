"""saves all relevant info from experiments"""

import csv
import json
from pathlib import Path

# use this folder as the boundary for saved results
BASE = Path(__file__).resolve().parent
import numpy as np
import torch

from local_code.base_class.result import result
from .Evaluate_Generation import generation_examples


def inside_folder(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE):
        raise ValueError(f"Output must be inside {BASE}")
    return path


def json_value(value):
    # convert numpy values into types that json can store
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path, value):
    with inside_folder(path).open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=json_value, allow_nan=False)


class Result_Saver(result):
    def __init__(self, output_folder, generation_options=None):
        super().__init__("Stage 4 experiment artifacts", "")
        self.output_folder = inside_folder(output_folder)
        self.generation_options = generation_options or {}

    def save(self):
        folder = self.output_folder
        folder.mkdir(parents=True, exist_ok=True)
        data = self.data
        # save the model settings and final test metrics
        summary = {}
        for key in ("config", "metrics", "best_epoch"):
            summary[key] = data[key]
        write_json(folder / "summary.json", summary)
        # keep each epoch so the report can plot training convergence
        with (folder / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data["history"][0]))
            writer.writeheader()
            writer.writerows(data["history"])
        if data["config"]["task"] == "generation":
            # save generated stories and their comparison with training text
            examples = generation_examples(self.model, self.model.data, **self.generation_options)
            write_json(folder / "generated_jokes.json", examples)
        # keep the trained weights so the model can be used again
        state_dict = {}
        for key, value in self.model.state_dict().items():
            state_dict[key] = value.detach().cpu()
        checkpoint = {
            "state_dict": state_dict,
            "config": data["config"], "vocab": self.model.data["vocab"],
            "best_epoch": data["best_epoch"],
        }
        torch.save(checkpoint, folder / "best_model.pt")
        print(f"Saved {folder}", flush=True)
