"""train and evaluate one stage 4 model"""

import sys

import argparse
from pathlib import Path

# project root for local imports later
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

import torch

from local_code.stage_4_code.Dataset_Loaders import (
    Joke_Dataset_Loader,
    Review_Dataset_Loader,
    load_embeddings,
    tokenize,
)
from local_code.stage_4_code.Evaluate_Classification import Evaluate_Classification
from local_code.stage_4_code.Method_RNN import Method_RNN
from local_code.stage_4_code.Result_Saver import Result_Saver, inside_folder
from local_code.stage_4_code.Setting_Provided_Split import Setting_Provided_Split


def main():
    # read the model and training settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("classification", "generation"), required=True)
    parser.add_argument("--unit", choices=("rnn", "lstm", "gru"), required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Variant folder; task/unit are appended")
    parser.add_argument("--glove", type=Path, help="GloVe file; auto-detected if omitted")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--prompt", action="append", help="Three starting words")
    args = parser.parse_args()

    # give each task + model type its own folder under results
    run_folder = inside_folder(args.output / args.task / args.unit)
    if run_folder.exists() and any(run_folder.iterdir()):
        raise FileExistsError(f"Results already exist: {run_folder}. Select a new --output folder.")

    generation_options = None
    if args.task == "generation":
        # check prompts before starting run
        prompts = args.prompt if args.prompt is not None else ["what did the", "why did the"]
        for prompt in prompts:
            if len(tokenize(prompt)) != 3:
                raise ValueError(f"Prompt must contain exactly three tokens: {prompt!r}")
        generation_options = dict(prompts=prompts, max_new_words=40, temperature=0.0)

    # load the dataset and glove vectors
    seed = 42
    data_root = PROJECT / "data" / "stage_4_data"
    glove_file = args.glove
    if glove_file is None:
        # find the single glove file if no path was supplied
        candidates = sorted(data_root.glob("*.txt")) + sorted((data_root / "glove").glob("*.txt"))
        if len(candidates) != 1:
            raise ValueError("Use --glove to select one embedding file")
        glove_file = candidates[0]
    if args.task == "classification":
        loader = Review_Dataset_Loader(data_root / "text_classification", seed=seed)
    else:
        loader = Joke_Dataset_Loader(data_root / "text_generation" / "data", seed=seed)
    data = loader.load()
    print(f"{args.task}: train={len(data['train']['X'])}, "
          f"validation={len(data['validation']['X'])}, test={len(data['test']['X'])}", flush=True)
    print("Loading embeddings...", flush=True)
    embeddings, embedding_info = load_embeddings(glove_file, data["vocab"], seed=seed)
    print(f"GloVe coverage: {embedding_info['matched']}/{embedding_info['eligible']}", flush=True)

    # seed the model before training
    torch.manual_seed(seed)
    model = Method_RNN(
        embeddings,
        task=args.task,
        unit=args.unit,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        freeze_embeddings=args.freeze_embeddings,
        learning_rate=args.learning_rate,
        max_epoch=args.epochs,
        batch_size=args.batch_size,
        seed=seed,
    )
    # record data sizes w/ the model settings
    split_sizes = {}
    for split in ("train", "validation", "test"):
        split_sizes[split] = len(data[split]["X"])
    model.config.update(embeddings=embedding_info,
                        max_length=loader.max_length, vocab_size=len(data["vocab"]),
                        split_sizes=split_sizes)
    if generation_options is not None:
        model.config["generation_options"] = generation_options

    # save the training eval
    saver = Result_Saver(run_folder, generation_options)
    setting = Setting_Provided_Split()
    evaluator = Evaluate_Classification() if args.task == "classification" else None
    setting.prepare(data, model, saver, evaluator)
    setting.load_run_save_evaluate()


if __name__ == "__main__":
    main()
