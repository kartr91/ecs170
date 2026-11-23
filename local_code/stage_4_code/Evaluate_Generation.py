"""evaluate next token predictions and generated text"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


TOKEN_METRIC_DEFINITION = {
    "version": 1,
    "prediction": "teacher-forced next-token argmax on the held-out test set",
    "ignored_target": "PAD (token ID 0)",
    "macro_labels": "union of observed target and predicted token IDs",
    "zero_division": 0,
    "special_tokens": "UNK and END are included",
}


def token_classification_metrics(true_y, pred_y, pad_id=0):
    #measure predictions at positions with a real target token
    true_y = np.asarray(true_y)
    pred_y = np.asarray(pred_y)
    if true_y.shape != pred_y.shape:
        raise ValueError("True and predicted token arrays must have matching shapes")

    # ignore padding because it is not a word to predict
    real_tokens = true_y != pad_id
    true_y = true_y[real_tokens]
    pred_y = pred_y[real_tokens]
    if not true_y.size:
        raise ValueError("No non-padding target tokens to evaluate")

    # include every token seen in either targets or predictions
    labels = np.union1d(true_y, pred_y)
    accuracy = float(accuracy_score(true_y, pred_y))
    result = dict(accuracy=accuracy, token_accuracy=accuracy, evaluated_tokens=int(true_y.size))

    # calculate both macro and frequency weighted scores
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_y, pred_y, labels=labels, average=average, zero_division=0)
        result.update({f"precision_{average}": float(precision),
                       f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    return result


def generation_examples(model, data, prompts=(), max_new_words=40, temperature=0.0):
    # remember the training text so exact copies can be identified
    training = set()
    for words in data["text"]["train"]:
        training.add(" ".join(words))

    # use three prompts from each split and any custom prompts
    requests = []
    for split in ("train", "test"):
        for words in data["text"][split][:3]:
            requests.append((split, " ".join(words[:3]), " ".join(words)))
    for prompt in prompts:
        requests.append(("custom", prompt, None))

    # generate text and record its source and reference
    examples = []
    for split, prompt, reference in requests:
        row = model.generate(prompt, data["vocab"], max_new_words, temperature)
        row.update(source_split=split, reference=reference,
                   exact_training_match=row["generated"] in training)
        examples.append(row)
    return examples
