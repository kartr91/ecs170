"""RNN/LSTM/GRU models sharing one training loop for both Stage 4 tasks."""

import copy
import math
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from local_code.base_class.method import method
from .Dataset_Loaders import PAD, UNK, END, tokenize
from .Evaluate_Generation import token_classification_metrics, TOKEN_METRIC_DEFINITION

# takes a split's arrays and packages them into batches for training
def make_batches(split, batch_size, shuffle=False, seed=42):
    # convert each numpy array into a pytorch tensor
    X = torch.from_numpy(split["X"])
    y = torch.from_numpy(split["y"])
    lengths = torch.from_numpy(split["lengths"])

    dataset = TensorDataset(X, y, lengths)
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )

# builds the model
class Method_RNN(method, nn.Module):
    # embeddings - word-vector matrix from loader
    # task - classification or generation
    # unit - rnn, lstm, or gru
    # hidden_size - size of hidden layers (128 or 256 neurons)
    # num_layers - number of recurrent layers, used 1/2
    # learning rate - 0.001 or 0.005
    # max epochs is 10 for training
    # freeze embeddings controls whether we can change the GloVe word vectors or not

    def __init__(self, embeddings, task="classification", unit="rnn", hidden_size=128,
                 num_layers=1, dropout=0.2, freeze_embeddings=False,
                 learning_rate=0.001, max_epoch=10, batch_size=64, seed=42):
        # initialize parent classes
        method.__init__(self, f"Stage 4 {task} {unit.upper()}")
        nn.Module.__init__(self)
        # save settings
        self.config = dict(task=task, unit=unit, hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout,
                           freeze_embeddings=freeze_embeddings, learning_rate=learning_rate,
                           max_epoch=max_epoch, batch_size=batch_size, seed=seed,
                           embedding_dimension=embeddings.shape[1])
        # turn GloVe matix into a lookup layer, eg if word has id 27, layer retrieves row 27 from the matrix
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embeddings, dtype=torch.float32),
            freeze=freeze_embeddings, padding_idx=0)
        # choose layer to construct
        recurrent_class = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[unit]
        # build recurrent layer; 0 dropout if 1 layer.
        self.recurrent = recurrent_class(
            embeddings.shape[1], hidden_size, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # create prediction layer
        # for classification, produce 2 scores (+ or -),
        # for generation, produce one score per vocabulary word
        self.output = nn.Linear(hidden_size, 2 if task == "classification" else len(embeddings))
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0) if task == "generation" else nn.CrossEntropyLoss()

    # turn word ids into prediction scores
    # X - padded word-id sequences
    # lengths - how many actual words each sequence contains
    def forward(self, X, lengths):
        # read word vectors and return output for each
        outputs, _ = self.recurrent(self.embedding(X))
        if self.config["task"] == "classification":
            # only need 1 prediction per review for classification (+/-)
            outputs = outputs[torch.arange(len(X)), lengths - 1]
        return self.output(self.dropout(outputs))

    # process one pass through the dataset
    def epoch_pass(self, loader, optimizer=None):
        training = optimizer is not None # true if optimizier provided, else false (Eval mode)
        self.train(training) # training/eval mode
        # total_loss - accumulated loss across batches
        # correct - # correct preds
        # count - total # examples processed
        total_loss, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(training):
            for X, y, lengths in loader:
                # find longest sequence in the batch and take its length
                width = int(lengths.max())
                X = X[:, :width]
                if self.config["task"] == "generation":
                    # trim target to mach input
                    y = y[:, :width]
                if training:
                    optimizer.zero_grad() # clear grad from prev batch
                scores = self(X, lengths) # run model on the batch, ie forward, get logits
                if self.config["task"] == "generation":
                    # reshape output/targs so loss function can process individual preds, flatten
                    # s.t. loss function receives 1 row of vocab scores and 1 expected token id per position
                    loss = self.loss_function(scores.reshape(-1, scores.shape[-1]), y.reshape(-1))
                    valid = y.ne(0) # not equal to 0 -> not padding, eg [7,4,0] = [t,t,f]
                    number = int(valid.sum().item()) # valid target tokens
                    correct += int(((scores.argmax(-1) == y) & valid).sum().item()) # count correct token predictions, ignore padding
                else: # classification
                    loss = self.loss_function(scores, y) # calculate loss from class scores/target labels
                    number = len(y) # number of examples for the batch
                    correct += int((scores.argmax(-1) == y).sum().item())
                if training: # perform parameter updates
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0) # scale for exploding gradients
                    optimizer.step() # update model parameters using gradients
                total_loss += loss.item() * number
                count += number
        return {"loss": total_loss / count, "accuracy": correct / count}

    # runs full training process
    def fit(self):
        # build batches from the training set (shuffling enabled)
        train = make_batches(self.data["train"], self.config["batch_size"], True, self.config["seed"])
        validation = make_batches(self.data["validation"], self.config["batch_size"])
        trainable_parameters = []
        for parameter in self.parameters(): # loop over all model params
            if parameter.requires_grad: # check if trainable (some may be frozen)
                trainable_parameters.append(parameter)
        # adam optimizer with prev. defined learning rate
        optimizer = torch.optim.Adam(trainable_parameters, lr=self.config["learning_rate"])
        # history - per-epoch results
        history, best_loss, best_state = [], float("inf"), None
        for epoch in range(1, self.config["max_epoch"] + 1):
            start = time.perf_counter() # for tracking training time
            train_scores = self.epoch_pass(train, optimizer) # training pass
            validation_scores = self.epoch_pass(validation) # val pass
            # dictionary containing useful info about this epoch for later use
            history.append(dict(epoch=epoch, train_loss=train_scores["loss"],
                                train_accuracy=train_scores["accuracy"],
                                validation_loss=validation_scores["loss"],
                                validation_accuracy=validation_scores["accuracy"],
                                seconds=time.perf_counter() - start))

            # progress message - used because training took a while
            print(f"{self.config['unit'].upper()} epoch {epoch}: "
                  f"train loss={train_scores['loss']:.4f}, "
                  f"validation loss={validation_scores['loss']:.4f}", flush=True)
            if validation_scores["loss"] < best_loss:
                best_loss = validation_scores["loss"] # best validation loss
                self.best_epoch = epoch # epoch which produced best validation loss
                best_state = {}
                for key, value in self.state_dict().items():
                    best_state[key] = value.detach().cpu().clone() # store an independent copy of the tensor
        self.load_state_dict(best_state)
        return history

    # returns highest scoring prediction for each input
    @torch.no_grad() # no gradient tracking
    def predict(self, split):
        self.eval() # eval mode
        predictions = []
        for X, _, lengths in make_batches(split, self.config["batch_size"]): # build and loop through batches
            scores = self(X[:, :int(lengths.max())], lengths) # trim batch inputs and run model
            # select highest scoring class/token, move to cpu, convert it to a numpy array and save the predictions
            predictions.append(scores.argmax(-1).cpu().numpy())
        return np.concatenate(predictions)

    # returns actual & predicted token IDs
    @torch.no_grad()
    def predict_tokens(self, split):
        self.eval() # eval mode
        targets, predictions = [], [] # actual, predicted
        for X, y, lengths in make_batches(split, self.config["batch_size"]): # loop through batches
            width = int(lengths.max()) # longest sequence in batch
            scores = self(X[:, :width], lengths) # run model after trimming the inputs
            y = y[:, :width] # trim targets
            valid = y.ne(0)  # non-padding
            targets.append(y[valid].numpy())
            # select highest scoring tokens
            predictions.append(scores.argmax(-1).cpu()[valid].numpy())
        return np.concatenate(targets), np.concatenate(predictions)

    # generate next word based on a 3 word prompt for joke generation
    @torch.no_grad()
    def generate(self, prompt, vocab, max_new_words=40, temperature=0.0):
        # ensure input is valid
        if self.config["task"] != "generation":
            raise ValueError("model must be generation")
        words = tokenize(prompt)
        if len(words) != 3:
            raise ValueError("need exactly 3 tokens to generate")
        if temperature < 0 or max_new_words < 1:
            raise ValueError("invalid settings for generation")
        unknown = []
        ids = []
        for word in words: # turn words into IDs
            if word not in vocab:
                unknown.append(word)
            ids.append(vocab.get(word, vocab[UNK]))
        inverse = {} # reverse lookup for ID->Word for predictions
        for word, index in vocab.items():
            inverse[index] = word
        self.eval() # eval mode
        ended = False # model hasn't predicted end yet; flag
        for _ in range(max_new_words): # predict new word n times
            X = torch.tensor([ids], dtype=torch.long) # tensor of IDs
            scores = self(X, torch.tensor([len(ids)]))[0, -1].clone() # run model, take scores for next word
            # 0 - first sequence in batch, -1 is last sequence
            scores[vocab[PAD]] = scores[vocab[UNK]] = -float("inf") # make PAD/UNK not possible
            next_id = (int(scores.argmax().item()) if temperature == 0 else # choose word with highest score if temp = 0
            # turn scores into probabilities and randomly choose a word using these probabilities, store in next_id
            int(torch.multinomial(torch.softmax(scores / temperature, -1), 1).item()))
            if next_id == vocab[END]:
                ended = True
                break
            ids.append(next_id)
            words.append(inverse[next_id])
        return dict(prompt=prompt, generated=" ".join(words),
                    unknown_prompt_words=unknown, ended=ended)
    # trains model, evaluates it, and saves the results
    def run(self):
        # trains the model - history contains per-epoch training/val results
        history = self.fit()
        # split test set into batches and evaluates the model no them
        scores = self.epoch_pass(make_batches(self.data["test"], self.config["batch_size"]))
        # learned - dictionary of results
        learned = dict(history=history, config=copy.deepcopy(self.config),
                       best_epoch=self.best_epoch,
                       metrics={"test_loss": scores["loss"]})
        if self.config["task"] == "classification":
            # store positive/negative labels for true/predicted
            learned.update(true_y=self.data["test"]["y"], pred_y=self.predict(self.data["test"]))
        else:
            # get true/predicted next-word IDs from the test set
            true_y, pred_y = self.predict_tokens(self.data["test"])
            # store  token IDs
            learned.update(true_y=true_y, pred_y=pred_y)
            # calculate the 4 metrics we're interested in and store the results in metrics
            learned["metrics"].update(token_classification_metrics(true_y, pred_y))
            # record generation metrics
            self.config["generation_evaluation"] = dict(TOKEN_METRIC_DEFINITION)
            # same metrics deepcopy
            learned["config"]["generation_evaluation"] = dict(TOKEN_METRIC_DEFINITION)

            learned["metrics"].update(token_accuracy=scores["accuracy"], # token accuracy (predicted/actual)
                                      perplexity=math.exp(scores["loss"]) if scores["loss"] < 700 else None) # convert test loss into perplexity, upper limit of 700 to prevent exploding exponents
        return learned
