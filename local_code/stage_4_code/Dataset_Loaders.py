"""Review/joke dataset loading and preparation + load GloVe"""

import csv
import random
import string
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset
"""
constants representing special tokens
PAD - padding
UNK - unknown (not in vocabulary)
END - end of a joke
"""
PAD, UNK, END = "<PAD>", "<UNK>", "<END>"

# Uses HTML Parser as a parent class to parse html text in IMDB reviews
class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, text):
        self.parts.append(text)

    def handle_starttag(self, tag, attrs):
        self.parts.append(" ")

    def handle_endtag(self, tag):
        self.parts.append(" ")

# Remove HTML, lowercase, and replace punctuation
def tokenize(text):
    parser = _TextParser()
    parser.feed(text) # read text
    parser.close()
    # combine pieces into string, convert to lowercase, change apostrophes
    text = "".join(parser.parts).lower().replace("’", "'")
    # replace punctuation
    punctuation = string.punctuation.replace("'", "") + "“”‘—–…"
    replacements = {}
    for char in punctuation:
        replacements[char] = " "
    # apply replacements
    text = text.translate(str.maketrans(replacements))
    # split string into words
    words = []
    for token in text.split():
        word = token.strip("'")
        if word:
            words.append(word)
    # return tokens
    return words

# map words to IDs
def build_vocabulary(token_lists, maximum):
    counts = Counter()
    # count how often each word appears
    for words in token_lists:
        for word in words:
            counts[word] += 1
    vocab = {PAD: 0, UNK: 1, END: 2} # assign ids to special tokens
    # add the most common words to the vocabulary
    for word, _ in counts.most_common(maximum - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

# turns tokenized reviews into numerical arrays
# examples - (list, 0/1 for positive/negative)
def encode_reviews(examples, vocab, max_length):
    # resultant X/y arrays
    X = np.zeros((len(examples), max_length), dtype=np.int64)
    y = np.empty(len(examples), dtype=np.int64)
    lengths = np.empty(len(examples), dtype=np.int64)
    #visit each review/label
    for index, (words, label) in enumerate(examples):
        #convert words to vocacbulary ids
        ids = []
        for word in words[:max_length]:
            ids.append(vocab.get(word, vocab[UNK]))
        ids = ids or [vocab[UNK]] # if vocab doesn't contain the word, use unknown id
        X[index, :len(ids)] = ids
        y[index], lengths[index] = label, len(ids) # save label/sequence length
    return {"X": X, "y": y, "lengths": lengths}

# Handle loading reviews
class Review_Dataset_Loader(dataset):
    # max vocab size is 20,000 words, max of 300 tokens per review, 10% of training reviews
    # reserved for validation
    def __init__(self, directory, vocab_size=20000, max_length=300,
                 validation_fraction=0.1, seed=42, limit_per_class=None):
        super().__init__("IMDb", "IMDB review classification")
        self.directory = Path(directory)
        self.vocab_size, self.max_length = vocab_size, max_length
        self.validation_fraction, self.seed = validation_fraction, seed
        self.limit_per_class = limit_per_class

    def load(self):
        rng = random.Random(self.seed)
        # empty group lists
        groups = {"train": [], "validation": [], "test": []}
        for split in ("train", "test"): # cycle through training/testing sets/folders
            for folder, label in (("neg", 0), ("pos", 1)): # cycle through positive/negative review folders
                files = sorted((self.directory / split / folder).glob("*.txt")) # find text files
                rng.shuffle(files) # shuffle the text files
                if self.limit_per_class is not None:
                    files = files[:self.limit_per_class]
                examples = []
                for path in files: # read and tokenize each review
                    words = tokenize(path.read_text(encoding="utf-8"))
                    examples.append((words, label))
                # separate training/validation/testing examples
                if split == "train":
                    count = max(1, min(len(examples) - 1,
                                       int(len(examples) * self.validation_fraction)))
                    groups["validation"].extend(examples[:count])
                    groups["train"].extend(examples[count:])
                else:
                    groups["test"].extend(examples)
        training_words = []
        for words, _ in groups["train"]:
            training_words.append(words)

        # build vocabulary using the training words
        self.vocab = build_vocabulary(training_words, self.vocab_size)
        data = {"vocab": self.vocab}
        # encode each of the 3 groups
        for split, examples in groups.items():
            rng.shuffle(examples)
            data[split] = encode_reviews(examples, self.vocab, self.max_length)
        return data

# Handle loading jokes and prep for next-token generation
class Joke_Dataset_Loader(dataset):
    # max vocab size - 10,000 words
    # max token length per joke - 80
    def __init__(self, filename, vocab_size=10000, max_length=80, seed=42, limit=None):
        super().__init__("Short jokes", "Next-word prediction")
        self.filename, self.vocab_size = Path(filename), vocab_size
        self.max_length, self.seed, self.limit = max_length, seed, limit

    # turn each joke into input words & the next word the model should predict
    def encode(self, examples):
        # creates one joke per row
        X = np.zeros((len(examples), self.max_length), dtype=np.int64)
        y = np.zeros_like(X)
        lengths = np.empty(len(examples), dtype=np.int64)
        for index, words in enumerate(examples): # visit each joke
            ids = []
            for word in words: # turn words into IDs
                ids.append(self.vocab.get(word, self.vocab[UNK]))
            ids.append(self.vocab[END])
            # shift sequence by one slot
            inputs = ids[:-1][:self.max_length]
            targets = ids[1:][:self.max_length]
            X[index, :len(inputs)] = inputs
            y[index, :len(targets)] = targets
            lengths[index] = len(inputs)
        return {"X": X, "y": y, "lengths": lengths}

    def load(self):
        # read csv
        with self.filename.open(encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            # clean jokes and remove duplicates
            examples = []
            seen = set()
            for row in reader:
                words = tokenize(row["Joke"] or "")
                if len(words) >= 3:
                    joke = tuple(words)
                    if joke not in seen:
                        examples.append(joke)
                        seen.add(joke)
        random.Random(self.seed).shuffle(examples) # shuffle jokes
        if self.limit is not None:
            examples = examples[:self.limit]
        count = max(1, int(len(examples) * 0.1))
        # split into test/validation/training groups
        groups = {"test": examples[:count], "validation": examples[count:2 * count],
                  "train": examples[2 * count:]}
        # build word-id dictionary
        self.vocab = build_vocabulary(groups["train"], self.vocab_size)
        data = {}
        # encode each of the 3 groups
        for split, rows in groups.items():
            data[split] = self.encode(rows)
        data.update(vocab=self.vocab, text=groups)
        return data

# builds a table of word vectors from GloVe
def load_embeddings(filename, vocab, dimension=100, seed=42):
    # one row per vocab word, random distribution with mean 0 and stdev 0.05
    matrix = np.random.default_rng(seed).normal(0, 0.05, (len(vocab), dimension)).astype(np.float32)
    matrix[vocab[PAD]] = 0
    wanted = set(vocab) - {PAD, UNK, END} # vocab words to find in GloVe
    matched = set() # empty set to store matches
    if filename is not None: # make sure file was given
        with Path(filename).open(encoding="utf-8") as handle:
            for line in handle: # read one line at a time
                word, separator, values = line.rstrip().partition(" ")
                if separator and word in wanted and word not in matched:
                    # convert numbers from text->numpy vector
                    vector = np.fromstring(values, sep=" ", dtype=np.float32)
                    matrix[vocab[word]] = vector
                    # record matches
                    matched.add(word)
                    if len(matched) == len(wanted):
                        # if all wanted words had matches, stop reading
                        break
    # returns both the vector matrix and a brief summary of where it came from
    return matrix, {"source": str(filename) if filename else "random",
                    "matched": len(matched), "eligible": len(wanted),
                    "coverage": len(matched) / max(1, len(wanted)), "dimension": dimension}
