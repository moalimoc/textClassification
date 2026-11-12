"""Classify a short review as positive or negative.

The CSV holds 400 reviews written for this exercise. label 1 is positive
and label 0 is negative. Half of them say the obvious word ("great",
"awful"). The other half flip the word with "not", "never", or "hardly":
"not awful" is praise, and "hardly worth the ticket" is a complaint.

A linear bag of words gives each word one weight, so "not" can only push
every review the same direction. The network averages word vectors and
then passes that average through a hidden layer, which can treat "not
awful" differently from "awful".
"""

import csv
import re
from pathlib import Path

import torch
from torch import nn

DATA_PATH = Path(__file__).with_name("reviews.csv")
UNK = "<unk>"
NEGATION = {"not", "never", "hardly"}
TOKEN = re.compile(r"[a-z]+")

SAMPLES = [
    "the soup was not bad",
    "the film is hardly worth the ticket",
    "i loved the ramen",
    "a dreadful hotel from start to finish",
    "The Imitation Game is a well-made, emotionally resonant prestige biopic and WWII thriller starring Benedict Cumberbatch as Alan Turing, earning generally favorable reviews for its immersive tension and outstanding performance despite fitting into a common awards-season genre.",
]


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def load_reviews(path: Path) -> tuple[list[str], torch.Tensor]:
    texts = []
    labels = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, torch.tensor(labels, dtype=torch.float32)


def build_vocab(texts: list[str]) -> dict[str, int]:
    vocab = {UNK: 0}
    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode(texts: list[str], vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    indices = []
    offsets = []
    cursor = 0
    for text in texts:
        offsets.append(cursor)
        tokens = tokenize(text) or [UNK]
        indices.extend(vocab.get(token, vocab[UNK]) for token in tokens)
        cursor += len(tokens)
    return torch.tensor(indices), torch.tensor(offsets)


def bow_matrix(texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    rows = torch.zeros(len(texts), len(vocab))
    for row, text in enumerate(texts):
        tokens = tokenize(text) or [UNK]
        for token in tokens:
            rows[row, vocab.get(token, vocab[UNK])] += 1.0
        rows[row] /= rows[row].sum()
    return rows


class ReviewNet(nn.Module):
    """Averaged word vectors, then a small hidden layer, then one logit."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, 32, mode="mean")
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(tokens, offsets)).squeeze(-1)


def train_embed(
    model: ReviewNet,
    tokens: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, 401):
        logits = model(tokens, offsets)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch {epoch:4d}   train accuracy {accuracy(logits, labels):.1%}")


def train_linear(model: nn.Linear, bows: torch.Tensor, labels: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(1, 401):
        logits = model(bows).squeeze(-1)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 100 == 0:
            print(f"  epoch {epoch:4d}   train accuracy {accuracy(logits, labels):.1%}")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (logits >= 0).float()
    return (predictions == labels).float().mean().item()


def subset_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> float:
    if not mask.any():
        return float("nan")
    return accuracy(logits[mask], labels[mask])


def negation_mask(texts: list[str]) -> torch.Tensor:
    flags = [bool(NEGATION.intersection(tokenize(text))) for text in texts]
    return torch.tensor(flags)


def main() -> None:
    torch.manual_seed(0)
    texts, labels = load_reviews(DATA_PATH)
    order = torch.randperm(len(texts))
    texts = [texts[i] for i in order]
    labels = labels[order]

    split = len(texts) - 80
    train_texts, test_texts = texts[:split], texts[split:]
    train_labels, test_labels = labels[:split], labels[split:]
    vocab = build_vocab(train_texts)

    train_tokens, train_offsets = encode(train_texts, vocab)
    test_tokens, test_offsets = encode(test_texts, vocab)
    train_bow = bow_matrix(train_texts, vocab)
    test_bow = bow_matrix(test_texts, vocab)
    test_negation = negation_mask(test_texts)

    print(f"loaded {len(texts)} reviews from {DATA_PATH.name}")
    print(f"vocabulary {len(vocab)} words, from the training reviews only")
    print(f"training on {len(train_texts)}, testing on {len(test_texts)}\n")

    print("linear bag of words (one weight per word)")
    linear = nn.Linear(len(vocab), 1)
    train_linear(linear, train_bow, train_labels)

    print("\nembedding network")
    net = ReviewNet(len(vocab))
    train_embed(net, train_tokens, train_offsets, train_labels)

    linear.eval()
    net.eval()
    with torch.no_grad():
        linear_test = linear(test_bow).squeeze(-1)
        net_test = net(test_tokens, test_offsets)

    print("\nheld-out reviews")
    print(f"  linear accuracy   {accuracy(linear_test, test_labels):.1%}")
    print(f"  network accuracy  {accuracy(net_test, test_labels):.1%}")
    print("held-out reviews that use not / never / hardly")
    print(f"  linear accuracy   {subset_accuracy(linear_test, test_labels, test_negation):.1%}")
    print(f"  network accuracy  {subset_accuracy(net_test, test_labels, test_negation):.1%}")

    print("\nnew reviews")
    sample_tokens, sample_offsets = encode(SAMPLES, vocab)
    with torch.no_grad():
        sample_logits = net(sample_tokens, sample_offsets)
    for text, logit in zip(SAMPLES, sample_logits):
        label = "positive" if logit >= 0 else "negative"
        print(f"  {label:8}  {text}")


if __name__ == "__main__":
    main()
