"""Shared fixtures: a tiny random DistilBERT classifier saved like the real one, no download."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edge_inference_optimization import env

env.configure()

import torch  # noqa: E402  (after env.configure, so caches stay in .tmp)
from transformers import DistilBertConfig, DistilBertForSequenceClassification  # noqa: E402

from edge_inference_optimization.data import Example  # noqa: E402

WORDS = ["good", "bad", "great", "awful", "movie", "film", "the", "a", "not", "very", "fun", "dull", "!", "."]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write config, weights, vocabulary and tokenizer config for a 2-layer, 64-wide model."""
    d = tmp_path_factory.mktemp("tiny")
    torch.manual_seed(0)
    cfg = DistilBertConfig(vocab_size=5 + len(WORDS), dim=64, n_layers=2, n_heads=2, hidden_dim=128, max_position_embeddings=64)
    DistilBertForSequenceClassification(cfg).save_pretrained(d)
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]) + "\n", encoding="utf-8")
    (d / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64, "do_lower_case": True}), encoding="utf-8")
    return d


@pytest.fixture
def examples() -> list[Example]:
    """Twenty short labelled sentences."""
    pos = ["good movie", "great film !", "very fun", "a good film", "the movie is great"]
    neg = ["bad movie", "awful film .", "very dull", "not fun", "the film is bad"]
    return [Example(t, 1) for t in pos * 2] + [Example(t, 0) for t in neg * 2]
