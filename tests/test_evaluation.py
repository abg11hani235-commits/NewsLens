"""Unit tests for evaluation and ROUGE calculation module."""

import pytest
from src.evaluation import compute_rouge_metrics


def test_compute_rouge_identical():
    preds = ["The international criminal court investigates crimes."]
    refs = ["The international criminal court investigates crimes."]
    scores = compute_rouge_metrics(preds, refs)
    assert scores["rouge1"]["f1"] == 100.0
    assert scores["rouge2"]["f1"] == 100.0
    assert scores["rougeL"]["f1"] == 100.0


def test_compute_rouge_disjoint():
    preds = ["Apples bananas oranges."]
    refs = ["Computer keyboard monitor."]
    scores = compute_rouge_metrics(preds, refs)
    assert scores["rouge1"]["f1"] == 0.0
    assert scores["rouge2"]["f1"] == 0.0
    assert scores["rougeL"]["f1"] == 0.0


def test_compute_rouge_empty_prediction():
    preds = [""]
    refs = ["Some valid reference text."]
    scores = compute_rouge_metrics(preds, refs)
    assert scores["rouge1"]["f1"] == 0.0
    assert scores["rouge2"]["f1"] == 0.0
    assert scores["rougeL"]["f1"] == 0.0


def test_compute_rouge_length_mismatch():
    preds = ["Only one prediction"]
    refs = ["Reference one", "Reference two"]
    with pytest.raises(ValueError):
        compute_rouge_metrics(preds, refs)


def test_compute_rouge_aggregation():
    preds = [
        "The court has jurisdiction over alleged crimes.",
        "Unrelated text completely here.",
    ]
    refs = [
        "The court has jurisdiction over crimes.",
        "Completely different topic elsewhere.",
    ]
    scores = compute_rouge_metrics(preds, refs)
    assert 0.0 < scores["rouge1"]["f1"] < 100.0
    assert "precision" in scores["rouge1"]
    assert "recall" in scores["rouge1"]
