#!/usr/bin/env python

from types import SimpleNamespace

import pytest

from lerobot.rl.acp_dataset_stats import compute_acp_indicator_stats, validate_acp_training_dataset


class DummyHFDataset:
    def __init__(self, columns: dict[str, list]):
        self._columns = columns
        self.column_names = list(columns.keys())

    def __getitem__(self, key: str):
        return self._columns[key]


class DummyTrainingDataset:
    def __init__(self, columns: dict[str, list], *, task="pick", include_task=True):
        self.hf_dataset = DummyHFDataset(columns)
        self.task = task
        self.include_task = include_task

    def __getitem__(self, index: int):
        item = {"index": index}
        if self.include_task:
            item["task"] = self.task
        return item


def test_compute_acp_indicator_stats_from_meta_stats():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            stats={
                "complementary_info.acp_indicator": {
                    "mean": [0.3],
                    "count": [10],
                }
            }
        )
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "meta.stats"
    assert stats.positive_ratio == 0.3
    assert stats.total_count == 10
    assert stats.positive_count == 3
    assert stats.invalid_count == -1


def test_compute_acp_indicator_stats_from_dataset_scan():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={}),
        hf_dataset=DummyHFDataset(
            {
                "complementary_info.acp_indicator": [[1], [0], [1], [2]],
            }
        ),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "hf_dataset_scan"
    assert stats.total_count == 4
    assert stats.positive_count == 2
    assert stats.positive_ratio == 0.5
    assert stats.invalid_count == 1


def test_compute_acp_indicator_stats_returns_none_when_missing_field():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(stats={}),
        hf_dataset=DummyHFDataset({"task": ["pick", "place"]}),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")
    assert stats is None


def test_compute_acp_indicator_stats_prefers_meta_stats():
    dataset = SimpleNamespace(
        meta=SimpleNamespace(
            stats={
                "complementary_info.acp_indicator": {
                    "mean": [0.75],
                    "count": [8],
                }
            }
        ),
        hf_dataset=DummyHFDataset({"complementary_info.acp_indicator": [0, 0, 0, 0]}),
    )

    stats = compute_acp_indicator_stats(dataset, "complementary_info.acp_indicator")

    assert stats is not None
    assert stats.source == "meta.stats"
    assert stats.positive_ratio == 0.75
    assert stats.total_count == 8
    assert stats.positive_count == 6


def test_validate_acp_training_dataset_accepts_scalar_integer_binary_column():
    field = "complementary_info.acp_indicator"
    dataset = DummyTrainingDataset({field: [0, 1, 0, 1]})

    stats = validate_acp_training_dataset(dataset, field)

    assert stats.source == "strict_hf_dataset_scan"
    assert stats.total_count == 4
    assert stats.positive_count == 2


@pytest.mark.parametrize(
    ("values", "exception", "message"),
    [
        ([0.0, 1.0], TypeError, "integer scalars"),
        ([[0], [1]], ValueError, "one scalar per frame"),
        ([0, 2], ValueError, "only 0/1"),
        ([1, 1], ValueError, "both 0"),
        ([], ValueError, "empty"),
    ],
)
def test_validate_acp_training_dataset_rejects_invalid_annotations(values, exception, message):
    field = "complementary_info.acp_indicator"
    dataset = DummyTrainingDataset({field: values})

    with pytest.raises(exception, match=message):
        validate_acp_training_dataset(dataset, field)


def test_validate_acp_training_dataset_rejects_missing_task():
    field = "complementary_info.acp_indicator"
    dataset = DummyTrainingDataset({field: [0, 1]}, include_task=False)

    with pytest.raises(KeyError, match="task"):
        validate_acp_training_dataset(dataset, field)
