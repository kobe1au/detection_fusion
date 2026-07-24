from pathlib import Path

import pytest
from torch.utils.data import ConcatDataset, DataLoader, Subset

import fusion.train as train_module
from fusion.dataset import RobustTriModalDataset


def _validated_val_dataset_stub() -> RobustTriModalDataset:
    dataset = object.__new__(RobustTriModalDataset)
    dataset.pt_dir = Path("validated-pt-pool")
    dataset.is_train = False
    dataset.robust_aug = False
    dataset.eval_perturb_type = None
    dataset.eval_perturb_strength = 0.0
    dataset.samples = [(Path("a.pt"), 0, "a", 2020), (Path("b.pt"), 1, "b", 2021)]
    dataset.sample_sids = ["a", "b"]
    dataset.sample_labels = [0, 1]
    dataset.sample_years = [2020, 2021]
    dataset.sample_groups = ["a", "b"]
    dataset.feature_dim = 515
    return dataset


def test_eval_perturbation_views_share_validated_index_without_mutating_base():
    base = _validated_val_dataset_stub()

    api_view = train_module._build_eval_perturbation_view(
        base,
        perturb_type="api_event_dropout",
        perturb_strength=0.25,
    )
    graph_view = train_module._build_eval_perturbation_view(
        base,
        perturb_type="graph_sparsify",
        perturb_strength=0.75,
    )

    assert api_view is not base
    assert graph_view is not base
    assert graph_view is not api_view
    assert api_view.samples is base.samples
    assert graph_view.samples is base.samples
    assert api_view.sample_sids is base.sample_sids
    assert graph_view.sample_groups is base.sample_groups
    assert base.eval_perturb_type is None
    assert base.eval_perturb_strength == 0.0
    assert base.is_train is False
    assert base.robust_aug is False
    assert (api_view.eval_perturb_type, api_view.eval_perturb_strength) == (
        "api_event_dropout",
        0.25,
    )
    assert (graph_view.eval_perturb_type, graph_view.eval_perturb_strength) == (
        "graph_sparsify",
        0.75,
    )

    api_view.eval_perturb_strength = 0.5
    assert graph_view.eval_perturb_strength == 0.75
    assert base.eval_perturb_strength == 0.0


def test_robust_test_loaders_reuse_validated_test_dataset_and_preserve_sweep(monkeypatch):
    base = _validated_val_dataset_stub()
    cfg = {
        "eval": {
            "perturb_tests": [
                "clean",
                "api_event_dropout",
                "api_missing",
                "graph_missing",
            ],
            "perturb_strengths": [0.2, 0.7],
        },
        "train": {},
    }
    loader_calls = []

    def forbidden_build_dataset(*args, **kwargs):
        raise AssertionError("robust test views must not rebuild/rescan the dataset")

    def capture_loader(cfg, dataset, is_train, **kwargs):
        loader_calls.append((dataset, is_train, kwargs))
        return dataset

    monkeypatch.setattr(train_module, "build_dataset", forbidden_build_dataset)
    monkeypatch.setattr(train_module, "build_loader", capture_loader)

    items = list(train_module.iter_robust_test_loaders(cfg, base))

    assert [item["result_key"] for item in items] == [
        "clean",
        "api_event_dropout_s0.2",
        "api_event_dropout_s0.7",
        "api_missing",
        "graph_missing",
    ]
    assert items[0]["loader"] is None
    views = [item["loader"] for item in items[1:]]
    assert len({id(view) for view in views}) == len(views)
    assert all(view.samples is base.samples for view in views)
    assert all(view.sample_sids is base.sample_sids for view in views)
    assert [view.eval_perturb_type for view in views] == [
        "api_event_dropout",
        "api_event_dropout",
        "api_missing",
        "graph_missing",
    ]
    assert [view.eval_perturb_strength for view in views] == [0.2, 0.7, 1.0, 1.0]
    assert all(is_train is False for _, is_train, _ in loader_calls)
    assert all(kwargs == {} for _, _, kwargs in loader_calls)
    assert base.eval_perturb_type is None
    assert base.eval_perturb_strength == 0.0


@pytest.mark.parametrize(
    ("eval_cfg", "message"),
    [
        ({"perturb_tests": []}, "non-empty sequence"),
        (
            {"perturb_tests": ["clean", "clean"]},
            "contains duplicates",
        ),
        (
            {"perturb_tests": ["not_a_real_transform"]},
            "unsupported mechanisms",
        ),
        (
            {
                "perturb_tests": ["clean"],
                "perturb_strengths": [0.3, 0.3],
            },
            "contains duplicates",
        ),
        (
            {
                "perturb_tests": ["clean"],
                "perturb_strengths": [0.3, 0.30000001],
            },
            "collide after result-key formatting",
        ),
        (
            {
                "perturb_tests": ["clean"],
                "perturb_strengths": [],
            },
            "non-empty sequence",
        ),
        (
            {
                "perturb_tests": ["clean"],
                "perturb_strengths": [True],
            },
            "not booleans",
        ),
        (
            {
                "perturb_tests": ["clean"],
                "perturb_strengths": [1.1],
            },
            "within \\[0, 1\\]",
        ),
    ],
)
def test_robust_test_protocol_rejects_ambiguous_or_invalid_cells(eval_cfg, message):
    with pytest.raises(ValueError, match=message):
        train_module._normalize_robust_test_protocol(eval_cfg)


def test_formal_robust_test_protocol_has_19_unique_result_cells():
    cfg = train_module.load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    perturbations, strengths = train_module._normalize_robust_test_protocol(
        cfg["eval"]
    )

    assert perturbations == [
        "clean",
        "api_event_dropout",
        "graph_sparsify",
        "manifest_permission_mask",
        "api_missing",
        "graph_missing",
        "manifest_missing",
    ]
    assert strengths == [0.1, 0.3, 0.5, 0.7, 0.9]
    assert train_module._robust_test_result_count(perturbations, strengths) == 19


def test_reliability_loaders_reuse_base_dataset_and_keep_views_independent(monkeypatch):
    base = _validated_val_dataset_stub()
    cfg = {
        "calibration": {
            "fit_perturbations": [
                "api_event_dropout",
                "graph_sparsify",
                "manifest_permission_mask",
            ],
            "perturb_strengths": [0.5],
        },
        "fusion": {
            "combination": "dempster",
            "reliability_calibration": {"enabled": True},
        },
        "robust": {},
        "train": {},
    }
    loader_calls = []

    def forbidden_build_dataset(*args, **kwargs):
        raise AssertionError("post-hoc views must not rebuild/rescan the dataset")

    def capture_loader(cfg, dataset, is_train, **kwargs):
        loader_calls.append((dataset, is_train, kwargs))
        return dataset

    monkeypatch.setattr(train_module, "build_dataset", forbidden_build_dataset)
    monkeypatch.setattr(train_module, "build_loader", capture_loader)

    items = train_module.build_reliability_calibration_loaders(cfg, base, [0])

    assert len(items) == len(train_module.reliability_calibration_scenarios(cfg))
    assert len(loader_calls) == 1
    assert len({id(item["loader"]) for item in items}) == 1
    combined_dataset, is_train, loader_kwargs = loader_calls[0]
    assert isinstance(combined_dataset, ConcatDataset)
    assert is_train is False
    assert loader_kwargs["persistent_workers_override"] is False
    assert isinstance(
        loader_kwargs["batch_sampler_override"],
        train_module._ScenarioBoundaryBatchSampler,
    )
    assert loader_kwargs["collate_fn_override"] is train_module._robust_calibration_collate_fn
    tagged_views = list(combined_dataset.datasets)
    assert len(tagged_views) == len(items)
    assert [item.source_index for item in tagged_views] == list(range(len(items)))
    subsets = [item.dataset for item in tagged_views]
    assert all(isinstance(dataset, Subset) for dataset in subsets)
    assert all(dataset.indices == [0] for dataset in subsets)
    views = [dataset.dataset for dataset in subsets]
    assert len({id(view) for view in views}) == len(items)
    assert all(view.samples is base.samples for view in views)
    assert [item["combined_source_index"] for item in items] == list(
        range(len(items))
    )
    assert base.eval_perturb_type is None
    assert base.eval_perturb_strength == 0.0


def test_calibration_batch_sampler_preserves_source_boundaries_order_and_size(monkeypatch):
    source_values = [["a0", "a1", "a2", "a3", "a4"], ["b0", "b1"], ["c0"]]
    tagged = [
        train_module._TaggedCalibrationScenarioDataset(values, source_index)
        for source_index, values in enumerate(source_values)
    ]
    combined = ConcatDataset(tagged)
    sampler = train_module._ScenarioBoundaryBatchSampler(
        [len(values) for values in source_values],
        batch_size=3,
    )

    monkeypatch.setattr(
        train_module,
        "robust_collate_fn",
        lambda values: {"values": list(values)},
    )
    loader = DataLoader(
        combined,
        batch_sampler=sampler,
        collate_fn=train_module._robust_calibration_collate_fn,
        num_workers=0,
    )
    batches = list(loader)

    assert [batch["calibration_source_index"] for batch in batches] == [0, 0, 1, 2]
    assert [batch["values"] for batch in batches] == [
        ["a0", "a1", "a2"],
        ["a3", "a4"],
        ["b0", "b1"],
        ["c0"],
    ]
    assert len(sampler) == 4
