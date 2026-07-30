from pathlib import Path

import pytest

import fusion.care_train as care_train_module
import fusion.runtime as runtime_module
from fusion.dataset import RobustTriModalDataset


def _validated_val_dataset_stub() -> RobustTriModalDataset:
    dataset = object.__new__(RobustTriModalDataset)
    dataset.pt_dir = Path("validated-pt-pool")
    dataset.is_train = False
    dataset.eval_perturb_plan = None
    dataset.care_digest_view = False
    dataset.samples = [(Path("a.pt"), 0, "a", 2020), (Path("b.pt"), 1, "b", 2021)]
    dataset.sample_sids = ["a", "b"]
    dataset.sample_labels = [0, 1]
    dataset.sample_years = [2020, 2021]
    dataset.sample_groups = ["a", "b"]
    dataset.feature_dim = 515
    return dataset


def test_eval_perturbation_views_share_validated_index_without_mutating_base():
    base = _validated_val_dataset_stub()

    api_view, api_records = runtime_module._build_eval_perturbation_view(
        base,
        perturb_type="api_event_dropout",
        perturb_strength=0.25,
        protocol_seed=424242,
    )
    graph_view, graph_records = runtime_module._build_eval_perturbation_view(
        base,
        perturb_type="graph_sparsify",
        perturb_strength=0.75,
        protocol_seed=424242,
    )

    assert api_view is not base
    assert graph_view is not base
    assert graph_view is not api_view
    assert api_view.samples is base.samples
    assert graph_view.samples is base.samples
    assert api_view.sample_sids is base.sample_sids
    assert graph_view.sample_groups is base.sample_groups
    assert base.eval_perturb_plan is None
    assert base.is_train is False
    assert [row["sampled_strength"] for row in api_records] == [0.25, 0.25]
    assert [row["sampled_strength"] for row in graph_records] == [0.75, 0.75]
    assert [row[1] for row in api_view.eval_perturb_plan] == [0.25, 0.25]
    assert [row[1] for row in graph_view.eval_perturb_plan] == [0.75, 0.75]
    api_view.eval_perturb_plan = ()
    assert len(graph_view.eval_perturb_plan) == 2
    assert base.eval_perturb_plan is None


def test_robust_test_loaders_reuse_validated_test_dataset_and_preserve_sweep(monkeypatch):
    base = _validated_val_dataset_stub()
    cfg = {
        "eval": {
            "controlled_view_protocol_seed": 424242,
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

    monkeypatch.setattr(runtime_module, "build_dataset", forbidden_build_dataset)
    monkeypatch.setattr(runtime_module, "build_loader", capture_loader)

    items = list(runtime_module.iter_robust_test_loaders(cfg, base))

    assert [item["result_key"] for item in items] == [
        "clean",
        "api_event_dropout@0.2",
        "api_event_dropout@0.7",
        "api_missing",
        "graph_missing",
    ]
    assert items[0]["loader"] is None
    views = [item["loader"] for item in items[1:]]
    assert len({id(view) for view in views}) == len(views)
    assert all(view.samples is base.samples for view in views)
    assert all(view.sample_sids is base.sample_sids for view in views)
    assert [
        view.eval_perturb_plan[0][1] for view in views
    ] == [0.2, 0.7, 1.0, 1.0]
    assert all(is_train is False for _, is_train, _ in loader_calls)
    assert all(
        kwargs["persistent_workers_override"] is False
        for _, _, kwargs in loader_calls
    )
    assert all(
        kwargs["seed_namespace"].startswith("care/test/")
        for _, _, kwargs in loader_calls
    )
    assert base.eval_perturb_plan is None


def test_care_and_baseline_fixed_test_views_share_exact_plan_and_digest_mode():
    base = _validated_val_dataset_stub()
    baseline_view, baseline_records = (
        runtime_module._build_eval_perturbation_view(
            base,
            perturb_type="graph_sparsify",
            perturb_strength=0.5,
            protocol_seed=424242,
        )
    )
    care_view, care_records = (
        care_train_module._make_fixed_deterministic_test_view(
            base,
            mechanism="graph_sparsify",
            strength=0.5,
            care_cfg={"views": {"protocol_seed": 424242}},
        )
    )

    assert baseline_view.eval_perturb_plan == care_view.eval_perturb_plan
    assert baseline_records == care_records
    assert baseline_view.care_digest_view is True
    assert care_view.care_digest_view is True


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
        runtime_module._normalize_robust_test_protocol(eval_cfg)


def test_formal_robust_test_protocol_has_19_unique_result_cells():
    cfg = runtime_module.load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    perturbations, strengths = runtime_module._normalize_robust_test_protocol(
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
    assert runtime_module._robust_test_result_count(
        perturbations,
        strengths,
    ) == 19
