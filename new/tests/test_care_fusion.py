from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import pytest

from fusion.care_fusion import (
    CAREPathHeads,
    CAREPathRiskHead,
    PATH_NAMES,
    hard_predict,
    path_availability,
    route_with_agm_anchor,
)
from fusion.care_train import (
    CareRoutedBatch,
    _activate_care_risk_training_partition,
    _care_data_lineage_payload,
    _care_data_lineage_sha256,
    _care_protocol_sha256,
    _care_role_identity_sha256,
    _care_upstream_protocol_sha256,
    _care_upstream_runtime_sha256,
    _make_fixed_deterministic_test_view,
    _oof_diagnostics,
    _preflight_care_output_target,
    _route_cached,
    _validate_source_training_seed,
    validate_care_config,
)
from fusion.care_training import (
    CareRiskCalibrationCache,
    deterministic_view_seed,
)
from fusion.losses import compute_robust_loss
from fusion.runtime import _dataset_common_kwargs, load_config


def _binary_logits(prediction: list[int]) -> torch.Tensor:
    odds = torch.tensor(
        [1.0 if value else -1.0 for value in prediction],
        dtype=torch.float32,
    )
    return torch.stack([torch.zeros_like(odds), odds], dim=-1)


def test_public_hard_prediction_assigns_exact_tie_to_malware() -> None:
    logits = torch.tensor(
        [[0.0, 0.0], [2.0, 1.0], [-1.0, 1.0]],
        dtype=torch.float32,
    )
    assert hard_predict(logits).tolist() == [1, 0, 1]


def test_four_path_heads_are_independent_and_zero_invalid_paths() -> None:
    heads = CAREPathHeads(
        {"api": 3, "graph": 4, "manifest": 5},
        hidden_dim=7,
        dropout=0.0,
    )
    parameter_sets = [
        {id(parameter) for parameter in heads.heads[name].parameters()}
        for name in PATH_NAMES
    ]
    for left in range(len(parameter_sets)):
        for right in range(left + 1, len(parameter_sets)):
            assert parameter_sets[left].isdisjoint(parameter_sets[right])

    embeddings = {
        "api": torch.randn(3, 3),
        "graph": torch.randn(3, 4),
        "manifest": torch.randn(3, 5),
    }
    alive = torch.tensor(
        [[True, True, True], [True, False, True], [True, False, False]]
    )
    logits, available = heads(embeddings, alive)

    assert torch.equal(available, path_availability(alive))
    assert available.tolist() == [
        [True, True, True, True],
        [False, False, True, False],
        [False, False, False, False],
    ]
    for path_index, name in enumerate(PATH_NAMES):
        invalid = ~available[:, path_index]
        assert torch.equal(
            logits[name][invalid],
            torch.zeros_like(logits[name][invalid]),
        )


def test_stage_a_loss_is_exact_equal_mean_of_four_clean_path_ces() -> None:
    labels = torch.tensor([0, 1], dtype=torch.long)
    paths = {
        "agm": torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "ag": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "am": torch.tensor([[0.5, 0.0], [0.0, 0.5]]),
        "gm": torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
    }
    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(paths[name], labels)
            for name in PATH_NAMES
        ]
    ).mean()
    actual, parts = compute_robust_loss(
        paths["agm"],
        labels,
        {
            "care_path_logits": paths,
            "care_path_available": torch.ones(
                (2, 4), dtype=torch.bool
            ),
        },
        {
            "objective": "care_stage_a_clean",
            "label_smoothing": 0.0,
        },
    )
    assert torch.allclose(actual, expected)
    assert parts["care_active_path_count"] == 4.0


def test_agm_anchor_route_switches_only_to_better_disagreeing_pair() -> None:
    path_logits = {
        "agm": _binary_logits([0, 1, 0, 1]),
        "ag": _binary_logits([1, 0, 1, 0]),
        "am": _binary_logits([0, 0, 0, 0]),
        "gm": _binary_logits([1, 0, 1, 0]),
    }
    alive = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, False, True],
            [True, False, False],
        ]
    )
    available = path_availability(alive)
    score = torch.tensor(
        [
            [0.60, 0.70, 0.99, 0.60],
            [0.60, 0.60, 0.95, 0.99],
            [0.10, 0.90, 0.80, 0.90],
            [0.99, 0.99, 0.99, 0.99],
        ]
    )

    routed = route_with_agm_anchor(path_logits, available, score)

    # Row 0: AM agrees with AGM and GM only ties AGM, so AG is the only
    # permitted, strictly better fallback. Row 1 keeps AGM because every pair
    # disagrees but none beats the declared high scores consistently with
    # availability/tie priority after setting AG equal; AM/GM predict benign,
    # so they disagree with the malware AGM and GM wins at 0.99.
    assert routed.selected_path_index.tolist() == [1, 3, 2, -1]
    assert routed.reject.tolist() == [False, False, False, True]


def test_multiview_routed_prediction_preserves_leading_dimensions() -> None:
    selected_logits = torch.tensor(
        [
            [[0.0, 0.0], [2.0, 1.0], [0.0, 2.0]],
            [[2.0, 1.0], [0.0, 0.0], [1.0, 2.0]],
        ]
    )
    eligible = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    routed = CareRoutedBatch(
        selected_path_index=torch.zeros((2, 3), dtype=torch.long),
        selected_logits=selected_logits,
        selected_score=torch.ones((2, 3)),
        eligible=eligible,
        disagreement_with_agm=torch.zeros((2, 3, 3), dtype=torch.bool),
    )
    assert routed.prediction.shape == (2, 3)
    assert routed.prediction.tolist() == [[1, 0, -1], [0, 1, 1]]


class _PartitionProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage_a = nn.Linear(3, 2)
        self.care_risk_head = CAREPathRiskHead(hidden_dim=4)


def test_risk_training_partition_reenables_only_risk_head() -> None:
    model = _PartitionProbe()
    for parameter in model.care_risk_head.parameters():
        parameter.requires_grad_(False)

    _activate_care_risk_training_partition(model)  # type: ignore[arg-type]

    assert all(
        not parameter.requires_grad for parameter in model.stage_a.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.care_risk_head.parameters()
    )


def test_role_identity_uses_semantic_digests_not_machine_path() -> None:
    roles = {
        name: {
            "num_samples": 2,
            "num_groups": 2,
            "sample_ids_sha256": f"sid-{name}",
            "groups_sha256": f"group-{name}",
            "semantic_rows_sha256": f"rows-{name}",
        }
        for name in (
            "expert_train",
            "expert_val",
            "routing_cal",
            "decision_cal",
            "test",
        )
    }
    first = {
        "protocol": "closed",
        "expert_split_seed": 4242,
        "expert_val_fraction": 0.1,
        "validation_role_assignment_path": "/machine-a/roles.json",
        "validation_role_assignment_semantic_sha256": "semantic",
        "roles": roles,
    }
    second = dict(first)
    second["validation_role_assignment_path"] = "/machine-b/roles.json"
    assert _care_role_identity_sha256(first) == _care_role_identity_sha256(
        second
    )
    changed = {
        **second,
        "roles": {
            **roles,
            "routing_cal": {
                **roles["routing_cal"],
                "sample_ids_sha256": "changed",
            },
        },
    }
    assert _care_role_identity_sha256(first) != _care_role_identity_sha256(
        changed
    )
    relabeled = {
        **second,
        "roles": {
            **roles,
            "test": {
                **roles["test"],
                "semantic_rows_sha256": "relabeled-test",
            },
        },
    }
    assert _care_role_identity_sha256(first) != _care_role_identity_sha256(
        relabeled
    )


def test_data_lineage_binds_content_not_machine_path(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    for root in (first_root, second_root):
        (root / "audit.json").write_bytes(b'{"pool":"same"}')
    provenance = {
        "verified": True,
        "manifest_vocab_sha256": "a" * 64,
        "train_csv_sha256": "b" * 64,
        "train_sample_ids_sha256": "c" * 64,
        "num_train_samples": 17,
    }

    def config(root) -> dict:
        return {
            "data": {
                "root": str(root),
                "expected_pt_build_fingerprint": "d" * 64,
                "pt_audit_certificate": "audit.json",
            }
        }

    first_payload = _care_data_lineage_payload(
        config(first_root),
        provenance,
    )
    second_payload = _care_data_lineage_payload(
        config(second_root),
        provenance,
    )
    assert first_payload == second_payload
    assert _care_data_lineage_sha256(
        config(first_root),
        provenance,
    ) == _care_data_lineage_sha256(
        config(second_root),
        provenance,
    )

    (second_root / "audit.json").write_bytes(b'{"pool":"changed"}')
    assert _care_data_lineage_sha256(
        config(first_root),
        provenance,
    ) != _care_data_lineage_sha256(
        config(second_root),
        provenance,
    )


def test_dataset_certificate_is_resolved_against_data_root(tmp_path) -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    cfg["data"]["root"] = str(tmp_path)
    cfg["data"]["pt_audit_certificate"] = "audit.json"
    kwargs = _dataset_common_kwargs(cfg, is_train=False)
    assert kwargs["pt_audit_certificate"] == str(tmp_path / "audit.json")

    cfg["data"]["pt_audit_certificate"] = ""
    kwargs = _dataset_common_kwargs(cfg, is_train=False)
    assert kwargs["pt_audit_certificate"] is None


def test_registered_eval_ablations_preserve_upstream_protocol() -> None:
    base = validate_care_config(
        load_config(
            [
                "config/experiments/tri_modal_robust/seeds/seed_42.yaml"
            ]
        )
    )
    base_upstream = _care_upstream_protocol_sha256(base)
    for name in (
        "no_learned_routing",
        "route_on_all_samples",
        "msp_acceptance",
    ):
        ablation = validate_care_config(
            load_config(
                [
                    "config/experiments/tri_modal_robust/ablations/care/"
                    f"{name}.yaml"
                ]
            )
        )
        assert _care_upstream_protocol_sha256(ablation) == base_upstream
        assert _care_protocol_sha256(ablation) != _care_protocol_sha256(
            base
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", 0),
        ("route_on_all_samples", "false"),
    ],
)
def test_care_routing_flags_require_real_booleans(
    field: str,
    value: object,
) -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    cfg["care"]["routing"][field] = value
    with pytest.raises(ValueError, match=rf"care\.routing\.{field}"):
        validate_care_config(cfg)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("train", "use_amp"),
        ("train", "deterministic"),
        ("train", "strict_deterministic"),
        ("eval", "eval_only"),
    ],
)
def test_care_lifecycle_flags_reject_string_booleans(
    section: str,
    field: str,
) -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    cfg[section][field] = "false"
    with pytest.raises(ValueError, match=rf"{section}\.{field}"):
        validate_care_config(cfg)


def test_eval_only_source_is_bound_to_model_training_seed() -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    source = {
        "training_seed": 42,
        "cfg": {"train": {"seed": 42}},
    }
    assert _validate_source_training_seed(source, cfg) == 42

    wrong_seed_cfg = copy.deepcopy(cfg)
    wrong_seed_cfg["train"]["seed"] = 2024
    with pytest.raises(ValueError, match="must match the source model"):
        _validate_source_training_seed(source, wrong_seed_cfg)

    corrupt_source = copy.deepcopy(source)
    corrupt_source["cfg"]["train"]["seed"] = 3407
    with pytest.raises(ValueError, match="disagrees"):
        _validate_source_training_seed(corrupt_source, cfg)


def test_no_learned_routing_keeps_structural_two_modality_fallback() -> None:
    path_logits = torch.zeros((1, 4, 2), dtype=torch.float32)
    path_logits[0, 1] = torch.tensor([0.0, 2.0])
    path_available = torch.tensor(
        [[False, True, False, False]],
        dtype=torch.bool,
    )
    routed = _route_cached(
        path_logits,
        path_available,
        torch.tensor([[0.0, 0.8, 0.0, 0.0]]),
        {"enabled": False, "route_on_all_samples": False},
    )
    assert routed.selected_path_index.tolist() == [1]
    assert routed.eligible.tolist() == [True]
    assert torch.equal(routed.selected_logits, path_logits[:, 1])


def test_disabled_care_routing_cannot_enable_route_on_all() -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/ablations/care/"
         "no_learned_routing.yaml"]
    )
    cfg["care"]["routing"]["route_on_all_samples"] = True
    with pytest.raises(
        ValueError,
        match="Disabled CARE routing requires",
    ):
        validate_care_config(cfg)


def test_protocol_identity_ignores_role_assignment_machine_path() -> None:
    base = validate_care_config(
        load_config(
            ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
        )
    )
    relocated = copy.deepcopy(base)
    relocated["roles"]["validation_role_assignment_path"] = (
        "/another/mount/validation_roles_protocol_v3.json"
    )
    assert _care_protocol_sha256(relocated) == _care_protocol_sha256(base)
    assert _care_upstream_protocol_sha256(
        relocated
    ) == _care_upstream_protocol_sha256(base)


def test_upstream_runtime_identity_binds_behavior_not_output_path() -> None:
    cfg = load_config(
        ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
    )
    original = _care_upstream_runtime_sha256(cfg)

    moved_output = copy.deepcopy(cfg)
    moved_output["data"]["out_dir"] = "another/machine/output"
    assert _care_upstream_runtime_sha256(moved_output) == original

    changed_budget = copy.deepcopy(cfg)
    changed_budget["data"]["max_api_events_per_sample"] = 1024
    assert _care_upstream_runtime_sha256(changed_budget) != original

    changed_amp = copy.deepcopy(cfg)
    changed_amp["train"]["use_amp"] = not bool(
        changed_amp["train"].get("use_amp", True)
    )
    assert _care_upstream_runtime_sha256(changed_amp) != original


def test_fixed_test_view_uses_protocol_seed_formula() -> None:
    class _DatasetProbe:
        sample_sids = ["sid-a", "sid-b"]
        is_train = False

    care_cfg = validate_care_config(
        load_config(
            ["config/experiments/tri_modal_robust/seeds/seed_42.yaml"]
        )
    )
    view, records = _make_fixed_deterministic_test_view(
        _DatasetProbe(),  # type: ignore[arg-type]
        mechanism="api_event_dropout",
        strength=0.3,
        care_cfg=care_cfg,
    )
    expected = [
        deterministic_view_seed(
            sid,
            "api_event_dropout",
            care_cfg["views"]["protocol_seed"],
        )
        for sid in _DatasetProbe.sample_sids
    ]
    assert [row["view_seed"] for row in records] == expected
    assert [item[2] for item in view.eval_perturb_plan] == expected
    assert [item[1] for item in view.eval_perturb_plan] == [0.3, 0.3]
    assert not hasattr(view, "eval_perturb_type")
    assert view.care_digest_view is True


def test_oof_switch_diagnostics_exclude_structural_pair_selection() -> None:
    # With API+Manifest alive, AM is selected structurally. AGM is invalid and
    # its zero-logit placeholder must not be credited as a correct anchor.
    path_logits = torch.zeros((1, 1, 4, 2), dtype=torch.float32)
    path_logits[0, 0, 2] = torch.tensor([1.0, 0.0])
    cache = CareRiskCalibrationCache.from_path_logits(
        sids=("sid",),
        groups=("group",),
        labels=torch.tensor([1], dtype=torch.long),
        view_names=("api_missing",),
        path_logits=path_logits,
        modality_alive=torch.tensor(
            [[[True, False, True]]], dtype=torch.bool
        ),
    )
    score = torch.zeros((1, 1, 4), dtype=torch.float32)
    score[0, 0, 2] = 0.8
    summary, rows = _oof_diagnostics(
        cache,
        score,
        torch.tensor([0], dtype=torch.long),
        {"enabled": True, "route_on_all_samples": False},
    )
    switch = summary["routing_switch"]
    assert switch["switch_count"] == 0
    assert switch["repair_count"] == 0
    assert switch["destruction_count"] == 0
    assert switch["structural_pair_selection_count"] == 1
    assert rows[0]["selected_path"] == "am"
    assert rows[0]["switched_from_agm"] is False
    assert rows[0]["structural_pair_selection"] is True
    assert math.isnan(rows[0]["relative_advantage_vs_agm"])


def test_output_preflight_rejects_collision_and_self_overwrite(
    tmp_path,
) -> None:
    cfg = {
        "data": {"out_dir": str(tmp_path)},
        "train": {"exp_name": "main", "seed": 42},
        "eval": {"eval_only": False},
    }
    target = tmp_path / "main" / "42"
    target.mkdir(parents=True)
    (target / "summary.yaml").write_text("done: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already contains"):
        _preflight_care_output_target(cfg, overwrite=False)

    (target / "best_care_pipeline.pt").write_bytes(b"source")
    self_eval = {
        **cfg,
        "eval": {
            "eval_only": True,
            "output_name": "main",
            "checkpoint_path": str(target / "best_care_pipeline.pt"),
        },
    }
    with pytest.raises(ValueError, match="equals its output"):
        _preflight_care_output_target(self_eval, overwrite=True)
