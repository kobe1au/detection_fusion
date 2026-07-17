from pathlib import Path

import torch

import run
from fusion.constants import EvidenceIndex
from fusion.discount_fusion import DiscountProbabilityFusion
from fusion.reliability_calibration import (
    MonotonicReliabilityCalibrator,
    RELIABILITY_FEATURE_SUPERSET_LAYOUT,
)
from fusion.train import load_config


ROOT = Path("config/experiments/tri_modal_robust")


def _resolved(relative: str) -> dict:
    return load_config([str(ROOT / relative)])


def _evidence(batch_size: int = 2) -> torch.Tensor:
    evidence = torch.ones(batch_size, EvidenceIndex.BASE_DIM)
    evidence[:, EvidenceIndex.MANIFEST_TO_CODE_CONFLICT] = 0.0
    evidence[:, EvidenceIndex.CODE_TO_MANIFEST_CONFLICT] = 0.0
    return evidence


def test_i1_feature_flags_preserve_calibrator_topology_and_initialization():
    torch.manual_seed(1701)
    full = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=True,
        use_model_visibility=True,
        use_evidential_uncertainty=True,
    )
    torch.manual_seed(1701)
    masked = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
        use_model_visibility=False,
        use_evidential_uncertainty=False,
    )

    assert full.state_dict().keys() == masked.state_dict().keys()
    for name, value in full.state_dict().items():
        assert value.shape == masked.state_dict()[name].shape
        assert torch.equal(value, masked.state_dict()[name])

    for branch, layout in RELIABILITY_FEATURE_SUPERSET_LAYOUT.items():
        first_layer = full.branches[branch].net[0]
        assert first_layer.raw_weight.shape[-1] == len(layout)


def test_disabled_i1_features_are_zero_masked_in_superset_slots():
    calibrator = MonotonicReliabilityCalibrator(
        hidden_dim=8,
        use_relation_evidence=False,
        use_model_visibility=False,
        use_evidential_uncertainty=False,
    )
    outputs = calibrator(_evidence())

    api = outputs["reliability_features_superset_api"]
    graph = outputs["reliability_features_superset_graph"]
    manifest = outputs["reliability_features_superset_manifest"]
    joint = outputs["reliability_features_superset_joint"]
    assert api.shape[-1] == 4
    assert graph.shape[-1] == 4
    assert manifest.shape[-1] == 2
    assert joint.shape[-1] == 5
    assert torch.equal(api[:, 1:], torch.zeros_like(api[:, 1:]))
    assert torch.equal(graph[:, 1:], torch.zeros_like(graph[:, 1:]))
    assert torch.equal(manifest[:, 1], torch.zeros_like(manifest[:, 1]))
    assert torch.equal(joint[:, 2], torch.zeros_like(joint[:, 2]))
    assert torch.equal(joint[:, 4], torch.zeros_like(joint[:, 4]))


def test_no_i1_holds_integrity_weighted_auxiliary_supervision_fixed():
    main = _resolved("evidential_trusted_fusion.yaml")
    no_i1 = _resolved("ablations/modules/no_reliability_discount.yaml")

    assert main["loss"]["integrity_weighted_aux"] is True
    assert no_i1["loss"]["integrity_weighted_aux"] is True
    assert main["loss"]["reliability_weighted_aux"] is None
    assert no_i1["loss"]["reliability_weighted_aux"] is None


def test_fair_no_i2_uses_fixed_evidential_opinion_pool():
    cfg = _resolved("ablations/i2/router_prior_only.yaml")

    assert cfg["fusion"]["combination"] == "routed"
    assert cfg["fusion"]["routing"]["enabled"] is True
    assert cfg["fusion"]["routing"]["mode"] == "prior_only"
    assert cfg["fusion"]["routing"]["train_end_to_end"] is False
    assert cfg["fusion"]["routing"]["posthoc_refine"] is False
    assert cfg["fusion"]["reliability_calibration"]["enabled"] is True
    assert cfg["fusion"]["use_reliability_discount"] is True


def test_removing_i1_calibrator_does_not_change_shared_router_initialization():
    full_cfg = _resolved("evidential_trusted_fusion.yaml")["fusion"]
    no_i1_cfg = _resolved(
        "ablations/modules/no_reliability_discount.yaml"
    )["fusion"]

    torch.manual_seed(1907)
    full = DiscountProbabilityFusion(full_cfg)
    torch.manual_seed(1907)
    no_i1 = DiscountProbabilityFusion(no_i1_cfg)

    assert full.opinion_router is not None
    assert no_i1.opinion_router is not None
    for name, value in full.opinion_router.state_dict().items():
        assert torch.equal(value, no_i1.opinion_router.state_dict()[name])


def test_i2_router_atomic_configs_change_declared_mechanisms():
    main = _resolved("evidential_trusted_fusion.yaml")
    assert main["fusion"]["routing"]["initial_known_retention"] == 0.99

    known_only = _resolved("ablations/i2/router_known_only.yaml")
    assert known_only["fusion"]["routing"]["mode"] == "known_only"
    assert known_only["fusion"]["routing"]["target_loss_weight"] == 0.0
    assert known_only["fusion"]["routing"]["prediction_loss_weight"] == 1.0

    no_disagreement = _resolved("ablations/i2/router_no_disagreement.yaml")
    assert no_disagreement["fusion"]["routing"]["use_disagreement"] is False

    no_encoder = _resolved("ablations/i2/router_no_encoder_training.yaml")
    assert no_encoder["fusion"]["routing"]["train_end_to_end"] is False
    assert no_encoder["fusion"]["routing"]["posthoc_refine"] is True

    no_posthoc = _resolved("ablations/i2/router_no_posthoc_refine.yaml")
    assert no_posthoc["fusion"]["routing"]["train_end_to_end"] is True
    assert no_posthoc["fusion"]["routing"]["posthoc_refine"] is False

    assert len(run.resolve_targets("i2_atomic")) == 5
    assert len(run.resolve_targets("i2_rules")) == 4


def test_i1_i2_factorial_changes_only_the_two_declared_axes():
    cells = {
        (True, True): _resolved(
            "seeds/seed_42.yaml"
        ),
        (False, True): _resolved(
            "ablations/modules/no_reliability_discount.yaml"
        ),
        (True, False): _resolved(
            "ablations/i2/router_prior_only.yaml"
        ),
        (False, False): _resolved(
            "ablations/factorial/i1_i2/i1_off_i2_off.yaml"
        ),
    }
    for (i1_enabled, i2_enabled), cfg in cells.items():
        assert cfg["fusion"]["reliability_calibration"]["enabled"] is i1_enabled
        assert cfg["fusion"]["use_reliability_discount"] is i1_enabled
        assert cfg["fusion"]["visible_integrity_modifier"]["enabled"] is i1_enabled
        assert cfg["fusion"]["routing"]["enabled"] is True
        assert cfg["fusion"]["combination"] == "routed"
        assert cfg["fusion"]["routing"]["mode"] == (
            "learned" if i2_enabled else "prior_only"
        )
        assert cfg["loss"]["integrity_weighted_aux"] is True
        assert cfg["selective_prediction"]["mode"] == "risk_control"


def test_i3_factorial_exposes_all_classification_and_risk_cells():
    for classification_enabled, risk_enabled, relative in (
        (True, True, "seeds/seed_42.yaml"),
        (
            False,
            True,
            "ablations/factorial/i3/classification_off_risk_on.yaml",
        ),
        (
            True,
            False,
            "ablations/factorial/i3/classification_on_risk_off.yaml",
        ),
        (
            False,
            False,
            "ablations/modules/no_i3_selective_rejection.yaml",
        ),
    ):
        cfg = _resolved(relative)
        assert cfg["classification_threshold"]["enabled"] is classification_enabled
        assert cfg["selective_prediction"]["enabled"] is risk_enabled
        assert cfg.get("eval", {}).get("eval_only", False) is (relative != "seeds/seed_42.yaml")

    assert len(run.resolve_targets("i1_i2_2x2")) == 4
    assert len(run.resolve_targets("i3_2x2")) == 4
    # The complete method is shared by the I1xI2 and I3 matrices, so eight
    # logical cells require only seven distinct runs.
    assert len(run.resolve_targets("factorial")) == 7
    assert len(run.resolve_targets("factorial_remaining")) == 6
