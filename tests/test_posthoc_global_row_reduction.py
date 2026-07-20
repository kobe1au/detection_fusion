from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.losses import (
    compute_posthoc_calibration_loss,
    compute_reliability_calibration_loss,
    reliability_alive_mask,
    reliability_correctness_target,
    reliability_per_sample_loss,
    routing_mixture_log_prob,
    routing_risk_per_sample_loss,
    routing_risk_target,
    routing_soft_oracle_per_sample_loss,
    routing_soft_oracle_target,
)
from fusion.train import (
    _compile_posthoc_row_weights,
    _compile_posthoc_source_masses,
)


SCENARIO_PRIOR = {"clean": 0.7, "perturb": 0.3}


def _sources_and_groups():
    # Two clean sources and three perturbation families. Family/source sizes,
    # row counts, and valid counts are deliberately unequal; f2 is all-dead.
    specs = (
        ("c1", "clean", ((1, 1, 1), (0, 0, 0)), (0, 1)),
        ("c2", "clean", ((1, 0, 0), (0, 0, 0), (1, 1, 0)), (1, 0, 1)),
        ("p1", "f1", ((0, 1, 1),), (1,)),
        ("p2", "f1", ((1, 1, 1), (0, 0, 0), (1, 0, 1), (0, 0, 0)), (0, 1, 1, 0)),
        ("p3", "f2", ((0, 0, 0), (0, 0, 0)), (1, 0)),
        ("p4", "f3", ((1, 1, 0), (1, 1, 1), (0, 1, 0)), (0, 1, 0)),
        ("p5", "f3", ((0, 0, 1), (0, 0, 0)), (1, 1)),
        ("p6", "f3", ((1, 0, 1),), (0,)),
    )
    items = []
    segments = []
    labels = []
    evidence = []
    offset = 0
    for name, family, alive_rows, source_labels in specs:
        rows = len(source_labels)
        source_evidence = torch.zeros(rows, EvidenceIndex.BASE_DIM)
        source_evidence[:, :3] = 1.0
        source_evidence[:, EvidenceIndex.API_ALIVE] = torch.tensor(
            [row[0] for row in alive_rows], dtype=torch.float32
        )
        source_evidence[:, EvidenceIndex.GRAPH_ALIVE] = torch.tensor(
            [row[1] for row in alive_rows], dtype=torch.float32
        )
        source_evidence[:, EvidenceIndex.MANIFEST_ALIVE] = torch.tensor(
            [row[2] for row in alive_rows], dtype=torch.float32
        )
        source_labels_tensor = torch.tensor(source_labels, dtype=torch.long)
        item = {
            "name": name,
            "scenario_group": family,
            "labels": source_labels_tensor,
            "evidence": source_evidence,
            "reliability_branches": ("api", "graph", "manifest"),
        }
        items.append(item)
        segments.append((offset, offset + rows))
        labels.append(source_labels_tensor)
        evidence.append(source_evidence)
        offset += rows

    clean = [item for item in items if item["scenario_group"] == "clean"]
    groups = []
    for family in ("f1", "f2", "f3"):
        groups.append(
            {
                "name": family,
                "clean": clean,
                "scenario": [
                    item for item in items if item["scenario_group"] == family
                ],
            }
        )
    return items, segments, groups, torch.cat(labels), torch.cat(evidence)


def _slice_outputs(outputs: dict, start: int, end: int, total: int) -> dict:
    return {
        key: (
            value[start:end]
            if isinstance(value, torch.Tensor)
            and value.ndim > 0
            and int(value.size(0)) == total
            else value
        )
        for key, value in outputs.items()
    }


def _old_group_reduction(
    outputs: dict,
    items: list[dict],
    segments: list[tuple[int, int]],
    groups: list[dict],
    config: dict,
) -> torch.Tensor:
    total = sum(int(item["labels"].numel()) for item in items)
    segment_by_id = {id(item): segment for item, segment in zip(items, segments)}

    def source_loss(item: dict) -> torch.Tensor:
        start, end = segment_by_id[id(item)]
        loss, _ = compute_posthoc_calibration_loss(
            _slice_outputs(outputs, start, end, total),
            item["labels"],
            item["evidence"],
            config,
            reliability_branches=item["reliability_branches"],
            materialize_diagnostics=False,
        )
        return loss

    group_losses = []
    for group in groups:
        clean_loss = torch.stack([source_loss(item) for item in group["clean"]]).mean()
        scenario = group.get("scenario") or []
        if scenario:
            perturb_loss = torch.stack([source_loss(item) for item in scenario]).mean()
            group_losses.append(
                SCENARIO_PRIOR["clean"] * clean_loss
                + SCENARIO_PRIOR["perturb"] * perturb_loss
            )
        else:
            group_losses.append(clean_loss)
    return torch.stack(group_losses).mean()


def _fixed_branch_probabilities(rows: int) -> torch.Tensor:
    row = torch.arange(rows, dtype=torch.float32).view(-1, 1)
    offsets = torch.tensor((-0.45, 0.15, 0.65), dtype=torch.float32).view(1, -1)
    malware = (0.1 + 0.8 * torch.sigmoid(torch.sin(row * 0.71 + offsets))).clamp(
        1.0e-4, 1.0 - 1.0e-4
    )
    return torch.stack((1.0 - malware, malware), dim=-1)


def _route_outputs(
    raw_scores: torch.Tensor,
    evidence: torch.Tensor,
    branch_probabilities: torch.Tensor,
) -> dict:
    alive = torch.stack(
        (
            evidence[:, EvidenceIndex.API_ALIVE],
            evidence[:, EvidenceIndex.GRAPH_ALIVE],
            evidence[:, EvidenceIndex.MANIFEST_ALIVE],
        ),
        dim=-1,
    ).bool()
    has_available = alive.any(dim=-1)
    masked_scores = raw_scores.masked_fill(~alive, -1.0e9)
    route_scores = torch.where(
        has_available.unsqueeze(-1), masked_scores, torch.zeros_like(masked_scores)
    )
    branch_distribution = F.softmax(route_scores, dim=-1)
    mixture = (branch_distribution.unsqueeze(-1) * branch_probabilities).sum(dim=1)
    mixture = torch.where(
        has_available.unsqueeze(-1), mixture, torch.full_like(mixture, 0.5)
    )
    return {
        "routing_active": torch.ones_like(has_available, dtype=torch.float32),
        "routing_has_available": has_available,
        "routing_mixture_prob": mixture,
        "routing_branch_distribution": branch_distribution,
        "routing_scores": route_scores,
        **{
            f"calibrated_log_prob_{name}": branch_probabilities[:, index].log()
            for index, name in enumerate(("api", "graph", "manifest"))
        },
    }


def test_compiled_source_and_row_weights_preserve_hierarchy_and_all_dead():
    items, segments, groups, _labels, evidence = _sources_and_groups()
    masses = _compile_posthoc_source_masses(groups, items, SCENARIO_PRIOR)
    by_name = {item["name"]: masses[id(item)] for item in items}

    assert by_name["c1"] == pytest.approx(0.35)
    assert by_name["c2"] == pytest.approx(0.35)
    assert by_name["p1"] == pytest.approx(0.05)
    assert by_name["p2"] == pytest.approx(0.05)
    assert by_name["p3"] == pytest.approx(0.10)
    for name in ("p4", "p5", "p6"):
        assert by_name[name] == pytest.approx(0.3 / 9.0)

    valid = evidence[:, EvidenceIndex.API_ALIVE : EvidenceIndex.MANIFEST_ALIVE + 1].bool().any(dim=-1)
    row_weights = _compile_posthoc_row_weights(items, segments, masses, valid)
    for item, (start, end) in zip(items, segments):
        expected = 0.0 if item["name"] == "p3" else masses[id(item)]
        assert float(row_weights[start:end].sum()) == pytest.approx(expected)
    # The dead source's objective mass is intentionally not redistributed.
    assert float(row_weights.sum()) == pytest.approx(0.9)


def test_compiled_clean_only_group_ignores_perturb_prior():
    items, segments, _groups, _labels, evidence = _sources_and_groups()
    clean = items[:2]
    clean_segments = segments[:2]
    groups = [{"name": "clean", "clean": clean, "scenario": []}]
    masses = _compile_posthoc_source_masses(groups, clean, SCENARIO_PRIOR)
    assert [masses[id(item)] for item in clean] == pytest.approx([0.5, 0.5])
    valid = evidence[: clean_segments[-1][1], EvidenceIndex.API_ALIVE : EvidenceIndex.MANIFEST_ALIVE + 1].bool().any(dim=-1)
    weights = _compile_posthoc_row_weights(clean, clean_segments, masses, valid)
    assert float(weights.sum()) == pytest.approx(1.0)


def test_global_route_loss_matches_source_family_reference_value_and_gradient():
    items, segments, groups, labels, evidence = _sources_and_groups()
    rows = labels.numel()
    branch_probabilities = _fixed_branch_probabilities(rows)
    initial_scores = torch.linspace(-0.9, 1.1, rows * 3).view(rows, 3)
    config = {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": {
            "enabled": True,
            "posthoc_refine": True,
            "calibration_weight": 0.8,
            "prediction_loss_weight": 1.2,
            "route_oracle_loss_weight": 0.35,
            "route_oracle_temperature": 0.7,
            "risk_loss_weight": 0.0,
        },
    }

    reference_scores = initial_scores.clone().requires_grad_(True)
    reference_outputs = _route_outputs(
        reference_scores, evidence, branch_probabilities
    )
    reference = _old_group_reduction(
        reference_outputs, items, segments, groups, config
    )
    reference.backward()

    global_scores = initial_scores.clone().requires_grad_(True)
    global_outputs = _route_outputs(global_scores, evidence, branch_probabilities)
    masses = _compile_posthoc_source_masses(groups, items, SCENARIO_PRIOR)
    prediction_weights = _compile_posthoc_row_weights(
        items,
        segments,
        masses,
        global_outputs["routing_has_available"],
    )
    mixture_log_prob = routing_mixture_log_prob(global_outputs)
    prediction_per_row = F.nll_loss(mixture_log_prob, labels, reduction="none")
    oracle_target, oracle_valid = routing_soft_oracle_target(
        global_outputs,
        labels,
        evidence,
        temperature=0.7,
    )
    oracle_weights = _compile_posthoc_row_weights(
        items, segments, masses, oracle_valid
    )
    oracle_per_row = routing_soft_oracle_per_sample_loss(
        global_outputs, oracle_target
    )
    global_loss = 0.8 * (
        1.2 * torch.dot(prediction_per_row, prediction_weights)
        + 0.35 * torch.dot(oracle_per_row, oracle_weights)
    ) / (1.2 + 0.35)
    global_loss.backward()

    torch.testing.assert_close(global_loss, reference, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        global_scores.grad, reference_scores.grad, rtol=2.0e-6, atol=2.0e-7
    )


@pytest.mark.parametrize("loss_type", ("bce", "brier"))
def test_global_i1_loss_matches_branch_source_reference_value_and_gradient(
    loss_type: str,
):
    items, segments, _routing_groups, labels, evidence = _sources_and_groups()
    rows = labels.numel()
    clean = items[:2]
    # I1 has one equally weighted group per branch.  Observable perturbation
    # sources differ across branches and include unequal row counts plus the
    # fully dead p3 source, matching the formal source-selection semantics.
    branch_scenarios = {
        "api": [items[2], items[3], items[4]],
        "graph": [items[3], items[5], items[6], items[7]],
        "manifest": [items[2], items[4], items[5]],
    }
    groups = [
        {
            "name": f"{branch}:observable",
            "branch": branch,
            "clean": clean,
            "scenario": branch_scenarios[branch],
        }
        for branch in ("api", "graph", "manifest")
    ]
    branch_index = {name: index for index, name in enumerate(("api", "graph", "manifest"))}
    row = torch.arange(rows, dtype=torch.float32).view(-1, 1, 1)
    branch = torch.arange(3, dtype=torch.float32).view(1, -1, 1)
    feature = torch.arange(3, dtype=torch.float32).view(1, 1, -1)
    features = torch.sin(0.37 * row + 0.61 * branch + 0.29 * feature)
    branch_logits = _fixed_branch_probabilities(rows).log()
    segment_by_id = {id(item): segment for item, segment in zip(items, segments)}

    initial_weights = torch.tensor(
        ((0.25, -0.35, 0.15), (-0.2, 0.4, 0.3), (0.1, -0.25, 0.5)),
        dtype=torch.float32,
    )
    initial_biases = torch.tensor((-0.2, 0.1, -0.4), dtype=torch.float32)

    def _reference_objective(
        weights: torch.Tensor, biases: torch.Tensor
    ) -> torch.Tensor:
        group_losses = []
        for group in groups:
            name = group["branch"]
            index = branch_index[name]

            def _source_loss(item: dict) -> torch.Tensor:
                start, end = segment_by_id[id(item)]
                logit = biases[index] + (
                    features[start:end, index] * weights[index]
                ).sum(dim=-1)
                alive = reliability_alive_mask(item["evidence"], name)
                outputs = {
                    f"predicted_reliability_{name}": torch.sigmoid(logit) * alive,
                    f"predicted_reliability_logit_{name}": logit,
                    f"{name}_logits_aux": branch_logits[start:end, index],
                }
                loss, _ = compute_reliability_calibration_loss(
                    outputs,
                    item["labels"],
                    item["evidence"],
                    {"branches": [name], "loss": loss_type},
                    materialize_diagnostics=False,
                )
                return loss

            clean_loss = torch.stack(
                [_source_loss(item) for item in group["clean"]]
            ).mean()
            scenario_loss = torch.stack(
                [_source_loss(item) for item in group["scenario"]]
            ).mean()
            group_losses.append(0.5 * clean_loss + 0.5 * scenario_loss)
        return torch.stack(group_losses).mean()

    reference_weights = initial_weights.clone().requires_grad_(True)
    reference_biases = initial_biases.clone().requires_grad_(True)
    reference = _reference_objective(reference_weights, reference_biases)
    reference.backward()

    global_weights = initial_weights.clone().requires_grad_(True)
    global_biases = initial_biases.clone().requires_grad_(True)
    global_branch_losses = []
    for group in groups:
        name = group["branch"]
        index = branch_index[name]
        selected_ids = {
            id(item)
            for item in [*group["clean"], *group["scenario"]]
        }
        selected_items = [item for item in items if id(item) in selected_ids]
        selected_segments = []
        row_indices = []
        offset = 0
        for item in selected_items:
            start, end = segment_by_id[id(item)]
            selected_segments.append((offset, offset + end - start))
            row_indices.extend(range(start, end))
            offset += end - start
        indices = torch.tensor(row_indices, dtype=torch.long)
        selected_evidence = evidence.index_select(0, indices)
        selected_labels = labels.index_select(0, indices)
        selected_logits = branch_logits.index_select(0, indices)[:, index]
        logit = global_biases[index] + (
            features.index_select(0, indices)[:, index] * global_weights[index]
        ).sum(dim=-1)
        alive = reliability_alive_mask(selected_evidence, name)
        reliability = torch.sigmoid(logit) * alive
        correctness = reliability_correctness_target(
            selected_logits, selected_labels
        )
        per_row = reliability_per_sample_loss(
            reliability,
            logit,
            correctness,
            loss_type=loss_type,
        )
        masses = _compile_posthoc_source_masses(
            [group], selected_items, {"clean": 0.5, "perturb": 0.5}
        )
        row_weights = _compile_posthoc_row_weights(
            selected_items,
            selected_segments,
            masses,
            alive,
        ) / float(len(groups))
        global_branch_losses.append(torch.dot(per_row, row_weights))
    global_loss = torch.stack(global_branch_losses).sum()
    global_loss.backward()

    torch.testing.assert_close(global_loss, reference, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        global_weights.grad,
        reference_weights.grad,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    torch.testing.assert_close(
        global_biases.grad,
        reference_biases.grad,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def _risk_static_outputs(labels: torch.Tensor, evidence: torch.Tensor) -> dict:
    rows = labels.numel()
    has_available = evidence[
        :, EvidenceIndex.API_ALIVE : EvidenceIndex.MANIFEST_ALIVE + 1
    ].bool().any(dim=-1)
    malware = (0.15 + 0.7 * torch.sigmoid(torch.linspace(-2.0, 2.0, rows))).clamp(
        1.0e-4, 1.0 - 1.0e-4
    )
    mixture = torch.stack((1.0 - malware, malware), dim=-1)
    threshold = 0.25
    pattern = torch.tensor((-0.8, threshold, 0.0, 0.6, -0.2, 0.9))
    odds = pattern.repeat(math.ceil(rows / pattern.numel()))[:rows]
    raw_log_prob = torch.stack((torch.zeros_like(odds), odds), dim=-1)
    return {
        "routing_active": torch.ones_like(has_available, dtype=torch.float32),
        "routing_has_available": has_available,
        "routing_mixture_prob": mixture,
        "uncalibrated_final_log_prob": raw_log_prob,
    }


@pytest.mark.parametrize("loss_type", ("bce", "brier"))
@pytest.mark.parametrize(
    "target_type",
    (
        "mixture_argmax_error",
        "threshold_classification_error",
        "threshold_malware_false_negative",
    ),
)
def test_global_risk_loss_matches_source_family_reference_value_and_gradient(
    loss_type: str,
    target_type: str,
):
    items, segments, groups, labels, evidence = _sources_and_groups()
    rows = labels.numel()
    static_outputs = _risk_static_outputs(labels, evidence)
    features = torch.linspace(-0.7, 1.2, rows * 5).view(rows, 5)
    initial_weights = torch.tensor((-0.4, 0.2, -0.1, 0.5, -0.25))
    initial_bias = torch.tensor(-0.3)
    routing_options = {
        "enabled": True,
        "posthoc_refine": True,
        "calibration_weight": 0.9,
        "prediction_loss_weight": 0.0,
        "route_oracle_loss_weight": 0.0,
        "risk_loss_weight": 1.1,
        "risk_loss": loss_type,
        "risk_target": target_type,
        "classification_log_odds_threshold": 0.25,
    }
    config = {
        "reliability_calibration": {"weight": 0.0},
        "probability_calibration": {"weight": 0.0},
        "routing": routing_options,
    }

    reference_weights = initial_weights.clone().requires_grad_(True)
    reference_bias = initial_bias.clone().requires_grad_(True)
    reference_logit = reference_bias + (
        features * F.softplus(reference_weights).view(1, -1)
    ).sum(dim=-1)
    reference_outputs = {
        **static_outputs,
        "routing_risk_probability": torch.sigmoid(reference_logit),
        "routing_risk_logit": reference_logit,
        "routing_risk_training_logit": reference_logit,
    }
    reference = _old_group_reduction(
        reference_outputs, items, segments, groups, config
    )
    reference.backward()

    global_weights = initial_weights.clone().requires_grad_(True)
    global_bias = initial_bias.clone().requires_grad_(True)
    global_logit = global_bias + (
        features * F.softplus(global_weights).view(1, -1)
    ).sum(dim=-1)
    global_outputs = {
        **static_outputs,
        "routing_risk_probability": torch.sigmoid(global_logit),
        "routing_risk_logit": global_logit,
        "routing_risk_training_logit": global_logit,
    }
    target, valid, resolved_loss, resolved_target = routing_risk_target(
        global_outputs,
        labels,
        routing_options,
    )
    assert resolved_loss == loss_type
    assert resolved_target == target_type
    masses = _compile_posthoc_source_masses(groups, items, SCENARIO_PRIOR)
    row_weights = _compile_posthoc_row_weights(
        items, segments, masses, valid
    )
    per_row = routing_risk_per_sample_loss(
        global_outputs["routing_risk_probability"],
        global_logit,
        target,
        valid,
        loss_type=loss_type,
    )
    global_loss = 0.9 * 1.1 * torch.dot(per_row, row_weights)
    global_loss.backward()

    torch.testing.assert_close(global_loss, reference, rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        global_weights.grad, reference_weights.grad, rtol=2.0e-6, atol=2.0e-7
    )
    torch.testing.assert_close(
        global_bias.grad, reference_bias.grad, rtol=2.0e-6, atol=2.0e-7
    )
    if target_type == "threshold_malware_false_negative":
        # The exact-threshold row predicts malware because comparison is >=,
        # and is therefore excluded from the conditional-FN fitting mask.
        assert static_outputs["uncalibrated_final_log_prob"][1, 1].item() == pytest.approx(0.25)
        assert not bool(valid[1])
