from __future__ import annotations

import argparse
import copy
import csv
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fusion.discount_fusion import compute_branch_confidence_proxy
from fusion.reliability_calibration import BRANCH_NAMES
from fusion.train import (
    build_dataset,
    build_loader,
    build_model,
    configure_determinism,
    configure_multiprocessing_sharing,
    deep_update,
    enforce_failed_ratio,
    evaluate,
    load_config,
    select_device,
    set_seed,
)


BRANCH_LOGIT_KEYS = {
    "api": "api_logits_aux",
    "graph": "graph_logits_aux",
    "manifest": "manifest_logits_aux",
}

DEFAULT_VARIANTS: dict[str, tuple[str, ...]] = {
    "full_original": ("api", "graph", "manifest"),
    "api_only": ("api",),
    "graph_only": ("graph",),
    "manifest_only": ("manifest",),
    "api_manifest": ("api", "manifest"),
    "api_graph": ("api", "graph"),
    "no_api": ("graph", "manifest"),
    "no_graph": ("api", "manifest"),
    "no_manifest": ("api", "graph"),
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


def _copy_runtime_paths(target: dict, source: dict) -> dict:
    """Use checkpoint semantics, but caller-provided data/output/runtime paths."""
    out = copy.deepcopy(target)
    for section in ("data", "train", "eval"):
        if section not in source:
            continue
        out.setdefault(section, {})
        if section == "data":
            for key in (
                "root",
                "train_pt_dir",
                "val_pt_dir",
                "test_pt_dir",
                "train_csv",
                "val_csv",
                "test_csv",
                "out_dir",
                "max_failed_ratio",
                "strict_split_integrity",
                "strict_partition_isolation",
                "allow_pt_superset",
            ):
                if key in source[section]:
                    out[section][key] = copy.deepcopy(source[section][key])
        elif section == "train":
            for key in (
                "device",
                "eval_batch_size",
                "batch_size",
                "eval_num_workers",
                "num_workers",
                "pin_memory",
                "allow_pyg_pin_memory",
                "persistent_workers",
                "use_amp",
                "multiprocessing_sharing_strategy",
            ):
                if key in source[section]:
                    out[section][key] = copy.deepcopy(source[section][key])
        elif section == "eval":
            out[section] = deep_update(out.get(section, {}), source[section])
    return out


def _branch_prob(extra: dict[str, Any], branch: str) -> torch.Tensor:
    calibrated = extra.get(f"calibrated_log_prob_{branch}")
    if isinstance(calibrated, torch.Tensor):
        return calibrated.float().exp()
    logits = extra.get(BRANCH_LOGIT_KEYS[branch])
    if not isinstance(logits, torch.Tensor):
        raise KeyError(f"Missing logits for branch {branch!r}")
    return F.softmax(logits.float(), dim=-1)


def _original_weights(extra: dict[str, Any], ref: torch.Tensor) -> torch.Tensor:
    weights = extra.get("fusion_weights")
    if isinstance(weights, torch.Tensor) and weights.ndim == 2 and weights.size(-1) == len(BRANCH_NAMES):
        return weights.float()
    values = []
    for branch in BRANCH_NAMES:
        value = extra.get(f"fusion_weight_{branch}")
        if isinstance(value, torch.Tensor):
            values.append(value.float().view(-1))
        else:
            values.append(torch.full((ref.size(0),), 1.0 / len(BRANCH_NAMES), device=ref.device))
    return torch.stack(values, dim=-1)


def _recompute_acceptance(
    extra: dict[str, Any],
    weights: torch.Tensor,
    final_logits: torch.Tensor,
    aggregation: str,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reliability_values = []
    for branch in BRANCH_NAMES:
        value = extra.get(f"predicted_reliability_{branch}")
        if isinstance(value, torch.Tensor):
            reliability_values.append(value.float().view(-1).clamp(0.0, 1.0))
        else:
            reliability_values.append(torch.ones((weights.size(0),), device=weights.device))
    reliability_matrix = torch.stack(reliability_values, dim=-1)
    total_reliability = (weights * reliability_matrix).sum(dim=-1).clamp(0.0, 1.0)
    proxy = compute_branch_confidence_proxy(final_logits.float(), temperature=1.0, eps=eps)
    uncertainty = proxy["uncertainty_proxy"].clamp(0.0, 1.0)
    conflict = extra.get("effective_conflict")
    if isinstance(conflict, torch.Tensor):
        conflict = conflict.float().view(-1).clamp(0.0, 1.0)
    else:
        conflict = torch.zeros_like(total_reliability)
    components = torch.stack([total_reliability, 1.0 - uncertainty, 1.0 - conflict], dim=-1)
    if aggregation == "min":
        acceptance = components.min(dim=-1).values
    elif aggregation == "product":
        acceptance = components.prod(dim=-1)
    else:
        raise ValueError("acceptance aggregation must be 'min' or 'product'")
    return acceptance.clamp(0.0, 1.0), total_reliability, uncertainty


class BranchSubsetWrapper(nn.Module):
    """Evaluate one trained model after forcing a branch subset at decision time."""

    def __init__(
        self,
        model: nn.Module,
        keep_branches: tuple[str, ...],
        *,
        weight_mode: str,
        acceptance_aggregation: str,
    ):
        super().__init__()
        unknown = sorted(set(keep_branches) - set(BRANCH_NAMES))
        if unknown:
            raise ValueError(f"Unknown branches in subset: {unknown}")
        if not keep_branches:
            raise ValueError("Branch subset must keep at least one branch")
        weight_mode = str(weight_mode).lower()
        if weight_mode not in {"renorm", "uniform"}:
            raise ValueError("weight_mode must be 'renorm' or 'uniform'")
        self.model = model
        self.keep_branches = tuple(keep_branches)
        self.weight_mode = weight_mode
        self.acceptance_aggregation = str(acceptance_aggregation).lower()

    def forward(self, graph_data, return_features: bool = False):
        _logits, extra = self.model(graph_data, return_features=return_features)
        branch_probs = torch.stack([_branch_prob(extra, branch) for branch in BRANCH_NAMES], dim=1)
        ref = branch_probs[:, 0]
        mask = torch.tensor(
            [1.0 if branch in self.keep_branches else 0.0 for branch in BRANCH_NAMES],
            device=ref.device,
            dtype=ref.dtype,
        ).view(1, -1)
        if self.weight_mode == "uniform":
            weights = mask.expand(ref.size(0), -1) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        else:
            original = _original_weights(extra, ref)
            weights = original * mask
            denom = weights.sum(dim=-1, keepdim=True)
            uniform = mask.expand_as(weights) / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            weights = torch.where(denom > 1.0e-8, weights / denom.clamp_min(1.0e-8), uniform)
        final_prob = (weights.unsqueeze(-1) * branch_probs).sum(dim=1)
        final_prob = final_prob / final_prob.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        final_logits = torch.log(final_prob.clamp_min(1.0e-8))
        acceptance, total_reliability, uncertainty = _recompute_acceptance(
            extra,
            weights,
            final_logits,
            self.acceptance_aggregation,
        )
        extra = dict(extra)
        extra["final_prob"] = final_prob
        extra["final_logits"] = final_logits
        extra["fusion_weights"] = weights
        extra["gate_weights"] = weights.detach()
        extra["gate_weights_train"] = weights
        extra["acceptance_score"] = acceptance
        extra["total_reliability"] = total_reliability
        extra["final_uncertainty_proxy"] = uncertainty
        primary_alive: dict[str, torch.Tensor] = {}
        for branch in ("api", "graph", "manifest"):
            value = extra.get(f"{branch}_alive")
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Branch-subset evaluation requires {branch}_alive diagnostics"
                )
            primary_alive[branch] = value.view(-1).to(device=ref.device) > 0.0
        extra["selective_eligible"] = torch.stack(
            [primary_alive[branch] for branch in self.keep_branches],
            dim=-1,
        ).any(dim=-1)
        for index, branch in enumerate(BRANCH_NAMES):
            extra[f"fusion_weight_{branch}"] = weights[:, index]
        return final_logits, extra


def _parse_variant(raw: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in raw:
        if raw not in DEFAULT_VARIANTS:
            raise ValueError(f"Unknown variant {raw!r}; use NAME=api,graph for custom subsets")
        return raw, DEFAULT_VARIANTS[raw]
    name, branches = raw.split("=", 1)
    name = name.strip()
    keep = tuple(branch.strip().lower() for branch in branches.split(",") if branch.strip())
    if not name or not keep:
        raise ValueError(f"Invalid variant spec: {raw!r}")
    return name, keep


def _load_checkpoint_model(cfg: dict, checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("cfg"), dict):
        raise ValueError(
            "The checkpoint must use the current schema and contain a cfg mapping"
        )
    model_cfg = _copy_runtime_paths(ckpt["cfg"], cfg)
    train_ds = build_dataset(model_cfg, "train", is_train=False)
    feature_dim = int(train_ds.feature_dim)
    model = build_model(model_cfg, feature_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_cfg, ckpt


def _scenario_specs(args: argparse.Namespace, cfg: dict) -> list[dict[str, Any]]:
    specs = [{"name": args.split, "split": args.split, "perturb_type": None, "strength": 0.0}]
    if not args.robust:
        return specs
    eval_cfg = cfg.get("eval", {}) or {}
    perturb_tests = list(args.perturb_tests or eval_cfg.get("perturb_tests", []))
    strengths = [float(value) for value in (args.perturb_strengths or eval_cfg.get("perturb_strengths", [0.5]))]
    for perturb in perturb_tests:
        perturb = str(perturb)
        if perturb == "clean":
            continue
        if perturb.endswith("_missing"):
            specs.append({"name": perturb, "split": args.split, "perturb_type": perturb, "strength": 1.0})
        else:
            for strength in strengths:
                specs.append(
                    {
                        "name": f"{perturb}_s{strength:g}",
                        "split": args.split,
                        "perturb_type": perturb,
                        "strength": strength,
                    }
                )
    return specs


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eval-only branch subset diagnostics for a trained tri-modal checkpoint."
    )
    parser.add_argument("--config", required=True, help="Base experiment YAML.")
    parser.add_argument("--extra-config", action="append", default=[], help="Optional override YAML, e.g. _autodl_paths.yaml.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a current pipeline-fitted checkpoint.",
    )
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", default="results/branch_subset_diagnostics")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant name or custom NAME=api,graph. Defaults to a standard diagnostic suite.",
    )
    parser.add_argument("--weight-mode", default="renorm", choices=["renorm", "uniform"])
    parser.add_argument("--robust", action="store_true", help="Also evaluate configured perturbation scenarios.")
    parser.add_argument("--perturb-tests", nargs="*", default=None)
    parser.add_argument("--perturb-strengths", nargs="*", type=float, default=None)
    parser.add_argument("--selective-threshold", type=float, default=None)
    parser.add_argument("--no-checkpoint-threshold", action="store_true")
    parser.add_argument("--dump-rows", action="store_true")
    args = parser.parse_args()

    cfg = load_config([args.config, *args.extra_config])
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    configure_determinism(
        bool(cfg.get("train", {}).get("deterministic", True)),
        strict=bool(cfg.get("train", {}).get("strict_deterministic", False)),
    )
    configure_multiprocessing_sharing(cfg)
    device = select_device(str(cfg.get("train", {}).get("device", "auto")))
    checkpoint = Path(args.checkpoint)
    model, model_cfg, ckpt = _load_checkpoint_model(cfg, checkpoint, device)
    use_amp = bool(model_cfg.get("train", {}).get("use_amp", True))
    threshold = args.selective_threshold
    if threshold is None and not args.no_checkpoint_threshold:
        raw_threshold = ckpt.get("rejection_threshold") if isinstance(ckpt, dict) else None
        threshold = float(raw_threshold) if raw_threshold is not None else None
    aggregation = str((model_cfg.get("fusion", {}) or {}).get("acceptance_aggregation", "product"))
    variants = [_parse_variant(item) for item in args.variant] if args.variant else list(DEFAULT_VARIANTS.items())
    scenarios = _scenario_specs(args, model_cfg)
    out_dir = Path(args.output_dir)
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "split": args.split,
        "weight_mode": args.weight_mode,
        "selective_threshold": threshold,
        "variants": {},
    }

    for scenario in scenarios:
        dataset = build_dataset(
            model_cfg,
            scenario["split"],
            is_train=False,
            perturb_type=scenario["perturb_type"],
            perturb_strength=float(scenario["strength"]),
        )
        loader = build_loader(model_cfg, dataset, is_train=False)
        for variant_name, keep in variants:
            wrapped = BranchSubsetWrapper(
                model,
                keep,
                weight_mode=args.weight_mode,
                acceptance_aggregation=aggregation,
            )
            metrics, rows = evaluate(
                wrapped,
                loader,
                device,
                use_amp,
                f"{scenario['name']}__{variant_name}",
                dump_rows=args.dump_rows,
                selective_threshold=threshold,
            )
            enforce_failed_ratio(metrics, model_cfg, f"{scenario['name']}__{variant_name}")
            record = {
                "scenario": scenario["name"],
                "variant": variant_name,
                "keep_branches": ",".join(keep),
                **metrics,
            }
            all_rows.append(record)
            summary["variants"].setdefault(variant_name, {})[scenario["name"]] = _json_safe(metrics)
            if rows:
                rows_path = out_dir / f"rows_{scenario['name']}__{variant_name}.csv"
                _write_metrics_csv(rows_path, rows)

    _write_metrics_csv(out_dir / "branch_subset_metrics.csv", all_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "branch_subset_summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(_json_safe(summary), f, sort_keys=False)
    print(f"Wrote branch subset diagnostics to {out_dir}")


if __name__ == "__main__":
    main()

