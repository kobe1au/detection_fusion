from __future__ import annotations

import torch
import torch.nn.functional as F

from fusion.constants import EvidenceIndex
from fusion.quality import compute_manifest_code_support_and_conflict
from fusion.semantic_categories import SEMANTIC_CATEGORY_DIM


def scalar_attr(graph_data, name: str, batch_size: int, device, dtype, default: float) -> torch.Tensor:
    value = getattr(graph_data, name, None)
    if isinstance(value, torch.Tensor):
        out = value.to(device=device, dtype=dtype).view(batch_size, -1)
        if out.size(1) > 1:
            out = out[:, :1]
        return torch.nan_to_num(out.clamp(0.0, 1.0), nan=float(default), posinf=1.0, neginf=0.0)
    return torch.full((batch_size, 1), float(default), device=device, dtype=dtype)


def observable_attr(
    graph_data,
    name: str,
    legacy_name: str | None,
    batch_size: int,
    device,
    dtype,
    default: float,
) -> torch.Tensor:
    if isinstance(getattr(graph_data, name, None), torch.Tensor):
        return scalar_attr(graph_data, name, batch_size, device, dtype, default)
    if legacy_name and isinstance(getattr(graph_data, legacy_name, None), torch.Tensor):
        return scalar_attr(graph_data, legacy_name, batch_size, device, dtype, default)
    return scalar_attr(graph_data, name, batch_size, device, dtype, default)


def semantic_counts_attr(graph_data, name: str, batch_size: int, device, dtype) -> torch.Tensor:
    value = getattr(graph_data, name, None)
    if not isinstance(value, torch.Tensor):
        return torch.zeros((batch_size, SEMANTIC_CATEGORY_DIM), device=device, dtype=dtype)
    out = value.to(device=device, dtype=dtype)
    out = out.view(1, -1).expand(batch_size, -1) if out.ndim == 1 else out.view(batch_size, -1)
    if out.size(1) != SEMANTIC_CATEGORY_DIM:
        return torch.zeros((batch_size, SEMANTIC_CATEGORY_DIM), device=device, dtype=dtype)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def confidence(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.detach(), dim=-1).max(dim=-1, keepdim=True).values.clamp(0.0, 1.0)


def cosine_counts(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    valid = (a.abs().sum(dim=-1, keepdim=True) > 0) & (b.abs().sum(dim=-1, keepdim=True) > 0)
    sim = F.cosine_similarity(a.float(), b.float(), dim=-1).view(-1, 1).clamp(0.0, 1.0)
    return torch.where(valid, sim, torch.zeros_like(sim))


def _fallback_manifest_signals(
    api_counts: torch.Tensor,
    graph_counts: torch.Tensor,
    manifest_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [
        compute_manifest_code_support_and_conflict(api_counts[i], graph_counts[i], manifest_counts[i])
        for i in range(api_counts.size(0))
    ]
    return tuple(
        api_counts.new_tensor([value[index] for value in values]).view(-1, 1)
        for index in range(3)
    )


def build_evidence(
    graph_data,
    api_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    manifest_logits: torch.Tensor,
    api_emb: torch.Tensor,
    graph_emb: torch.Tensor,
    manifest_emb: torch.Tensor,
    *,
    use_consistency_evidence: bool,
    use_conflict_evidence: bool,
    use_perturbation_evidence: bool = False,
    diagnostics_only: bool = False,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    del api_emb, graph_emb, manifest_emb
    if use_perturbation_evidence:
        raise ValueError(
            "Synthetic pert_* metadata is diagnostic-only and cannot be read by build_evidence."
        )

    batch_size = api_logits.size(0)
    device = api_logits.device
    dtype = api_logits.dtype

    api_integrity = observable_attr(graph_data, "api_integrity", "q_api", batch_size, device, dtype, 0.0)
    graph_integrity = observable_attr(graph_data, "graph_integrity", "q_graph", batch_size, device, dtype, 0.0)
    manifest_integrity = observable_attr(graph_data, "manifest_integrity", "q_manifest", batch_size, device, dtype, 0.0)
    code_integrity = observable_attr(
        graph_data,
        "code_integrity",
        None,
        batch_size,
        device,
        dtype,
        0.0,
    )
    missing_code_integrity = not isinstance(getattr(graph_data, "code_integrity", None), torch.Tensor)
    if missing_code_integrity:
        code_integrity = torch.sqrt(api_integrity * graph_integrity).clamp(0.0, 1.0)

    anchor_support = observable_attr(
        graph_data,
        "api_graph_anchor_support",
        "q_align",
        batch_size,
        device,
        dtype,
        0.0,
    )
    api_alive = observable_attr(graph_data, "api_alive", None, batch_size, device, dtype, 0.0)
    graph_alive = observable_attr(graph_data, "graph_alive", None, batch_size, device, dtype, 0.0)
    manifest_alive = observable_attr(graph_data, "manifest_alive", None, batch_size, device, dtype, 0.0)
    if not isinstance(getattr(graph_data, "api_alive", None), torch.Tensor):
        api_alive = (api_integrity > 0.0).to(dtype=dtype)
    if not isinstance(getattr(graph_data, "graph_alive", None), torch.Tensor):
        graph_alive = (graph_integrity > 0.0).to(dtype=dtype)
    if not isinstance(getattr(graph_data, "manifest_alive", None), torch.Tensor):
        manifest_alive = (manifest_integrity > 0.0).to(dtype=dtype)

    api_counts = semantic_counts_attr(graph_data, "api_semantic_category_counts", batch_size, device, dtype)
    graph_counts = semantic_counts_attr(graph_data, "graph_semantic_category_counts", batch_size, device, dtype)
    manifest_counts = semantic_counts_attr(graph_data, "manifest_category_counts", batch_size, device, dtype)
    missing_manifest_support = not isinstance(
        getattr(graph_data, "manifest_code_support", None), torch.Tensor
    )
    missing_manifest_conflict = not isinstance(
        getattr(graph_data, "manifest_to_code_conflict", None), torch.Tensor
    )
    missing_code_conflict = not isinstance(
        getattr(graph_data, "code_to_manifest_conflict", None), torch.Tensor
    )
    # Current PTs materialize all three observable relations in the Dataset.
    # Their fallback used to be recomputed unconditionally with a Python loop
    # over every GPU batch, even though the result was then discarded.  Only
    # execute that path for genuinely incomplete direct-call inputs.
    if missing_manifest_support or missing_manifest_conflict or missing_code_conflict:
        (
            fallback_support,
            fallback_manifest_conflict,
            fallback_code_conflict,
        ) = _fallback_manifest_signals(
            api_counts,
            graph_counts,
            manifest_counts,
        )
    else:
        fallback_support = fallback_manifest_conflict = fallback_code_conflict = None
    manifest_support = observable_attr(
        graph_data,
        "manifest_code_support",
        None,
        batch_size,
        device,
        dtype,
        0.0,
    )
    manifest_conflict = observable_attr(
        graph_data,
        "manifest_to_code_conflict",
        None,
        batch_size,
        device,
        dtype,
        0.0,
    )
    code_conflict = observable_attr(
        graph_data,
        "code_to_manifest_conflict",
        None,
        batch_size,
        device,
        dtype,
        0.0,
    )
    if missing_manifest_support:
        assert fallback_support is not None
        manifest_support = fallback_support
    if missing_manifest_conflict:
        assert fallback_manifest_conflict is not None
        manifest_conflict = fallback_manifest_conflict
    if missing_code_conflict:
        assert fallback_code_conflict is not None
        code_conflict = fallback_code_conflict

    evidence_anchor = anchor_support if use_consistency_evidence else torch.zeros_like(anchor_support)
    evidence_manifest_support = manifest_support if use_consistency_evidence else torch.zeros_like(manifest_support)
    evidence_manifest_conflict = manifest_conflict if use_conflict_evidence else torch.zeros_like(manifest_conflict)
    evidence_code_conflict = code_conflict if use_conflict_evidence else torch.zeros_like(code_conflict)

    api_total_pipeline_coverage = scalar_attr(
        graph_data, "api_encoder_coverage", batch_size, device, dtype, 1.0
    )
    api_extractor_coverage = scalar_attr(
        graph_data, "api_extractor_coverage", batch_size, device, dtype, 1.0
    )
    api_runtime_encoder_coverage = scalar_attr(
        graph_data, "api_runtime_encoder_coverage", batch_size, device, dtype, 1.0
    )
    # Reliability should describe what the model receives from the already
    # constructed PT representation. Fixed per-method/per-DEX extraction
    # budgets are representation design choices, not sample-specific encoder
    # failures. Keep their combined ratio for diagnostics, while the formal
    # visibility modifier consumes only the runtime encoder coverage.
    api_encoder_coverage = api_runtime_encoder_coverage
    graph_encoder_coverage = scalar_attr(
        graph_data, "graph_encoder_coverage", batch_size, device, dtype, 1.0
    )
    graph_feature_valid_ratio = scalar_attr(
        graph_data, "graph_feature_valid_ratio", batch_size, device, dtype, 1.0
    )
    evidence = None
    if not diagnostics_only:
        evidence = torch.cat(
            [
                api_integrity,
                graph_integrity,
                manifest_integrity,
                code_integrity,
                evidence_anchor,
                evidence_manifest_support,
                evidence_manifest_conflict,
                evidence_code_conflict,
                api_alive,
                graph_alive,
                manifest_alive,
                api_encoder_coverage,
                graph_encoder_coverage,
            ],
            dim=-1,
        )
        if evidence.size(-1) != EvidenceIndex.BASE_DIM:
            raise RuntimeError(
                f"Observable evidence dimension mismatch: built {evidence.size(-1)}, "
                f"expected {EvidenceIndex.BASE_DIM}"
            )

    api_manifest_consistency = cosine_counts(api_counts, manifest_counts)
    graph_manifest_consistency = cosine_counts(graph_counts, manifest_counts)
    api_truncated = scalar_attr(
        graph_data,
        "api_truncated_by_encoder_budget",
        batch_size,
        device,
        dtype,
        0.0,
    )
    api_extractor_truncated = scalar_attr(
        graph_data,
        "api_truncated_by_extractor_budget",
        batch_size,
        device,
        dtype,
        0.0,
    )
    api_integrity_before_budget = scalar_attr(
        graph_data,
        "api_integrity_before_encoder_budget",
        batch_size,
        device,
        dtype,
        0.0,
    )
    graph_truncated = scalar_attr(
        graph_data,
        "graph_truncated_by_encoder_budget",
        batch_size,
        device,
        dtype,
        0.0,
    )
    graph_integrity_before_budget = scalar_attr(
        graph_data,
        "graph_integrity_before_encoder_budget",
        batch_size,
        device,
        dtype,
        0.0,
    )
    diagnostics = {
        "api_integrity": api_integrity.detach().view(batch_size),
        "api_encoder_coverage": api_encoder_coverage.detach().view(batch_size),
        "api_total_pipeline_coverage": api_total_pipeline_coverage.detach().view(batch_size),
        "api_extractor_coverage": api_extractor_coverage.detach().view(batch_size),
        "api_runtime_encoder_coverage": api_runtime_encoder_coverage.detach().view(batch_size),
        "effective_api_integrity": (api_integrity * api_encoder_coverage).detach().view(batch_size),
        "api_truncated_by_extractor_budget": api_extractor_truncated.detach().view(batch_size),
        "api_truncated_by_encoder_budget": api_truncated.detach().view(batch_size),
        "api_integrity_before_encoder_budget": api_integrity_before_budget.detach().view(batch_size),
        "graph_integrity": graph_integrity.detach().view(batch_size),
        "graph_encoder_coverage": graph_encoder_coverage.detach().view(batch_size),
        "graph_feature_valid_ratio": graph_feature_valid_ratio.detach().view(batch_size),
        "effective_graph_integrity": (graph_integrity * graph_encoder_coverage).detach().view(batch_size),
        "graph_truncated_by_encoder_budget": graph_truncated.detach().view(batch_size),
        "graph_integrity_before_encoder_budget": graph_integrity_before_budget.detach().view(batch_size),
        "manifest_integrity": manifest_integrity.detach().view(batch_size),
        "effective_manifest_integrity": manifest_integrity.detach().view(batch_size),
        "code_integrity": code_integrity.detach().view(batch_size),
        "api_graph_anchor_support": anchor_support.detach().view(batch_size),
        "manifest_code_support": manifest_support.detach().view(batch_size),
        "manifest_to_code_conflict": manifest_conflict.detach().view(batch_size),
        "code_to_manifest_conflict": code_conflict.detach().view(batch_size),
        "api_alive": api_alive.detach().view(batch_size),
        "graph_alive": graph_alive.detach().view(batch_size),
        "manifest_alive": manifest_alive.detach().view(batch_size),
        # Compatibility aliases for existing losses and diagnostics.
        "q_api": api_integrity.detach().view(batch_size),
        "q_graph": graph_integrity.detach().view(batch_size),
        "q_manifest": manifest_integrity.detach().view(batch_size),
        "q_align": anchor_support.detach().view(batch_size),
        "api_manifest_consistency": api_manifest_consistency.detach().view(batch_size),
        "graph_manifest_consistency": graph_manifest_consistency.detach().view(batch_size),
        "api_confidence": confidence(api_logits).to(dtype=dtype).detach().view(batch_size),
        "graph_confidence": confidence(graph_logits).to(dtype=dtype).detach().view(batch_size),
        "manifest_confidence": confidence(manifest_logits).to(dtype=dtype).detach().view(batch_size),
        "api_semantic_category_counts": api_counts.detach(),
        "graph_semantic_category_counts": graph_counts.detach(),
        "manifest_category_counts": manifest_counts.detach(),
        "api_category_counts": api_counts.detach(),
        "graph_category_counts": graph_counts.detach(),
        "gate_uses_perturbation_evidence": torch.zeros((batch_size,), device=device, dtype=dtype),
    }
    return evidence, diagnostics
