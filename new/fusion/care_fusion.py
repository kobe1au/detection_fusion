from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


MODALITY_NAMES = ("api", "graph", "manifest")
PATH_NAMES = ("agm", "ag", "am", "gm")
PATH_MODALITIES = {
    "agm": ("api", "graph", "manifest"),
    "ag": ("api", "graph"),
    "am": ("api", "manifest"),
    "gm": ("graph", "manifest"),
}
PATH_INDEX = {name: index for index, name in enumerate(PATH_NAMES)}


def binary_log_odds(logits: torch.Tensor) -> torch.Tensor:
    """Return malware-versus-benign log-odds from binary class logits."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError("binary logits must be a tensor")
    if logits.ndim != 2 or logits.size(-1) != 2:
        raise ValueError(
            "CARE-Droid requires binary logits with shape [B, 2], "
            f"got {tuple(logits.shape)}"
        )
    return logits[:, 1] - logits[:, 0]


def hard_predict_log_odds(log_odds: torch.Tensor) -> torch.Tensor:
    """Apply CARE's public hard rule to precomputed binary log-odds."""

    if not isinstance(log_odds, torch.Tensor):
        raise TypeError("binary log-odds must be a tensor")
    if not log_odds.is_floating_point():
        raise TypeError("binary log-odds must be floating point")
    return log_odds.ge(0.0).long()


def hard_predict(logits: torch.Tensor) -> torch.Tensor:
    """Make the one public binary decision used by training and evaluation.

    A zero malware-versus-benign log-odds is deliberately assigned to malware.
    This makes every fixed-path correctness target and every routing
    disagreement use the same conservative tie rule.
    """

    return hard_predict_log_odds(binary_log_odds(logits))


def _validate_embeddings(
    embeddings: Mapping[str, torch.Tensor],
    embedding_dims: Mapping[str, int],
) -> tuple[int, torch.device, torch.dtype]:
    if tuple(embeddings) != MODALITY_NAMES and set(embeddings) != set(
        MODALITY_NAMES
    ):
        raise ValueError(
            "CARE embeddings must contain exactly API, Graph, and Manifest"
        )
    reference = embeddings["api"]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 2:
        raise ValueError("CARE API embedding must have shape [B, D]")
    batch_size = reference.size(0)
    for name in MODALITY_NAMES:
        value = embeddings[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.size(0) != batch_size
            or value.size(1) != int(embedding_dims[name])
        ):
            raise ValueError(
                f"CARE {name} embedding must have shape "
                f"[B, {int(embedding_dims[name])}]"
            )
        if value.device != reference.device or value.dtype != reference.dtype:
            raise ValueError(
                "CARE embeddings must share one device and floating dtype"
            )
    return batch_size, reference.device, reference.dtype


def path_availability(modality_alive: torch.Tensor) -> torch.Tensor:
    """Derive AGM/AG/AM/GM availability from three modality alive bits."""

    if (
        not isinstance(modality_alive, torch.Tensor)
        or modality_alive.ndim != 2
        or modality_alive.size(-1) != len(MODALITY_NAMES)
    ):
        raise ValueError("modality_alive must have shape [B, 3]")
    modality_alive = modality_alive.bool()
    api, graph, manifest = modality_alive.unbind(dim=-1)
    return torch.stack(
        [
            api & graph & manifest,
            api & graph,
            api & manifest,
            graph & manifest,
        ],
        dim=-1,
    )


class CAREPathHeads(nn.Module):
    """Four independent lightweight classifiers over shared encoder outputs."""

    def __init__(
        self,
        embedding_dims: Mapping[str, int],
        *,
        class_count: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if set(embedding_dims) != set(MODALITY_NAMES):
            raise ValueError(
                "CARE path heads require exactly API, Graph, and Manifest dimensions"
            )
        if int(class_count) != 2:
            raise ValueError("CARE-Droid currently supports binary classification only")
        if int(hidden_dim) <= 0:
            raise ValueError("CARE path hidden_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("CARE path dropout must lie within [0, 1)")
        self.embedding_dims = {
            name: int(embedding_dims[name]) for name in MODALITY_NAMES
        }
        self.class_count = 2
        self.heads = nn.ModuleDict()
        for path_name in PATH_NAMES:
            input_dim = sum(
                self.embedding_dims[name] for name in PATH_MODALITIES[path_name]
            )
            self.heads[path_name] = nn.Sequential(
                nn.Linear(input_dim, int(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(int(hidden_dim), self.class_count),
            )

    def forward(
        self,
        embeddings: Mapping[str, torch.Tensor],
        modality_alive: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        batch_size, _device, dtype = _validate_embeddings(
            embeddings, self.embedding_dims
        )
        if (
            not isinstance(modality_alive, torch.Tensor)
            or modality_alive.shape != (batch_size, len(MODALITY_NAMES))
        ):
            raise ValueError("modality_alive must have shape [B, 3]")
        available = path_availability(modality_alive)
        masked_embeddings = {
            name: embeddings[name]
            * modality_alive[:, index : index + 1].to(dtype=dtype)
            for index, name in enumerate(MODALITY_NAMES)
        }
        logits: dict[str, torch.Tensor] = {}
        for path_index, path_name in enumerate(PATH_NAMES):
            path_input = torch.cat(
                [
                    masked_embeddings[name]
                    for name in PATH_MODALITIES[path_name]
                ],
                dim=-1,
            )
            raw_logits = self.heads[path_name](path_input)
            # An unavailable path carries no class evidence. In particular,
            # its classifier bias must not leak into the shared risk context.
            logits[path_name] = torch.where(
                available[:, path_index : path_index + 1],
                raw_logits,
                torch.zeros_like(raw_logits),
            )
        return logits, available


class CAREPathRiskHead(nn.Module):
    """One shared low-capacity fixed-path correctness estimator.

    For each candidate path the input is exactly four normalized path
    log-odds, three modality-alive bits, and a four-dimensional path one-hot.
    The returned value is a correctness probability, not a routing weight.
    """

    feature_dim = len(PATH_NAMES) + len(MODALITY_NAMES) + len(PATH_NAMES)

    def __init__(self, *, hidden_dim: int = 16) -> None:
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("CARE risk hidden_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer(
            "log_odds_center", torch.zeros(len(PATH_NAMES))
        )
        self.register_buffer(
            "log_odds_scale", torch.ones(len(PATH_NAMES))
        )
        self.register_buffer(
            "normalization_is_fitted", torch.tensor(False, dtype=torch.bool)
        )

    def set_log_odds_normalization(
        self,
        center: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        center = torch.as_tensor(
            center,
            device=self.log_odds_center.device,
            dtype=self.log_odds_center.dtype,
        ).view(-1)
        scale = torch.as_tensor(
            scale,
            device=self.log_odds_scale.device,
            dtype=self.log_odds_scale.dtype,
        ).view(-1)
        expected = len(PATH_NAMES)
        if center.numel() != expected or scale.numel() != expected:
            raise ValueError(
                f"CARE log-odds normalization requires {expected} center/scale values"
            )
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("CARE log-odds normalization must be finite")
        if not scale.gt(0.0).all():
            raise ValueError("CARE log-odds normalization scale must be positive")
        self.log_odds_center.copy_(center)
        self.log_odds_scale.copy_(scale)
        self.normalization_is_fitted.fill_(True)

    def normalize(
        self,
        path_log_odds: torch.Tensor,
        path_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_common(path_log_odds, None)
        normalized = (
            path_log_odds - self.log_odds_center.to(path_log_odds)
        ) / self.log_odds_scale.to(path_log_odds)
        if path_available is not None:
            if path_available.shape != path_log_odds.shape:
                raise ValueError("path_available must have shape [B, 4]")
            normalized = torch.where(
                path_available.to(device=path_log_odds.device).bool(),
                normalized,
                torch.zeros_like(normalized),
            )
        return normalized

    @staticmethod
    def _validate_common(
        path_log_odds: torch.Tensor,
        modality_alive: torch.Tensor | None,
    ) -> int:
        if (
            not isinstance(path_log_odds, torch.Tensor)
            or path_log_odds.ndim != 2
            or path_log_odds.size(-1) != len(PATH_NAMES)
        ):
            raise ValueError("path_log_odds must have shape [B, 4]")
        if not torch.isfinite(path_log_odds).all():
            raise ValueError("path_log_odds must be finite")
        batch_size = path_log_odds.size(0)
        if modality_alive is not None:
            if (
                not isinstance(modality_alive, torch.Tensor)
                or modality_alive.shape
                != (batch_size, len(MODALITY_NAMES))
            ):
                raise ValueError("modality_alive must have shape [B, 3]")
        return batch_size

    def _features(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
        candidate_path_index: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        batch_size = self._validate_common(
            normalized_log_odds, modality_alive
        )
        if (
            not isinstance(candidate_path_index, torch.Tensor)
            or candidate_path_index.numel() != batch_size
        ):
            raise ValueError(
                "candidate_path_index must contain exactly one index per sample"
            )
        candidate_path_index = candidate_path_index.to(
            device=normalized_log_odds.device,
            dtype=torch.long,
        ).view(batch_size)
        if (
            candidate_path_index.lt(0).any()
            or candidate_path_index.ge(len(PATH_NAMES)).any()
        ):
            raise ValueError("candidate_path_index must lie within [0, 3]")
        path_one_hot = F.one_hot(
            candidate_path_index, num_classes=len(PATH_NAMES)
        ).to(dtype=normalized_log_odds.dtype)
        features = torch.cat(
            [
                normalized_log_odds.detach(),
                modality_alive.to(
                    device=normalized_log_odds.device,
                    dtype=normalized_log_odds.dtype,
                ).detach(),
                path_one_hot,
            ],
            dim=-1,
        )
        return features, batch_size

    def correctness_logits(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
        candidate_path_index: torch.Tensor,
    ) -> torch.Tensor:
        features, batch_size = self._features(
            normalized_log_odds,
            modality_alive,
            candidate_path_index,
        )
        return self.net(features).view(batch_size)

    def forward(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
        candidate_path_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return fixed-path correctness probabilities for candidate paths."""

        return torch.sigmoid(
            self.correctness_logits(
                normalized_log_odds,
                modality_alive,
                candidate_path_index,
            )
        )

    def score_all(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self._validate_common(
            normalized_log_odds, modality_alive
        )
        candidate = torch.arange(
            len(PATH_NAMES), device=normalized_log_odds.device
        ).view(1, -1).expand(batch_size, -1)
        expanded_odds = normalized_log_odds.unsqueeze(1).expand(
            -1, len(PATH_NAMES), -1
        )
        expanded_alive = modality_alive.unsqueeze(1).expand(
            -1, len(PATH_NAMES), -1
        )
        return self(
            expanded_odds.reshape(-1, len(PATH_NAMES)),
            expanded_alive.reshape(-1, len(MODALITY_NAMES)),
            candidate.reshape(-1),
        ).view(batch_size, len(PATH_NAMES))

    def score_all_logits(
        self,
        normalized_log_odds: torch.Tensor,
        modality_alive: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = self._validate_common(
            normalized_log_odds, modality_alive
        )
        candidate = torch.arange(
            len(PATH_NAMES), device=normalized_log_odds.device
        ).view(1, -1).expand(batch_size, -1)
        expanded_odds = normalized_log_odds.unsqueeze(1).expand(
            -1, len(PATH_NAMES), -1
        )
        expanded_alive = modality_alive.unsqueeze(1).expand(
            -1, len(PATH_NAMES), -1
        )
        return self.correctness_logits(
            expanded_odds.reshape(-1, len(PATH_NAMES)),
            expanded_alive.reshape(-1, len(MODALITY_NAMES)),
            candidate.reshape(-1),
        ).view(batch_size, len(PATH_NAMES))

@dataclass(frozen=True)
class CARERoutingOutput:
    selected_path_index: torch.Tensor
    selected_logits: torch.Tensor
    selected_score: torch.Tensor
    reject: torch.Tensor
    disagreement_with_agm: torch.Tensor


def route_with_agm_anchor(
    path_logits: Mapping[str, torch.Tensor],
    path_available: torch.Tensor,
    correctness: torch.Tensor,
) -> CARERoutingOutput:
    """Route by availability, using AGM as the three-view decision anchor.

    With all three modalities alive, AGM remains selected unless an available
    pair both disagrees with AGM and has strictly higher predicted correctness.
    With exactly two alive, the unique pair is selected without consulting the
    learned head. With fewer than two alive, the sample is explicitly rejected.
    """

    if set(path_logits) != set(PATH_NAMES):
        raise ValueError("path_logits must contain exactly AGM, AG, AM, and GM")
    reference = path_logits["agm"]
    binary_log_odds(reference)
    batch_size = reference.size(0)
    if path_available.shape != (batch_size, len(PATH_NAMES)):
        raise ValueError("path_available must have shape [B, 4]")
    if correctness.shape != (batch_size, len(PATH_NAMES)):
        raise ValueError("correctness must have shape [B, 4]")
    stacked_logits = torch.stack(
        [path_logits[name] for name in PATH_NAMES], dim=1
    )
    path_predictions = torch.stack(
        [hard_predict(path_logits[name]) for name in PATH_NAMES], dim=-1
    )
    all_three = path_available[:, PATH_INDEX["agm"]]
    pair_available = path_available[:, 1:]
    alive_count = pair_available.sum(dim=-1)
    exactly_two = (~all_three) & alive_count.eq(1)
    reject = (~all_three) & (~exactly_two)

    selected = torch.full(
        (batch_size,), -1, device=reference.device, dtype=torch.long
    )
    selected = torch.where(
        all_three,
        torch.full_like(selected, PATH_INDEX["agm"]),
        selected,
    )
    unique_pair = pair_available.to(dtype=torch.long).argmax(dim=-1) + 1
    selected = torch.where(exactly_two, unique_pair, selected)

    disagreement = path_predictions[:, 1:].ne(
        path_predictions[:, :1]
    )
    candidate_mask = torch.cat(
        [
            all_three.unsqueeze(-1),
            all_three.unsqueeze(-1) & pair_available & disagreement,
        ],
        dim=-1,
    )
    candidate_score = correctness.masked_fill(
        ~candidate_mask, torch.finfo(correctness.dtype).min
    )
    best_three_alive = candidate_score.argmax(dim=-1)
    selected = torch.where(all_three, best_three_alive, selected)

    safe_selected = selected.clamp_min(0)
    selected_logits = stacked_logits[
        torch.arange(batch_size, device=reference.device), safe_selected
    ]
    selected_score = correctness[
        torch.arange(batch_size, device=reference.device), safe_selected
    ]
    selected_logits = torch.where(
        reject.unsqueeze(-1), torch.zeros_like(selected_logits), selected_logits
    )
    selected_score = torch.where(
        reject, torch.zeros_like(selected_score), selected_score
    )
    return CARERoutingOutput(
        selected_path_index=selected,
        selected_logits=selected_logits,
        selected_score=selected_score,
        reject=reject,
        disagreement_with_agm=disagreement,
    )
