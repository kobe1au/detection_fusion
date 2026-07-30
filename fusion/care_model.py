from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
import torch.nn as nn

from fusion.care_fusion import (
    MODALITY_NAMES,
    PATH_INDEX,
    PATH_MODALITIES,
    PATH_NAMES,
    CAREPathHeads,
    CAREPathRiskHead,
    binary_log_odds,
    route_with_agm_anchor,
)
from fusion.evidence import build_fusion_availability_and_diagnostics
from fusion.modality_encoders import TriModalEncoderBackbone


class CareDroidModel(TriModalEncoderBackbone):
    """The single model implementation owned by the CARE-Droid protocol.

    The model contains only the shared three-modality encoder backbone, four
    fixed path classifiers (AGM, AG, AM, and GM), and one shared path-risk
    head.  Baseline branch heads and baseline fusion modules deliberately do
    not exist on this class.
    """

    def __init__(
        self,
        *,
        in_feat_dim: int = 515,
        num_classes: int = 2,
        api_num_hash_buckets: int = 8192,
        api_type_vocab_size: int = 16,
        api_emb_dim: int = 128,
        api_hidden_dim: int = 256,
        api_dropout: float = 0.15,
        api_encoder_type: str = "transformer",
        api_layers: int = 2,
        api_heads: int = 4,
        api_max_seq_len: int = 1024,
        graph_emb_dim: int = 128,
        graph_hidden: int = 128,
        graph_heads: int = 4,
        graph_layers: int = 2,
        graph_encoder_type: str = "gatv2",
        max_nodes_gnn: int = 12288,
        use_graph_behavior_hint: bool = False,
        manifest_in_dim: int = 256,
        manifest_emb_dim: int = 128,
        manifest_hidden_dim: int = 256,
        manifest_dropout: float = 0.1,
    ) -> None:
        if int(num_classes) != 2:
            raise ValueError(
                "CARE-Droid currently supports binary classification only"
            )
        super().__init__(
            in_feat_dim=in_feat_dim,
            api_num_hash_buckets=api_num_hash_buckets,
            api_type_vocab_size=api_type_vocab_size,
            api_emb_dim=api_emb_dim,
            api_hidden_dim=api_hidden_dim,
            api_dropout=api_dropout,
            api_encoder_type=api_encoder_type,
            api_layers=api_layers,
            api_heads=api_heads,
            api_max_seq_len=api_max_seq_len,
            graph_emb_dim=graph_emb_dim,
            graph_hidden=graph_hidden,
            graph_heads=graph_heads,
            graph_layers=graph_layers,
            graph_encoder_type=graph_encoder_type,
            max_nodes_gnn=max_nodes_gnn,
            use_graph_behavior_hint=use_graph_behavior_hint,
            manifest_in_dim=manifest_in_dim,
            manifest_emb_dim=manifest_emb_dim,
            manifest_hidden_dim=manifest_hidden_dim,
            manifest_dropout=manifest_dropout,
        )
        self.fusion_mode = "care_droid"
        self.num_classes = 2
        self.care_path_heads = CAREPathHeads(
            {
                "api": self.api_emb_dim,
                "graph": self.graph_emb_dim,
                "manifest": self.manifest_emb_dim,
            },
            class_count=self.num_classes,
        )
        self.care_risk_head = CAREPathRiskHead()
        self._care_risk_active = False

    def set_care_risk_active(self, enabled: bool) -> None:
        """Enable routing only after its log-odds normalizer has been fitted."""

        enabled = bool(enabled)
        if enabled and not bool(
            self.care_risk_head.normalization_is_fitted.item()
        ):
            raise RuntimeError(
                "CARE risk routing cannot be activated before log-odds "
                "normalization is fitted"
            )
        self._care_risk_active = enabled

    @property
    def care_risk_active(self) -> bool:
        return bool(self._care_risk_active)

    def _care_stage_a_modules(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return the exact modules owned by the clean Stage-A artifact."""

        modules = (
            ("api_encoder", self.api_encoder),
            ("graph_encoder", self.graph_encoder),
            ("manifest_encoder", self.manifest_encoder),
            ("care_path_heads", self.care_path_heads),
        )
        stage_parameter_ids = {
            id(parameter)
            for _module_name, module in modules
            for parameter in module.parameters()
        }
        risk_parameter_ids = {
            id(parameter) for parameter in self.care_risk_head.parameters()
        }
        if stage_parameter_ids & risk_parameter_ids:
            raise RuntimeError(
                "CARE Stage-A and risk-head parameter partitions overlap"
            )
        all_parameter_ids = {
            id(parameter) for parameter in self.parameters()
        }
        if stage_parameter_ids | risk_parameter_ids != all_parameter_ids:
            unowned = sorted(
                name
                for name, parameter in self.named_parameters()
                if id(parameter)
                not in (stage_parameter_ids | risk_parameter_ids)
            )
            raise RuntimeError(
                "CARE model contains parameters outside the declared Stage-A "
                f"and risk-head partitions: {unowned}"
            )

        stage_state_keys = {
            f"{module_name}.{local_key}"
            for module_name, module in modules
            for local_key in module.state_dict()
        }
        risk_state_keys = {
            f"care_risk_head.{local_key}"
            for local_key in self.care_risk_head.state_dict()
        }
        full_state_keys = set(self.state_dict())
        if stage_state_keys & risk_state_keys:
            raise RuntimeError(
                "CARE Stage-A and risk-head state partitions overlap"
            )
        if stage_state_keys | risk_state_keys != full_state_keys:
            undeclared = sorted(
                full_state_keys - stage_state_keys - risk_state_keys
            )
            raise RuntimeError(
                "CARE model contains state outside the declared Stage-A and "
                f"risk-head partitions: {undeclared}"
            )
        return modules

    def care_stage_a_state_keys(self) -> tuple[str, ...]:
        """Return the exact ordered key contract for a Stage-A artifact."""

        keys: list[str] = []
        for module_name, module in self._care_stage_a_modules():
            keys.extend(
                f"{module_name}.{local_key}"
                for local_key in module.state_dict()
            )
        if len(keys) != len(set(keys)):
            raise RuntimeError(
                "CARE Stage-A state contains duplicate keys"
            )
        if any(key.startswith("care_risk_head.") for key in keys):
            raise RuntimeError(
                "CARE Stage-A state must exclude the path-risk head"
            )
        return tuple(keys)

    def care_stage_a_state_dict(
        self,
    ) -> OrderedDict[str, torch.Tensor]:
        """Materialize an independent encoder-plus-path-head artifact."""

        state: OrderedDict[str, torch.Tensor] = OrderedDict()
        for module_name, module in self._care_stage_a_modules():
            for local_key, value in module.state_dict().items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "CARE Stage-A artifacts support tensor parameters and "
                        f"buffers only; {module_name}.{local_key} has type "
                        f"{type(value).__name__}"
                    )
                state[f"{module_name}.{local_key}"] = (
                    value.detach().clone()
                )
        expected = self.care_stage_a_state_keys()
        if tuple(state) != expected:
            raise RuntimeError(
                "CARE Stage-A state order disagrees with its key contract"
            )
        return state

    def load_care_stage_a_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly load Stage A without reading or mutating the risk head."""

        if not isinstance(state, Mapping):
            raise TypeError("CARE Stage-A state must be a mapping")
        if any(not isinstance(key, str) for key in state):
            raise TypeError("CARE Stage-A state keys must be strings")

        expected_keys = self.care_stage_a_state_keys()
        expected_set = set(expected_keys)
        actual_set = set(state)
        missing = sorted(expected_set - actual_set)
        unexpected = sorted(actual_set - expected_set)
        if missing or unexpected:
            raise ValueError(
                "CARE Stage-A state keys disagree with the current model: "
                f"missing={missing}, unexpected={unexpected}"
            )

        current_state = self.state_dict()
        invalid_types: list[str] = []
        shape_mismatches: dict[
            str, dict[str, tuple[int, ...]]
        ] = {}
        for key in expected_keys:
            value = state[key]
            if not isinstance(value, torch.Tensor):
                invalid_types.append(
                    f"{key}:{type(value).__name__}"
                )
                continue
            expected_shape = tuple(current_state[key].shape)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                shape_mismatches[key] = {
                    "expected": expected_shape,
                    "actual": actual_shape,
                }
        if invalid_types:
            raise TypeError(
                "CARE Stage-A state values must be tensors: "
                + ", ".join(invalid_types)
            )
        if shape_mismatches:
            raise ValueError(
                "CARE Stage-A tensor shapes disagree with the current model: "
                f"{shape_mismatches}"
            )

        # Validation above is global, so no live module is touched before the
        # complete key/type/shape contract is known to be valid.
        for module_name, module in self._care_stage_a_modules():
            prefix = f"{module_name}."
            local_state = OrderedDict(
                (key[len(prefix) :], state[key])
                for key in expected_keys
                if key.startswith(prefix)
            )
            module.load_state_dict(local_state, strict=True)
        self._care_risk_active = False

    def forward(self, graph_data):
        encoded = self.encode_modalities(graph_data)
        device = encoded["device"]
        dtype = encoded["dtype"]
        batch_size = int(encoded["batch_size"])
        api_emb = encoded["api_emb"]
        graph_emb = encoded["graph_emb"]
        manifest_emb = encoded["manifest_emb"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (api_emb, graph_emb, manifest_emb)
        ):
            raise RuntimeError(
                "CARE encoder backbone returned invalid modality embeddings"
            )

        placeholder_logits = api_emb.new_zeros((batch_size, 2))
        availability, diagnostics = (
            build_fusion_availability_and_diagnostics(
                graph_data,
                placeholder_logits,
                placeholder_logits,
                placeholder_logits,
                api_emb,
                graph_emb,
                manifest_emb,
                materialize_diagnostics=not self.training,
            )
        )
        modality_alive = availability.bool()
        path_logits, path_available = self.care_path_heads(
            {
                "api": api_emb,
                "graph": graph_emb,
                "manifest": manifest_emb,
            },
            modality_alive,
        )
        path_log_odds = torch.stack(
            [
                binary_log_odds(path_logits[name])
                for name in PATH_NAMES
            ],
            dim=-1,
        )
        extra: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        extra.update(diagnostics)
        extra.update(
            {
                "fusion_availability": availability.detach(),
                "care_path_logits": path_logits,
                "care_path_available": path_available.detach(),
                "care_modality_alive": modality_alive.detach(),
                "care_path_log_odds": path_log_odds.detach(),
            }
        )
        selective_eligible = modality_alive.sum(dim=-1).ge(2)

        if self.care_risk_active:
            normalized_log_odds = self.care_risk_head.normalize(
                path_log_odds,
                path_available,
            )
            correctness = self.care_risk_head.score_all(
                normalized_log_odds,
                modality_alive,
            )
            correctness = torch.where(
                path_available,
                correctness,
                torch.zeros_like(correctness),
            )
            routing = route_with_agm_anchor(
                path_logits,
                path_available,
                correctness,
            )
            logits = routing.selected_logits
            selected_path_index = routing.selected_path_index
            extra.update(
                {
                    "care_path_correctness": correctness,
                    "care_selected_path_index": (
                        routing.selected_path_index
                    ),
                    "care_selected_score": routing.selected_score,
                    "care_reject": routing.reject,
                    "care_disagreement_with_agm": (
                        routing.disagreement_with_agm
                    ),
                }
            )
        else:
            # Stage A is selected only by clean AGM performance. Pair heads
            # still shape the shared encoders through their own clean losses.
            logits = path_logits["agm"]
            selected_path_index = torch.where(
                path_available[:, PATH_INDEX["agm"]],
                torch.full(
                    (batch_size,),
                    PATH_INDEX["agm"],
                    device=device,
                    dtype=torch.long,
                ),
                torch.full(
                    (batch_size,),
                    -1,
                    device=device,
                    dtype=torch.long,
                ),
            )

        gate_weights = logits.new_zeros(
            (batch_size, len(MODALITY_NAMES))
        )
        for path_index, path_name in enumerate(PATH_NAMES):
            selected_mask = selected_path_index.eq(path_index).to(
                dtype=dtype
            )
            contribution = 1.0 / float(
                len(PATH_MODALITIES[path_name])
            )
            for modality_name in PATH_MODALITIES[path_name]:
                modality_index = MODALITY_NAMES.index(modality_name)
                gate_weights[:, modality_index] = (
                    gate_weights[:, modality_index]
                    + selected_mask * contribution
                )

        # Fewer than two live modalities is the one structural endpoint. Zero
        # logits represent an explicit uniform predictive distribution rather
        # than classifier-bias evidence from an unavailable path.
        logits = torch.where(
            selective_eligible.unsqueeze(-1),
            logits,
            torch.zeros_like(logits),
        )
        extra["selective_eligible"] = selective_eligible
        extra["gate_weights"] = gate_weights.detach()
        return logits, extra


__all__ = ["CareDroidModel"]
