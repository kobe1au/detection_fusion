from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from torch_geometric.utils import softmax

from fusion.graph_encoders import GraphEncoderGAT, GraphEncoderGCN
from fusion.semantic_categories import validate_api_type_mapping
from fusion.utils import strict_finite_integer


class ApiSequenceEncoder(nn.Module):
    """Shared API-event encoder used by CARE-Droid and comparison baselines."""

    def __init__(
        self,
        num_hash_buckets: int,
        type_vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
        encoder_type: str = "transformer",
        num_layers: int = 2,
        num_heads: int = 4,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.num_hash_buckets = int(num_hash_buckets)
        self.type_vocab_size = int(type_vocab_size)
        self.emb_dim = int(emb_dim)
        self.encoder_type = str(encoder_type).lower()
        self.max_seq_len = strict_finite_integer(
            max_seq_len,
            field_name="ApiSequenceEncoder.max_seq_len",
        )
        if self.max_seq_len <= 0:
            raise ValueError("ApiSequenceEncoder.max_seq_len must be positive")
        self.max_valid_api_id = self.num_hash_buckets + 1
        self.overflow_id = self.num_hash_buckets + 2
        self.api_embedding = nn.Embedding(
            self.overflow_id + 1,
            emb_dim,
            padding_idx=0,
        )
        # Extracted API hash ids occupy 2..N+1. Keep N+1 as a valid bucket
        # and reserve N+2 exclusively for out-of-range values.
        nn.init.zeros_(self.api_embedding.weight[self.overflow_id])
        self.type_embedding = nn.Embedding(self.type_vocab_size, emb_dim)
        self.sensitive_embedding = nn.Embedding(2, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_dropout = nn.Dropout(dropout)

        if self.encoder_type == "bigru":
            if emb_dim % 2 != 0:
                raise ValueError("emb_dim must be even for bigru API encoder")
            self.sequence_encoder = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim // 2,
                num_layers=max(1, int(num_layers)),
                batch_first=True,
                bidirectional=True,
                dropout=dropout if int(num_layers) > 1 else 0.0,
            )
            self.pos_embedding = None
        elif self.encoder_type == "transformer":
            if emb_dim % int(num_heads) != 0:
                raise ValueError("api emb_dim must be divisible by api heads")
            self.pos_embedding = nn.Embedding(self.max_seq_len, emb_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=int(num_heads),
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                layer,
                num_layers=max(1, int(num_layers)),
            )
        else:
            raise ValueError(f"Unsupported API encoder type: {encoder_type}")

        self.out_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(emb_dim, max(hidden_dim // 2, 32)),
            nn.Tanh(),
            nn.Linear(max(hidden_dim // 2, 32), 1),
        )

    def _empty_output(self, num_graphs: int, device, dtype):
        return (
            torch.zeros((0, self.emb_dim), device=device, dtype=dtype),
            torch.zeros(
                (num_graphs, self.emb_dim),
                device=device,
                dtype=dtype,
            ),
            torch.empty((0,), device=device, dtype=torch.long),
        )

    @staticmethod
    def _padded_batch(
        event_emb: torch.Tensor,
        api_batch: torch.Tensor,
        num_graphs: int,
    ):
        device = event_emb.device
        lengths = torch.bincount(
            api_batch,
            minlength=num_graphs,
        ).to(device=device)
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        if max_len <= 0:
            return None, None, lengths, None
        padded = event_emb.new_zeros(
            (num_graphs, max_len, event_emb.size(-1))
        )
        key_padding_mask = torch.ones(
            (num_graphs, max_len),
            device=device,
            dtype=torch.bool,
        )
        offsets = torch.zeros(
            (num_graphs + 1,),
            device=device,
            dtype=torch.long,
        )
        offsets[1:] = lengths.cumsum(dim=0)
        restore_pos = (
            torch.arange(event_emb.size(0), device=device)
            - offsets[api_batch]
        )
        padded[api_batch, restore_pos] = event_emb
        key_padding_mask[api_batch, restore_pos] = False
        empty_rows = lengths == 0
        if empty_rows.any():
            key_padding_mask[empty_rows, 0] = False
        return padded, key_padding_mask, lengths, restore_pos

    def _normalize_api_ids(
        self,
        api_ids: torch.Tensor,
        device,
    ) -> torch.Tensor:
        api_ids = api_ids.to(device=device, dtype=torch.long)
        overflow = torch.full_like(api_ids, self.overflow_id)
        api_ids = torch.where(
            api_ids > self.max_valid_api_id,
            overflow,
            api_ids,
        )
        return api_ids.clamp_min(0)

    def forward(self, graph_data, num_graphs: int, device, dtype):
        api_ids = getattr(graph_data, "api_ids", None)
        api_batch = getattr(graph_data, "api_batch", None)
        if (
            api_ids is None
            or api_batch is None
            or api_ids.numel() == 0
        ):
            return self._empty_output(num_graphs, device, dtype)

        api_ids = self._normalize_api_ids(api_ids, device)
        api_batch = api_batch.to(
            device=device,
            dtype=torch.long,
        ).view(-1)
        if api_batch.numel() != api_ids.numel():
            raise ValueError(
                f"api_batch length {api_batch.numel()} does not match "
                f"api_ids length {api_ids.numel()}"
            )
        if (
            num_graphs <= 0
            or (api_batch < 0).any()
            or (api_batch >= num_graphs).any()
        ):
            raise ValueError(
                "api_batch contains indices outside "
                f"[0, {max(num_graphs - 1, 0)}]"
            )
        if (
            api_batch.numel() > 1
            and (api_batch[1:] < api_batch[:-1]).any()
        ):
            raise ValueError(
                "api_batch must be grouped in non-decreasing sample order"
            )
        raw_lengths = torch.bincount(
            api_batch,
            minlength=num_graphs,
        ).to(device=device)
        observed_max = (
            int(raw_lengths.max().item())
            if raw_lengths.numel() > 0
            else 0
        )
        if observed_max > self.max_seq_len:
            raise RuntimeError(
                "API dataset/encoder budget contract was violated: the "
                "encoder must never truncate silently because reliability "
                "evidence and semantic counts were already computed from the "
                "dataset output; observed per-sample length "
                f"{observed_max} exceeds max_seq_len={self.max_seq_len}"
            )

        raw_type_ids = getattr(graph_data, "api_type_ids", None)
        api_type_ids = (
            torch.zeros_like(api_ids)
            if raw_type_ids is None
            else raw_type_ids.to(
                device=device,
                dtype=torch.long,
            ).clamp(0, self.type_vocab_size - 1)
        )
        raw_sensitive = getattr(graph_data, "api_sensitive_mask", None)
        api_sensitive = (
            torch.zeros_like(api_ids)
            if raw_sensitive is None
            else raw_sensitive.to(device=device)
            .float()
            .gt(0.5)
            .long()
            .clamp(0, 1)
        )
        event_emb = (
            self.api_embedding(api_ids)
            + self.type_embedding(api_type_ids)
            + self.sensitive_embedding(api_sensitive)
        )
        event_emb = self.input_dropout(self.input_norm(event_emb))
        padded, key_padding_mask, _lengths, restore_pos = (
            self._padded_batch(event_emb, api_batch, num_graphs)
        )
        if padded is None:
            return self._empty_output(num_graphs, device, dtype)

        if self.encoder_type == "transformer":
            if self.pos_embedding is None:
                raise RuntimeError(
                    "Transformer API encoder omitted positional embeddings"
                )
            pos = torch.arange(
                padded.size(1),
                device=device,
            ).clamp(max=self.max_seq_len - 1)
            encoded = self.sequence_encoder(
                padded + self.pos_embedding(pos).unsqueeze(0),
                src_key_padding_mask=key_padding_mask,
            )
        else:
            encoded, _ = self.sequence_encoder(padded)
        token_emb = encoded[api_batch, restore_pos]
        token_emb = self.out_proj(token_emb)
        scores = self.pool_score(token_emb).view(-1)
        scores = scores.masked_fill(~torch.isfinite(scores), 0.0)
        weights = softmax(
            scores,
            api_batch,
            num_nodes=num_graphs,
        ).view(-1, 1)
        pooled = token_emb.new_zeros((num_graphs, self.emb_dim))
        pooled.index_add_(0, api_batch, token_emb * weights)
        return token_emb, pooled.to(dtype=dtype), api_batch


class ManifestEncoder(nn.Module):
    """Shared dense Manifest encoder."""

    def __init__(
        self,
        in_dim: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.emb_dim = int(emb_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class TriModalEncoderBackbone(nn.Module):
    """Fusion-agnostic API, Graph, and Manifest encoder backbone."""

    def __init__(
        self,
        *,
        in_feat_dim: int = 515,
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
        super().__init__()
        validate_api_type_mapping()
        self.api_emb_dim = int(api_emb_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.manifest_in_dim = int(manifest_in_dim)
        self.manifest_emb_dim = int(manifest_emb_dim)

        self.api_encoder = ApiSequenceEncoder(
            num_hash_buckets=api_num_hash_buckets,
            type_vocab_size=api_type_vocab_size,
            emb_dim=api_emb_dim,
            hidden_dim=api_hidden_dim,
            dropout=api_dropout,
            encoder_type=api_encoder_type,
            num_layers=api_layers,
            num_heads=api_heads,
            max_seq_len=api_max_seq_len,
        )

        graph_encoder_type = str(
            graph_encoder_type or "gatv2"
        ).lower()
        if graph_encoder_type in {"gat", "gatv2"}:
            self.graph_encoder = GraphEncoderGAT(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                heads=graph_heads,
                num_layers=graph_layers,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        elif graph_encoder_type == "gcn":
            self.graph_encoder = GraphEncoderGCN(
                in_dim=in_feat_dim,
                out_dim=graph_emb_dim,
                hidden=graph_hidden,
                max_nodes=max_nodes_gnn,
                use_behavior_hint=use_graph_behavior_hint,
            )
        else:
            raise ValueError(
                f"Unsupported graph_encoder_type: {graph_encoder_type}"
            )

        self.manifest_encoder = ManifestEncoder(
            in_dim=manifest_in_dim,
            emb_dim=manifest_emb_dim,
            hidden_dim=manifest_hidden_dim,
            dropout=manifest_dropout,
        )

    def _manifest_input(
        self,
        graph_data,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        value = getattr(graph_data, "manifest_x", None)
        if not isinstance(value, torch.Tensor):
            return torch.zeros(
                (batch_size, self.manifest_in_dim),
                device=device,
                dtype=dtype,
            )
        value = value.to(
            device=device,
            dtype=dtype,
        ).view(batch_size, -1)
        if value.size(1) < self.manifest_in_dim:
            value = torch.cat(
                [
                    value,
                    value.new_zeros(
                        (
                            batch_size,
                            self.manifest_in_dim - value.size(1),
                        )
                    ),
                ],
                dim=-1,
            )
        elif value.size(1) > self.manifest_in_dim:
            warnings.warn(
                f"manifest_x dim {value.size(1)} > configured "
                f"{self.manifest_in_dim}; truncating trailing features. "
                "Check dataset or model config.",
                stacklevel=2,
            )
            value = value[:, : self.manifest_in_dim]
        return torch.nan_to_num(
            value,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def _encode_api(
        self,
        graph_data,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        """Encode the currently visible API events."""

        _, pooled, _ = self.api_encoder(
            graph_data,
            batch_size,
            device,
            dtype,
        )
        return pooled

    @staticmethod
    def _availability_mask(
        graph_data,
        name: str,
        batch_size: int,
        device,
        dtype,
    ) -> torch.Tensor:
        value = getattr(graph_data, name, None)
        if not isinstance(value, torch.Tensor):
            raise ValueError(
                "Current dataset batch is missing mandatory availability "
                f"mask {name!r}"
            )
        value = value.to(
            device=device,
            dtype=dtype,
        ).view(batch_size, -1)
        if value.size(1) == 0:
            return torch.zeros(
                (batch_size, 1),
                device=device,
                dtype=dtype,
            )
        return value[:, :1].gt(0.0).to(dtype=dtype)

    def encode_modalities(
        self,
        graph_data,
    ) -> dict[str, torch.Tensor | int | torch.device | torch.dtype]:
        """Return the common encoded state without applying fusion semantics."""

        reference = next(self.parameters())
        device = reference.device
        dtype = reference.dtype
        batch_size = int(getattr(graph_data, "num_graphs", 1))
        api_emb = self._encode_api(
            graph_data,
            batch_size,
            device,
            dtype,
        )
        _, graph_emb, _, _ = self.graph_encoder(graph_data)
        manifest_x = self._manifest_input(
            graph_data,
            batch_size,
            device,
            dtype,
        )
        manifest_emb = self.manifest_encoder(manifest_x)
        api_available = self._availability_mask(
            graph_data,
            "api_alive",
            batch_size,
            device,
            dtype,
        )
        graph_available = self._availability_mask(
            graph_data,
            "graph_alive",
            batch_size,
            device,
            dtype,
        )
        manifest_available = self._availability_mask(
            graph_data,
            "manifest_alive",
            batch_size,
            device,
            dtype,
        )
        modality_alive = torch.cat(
            [api_available, graph_available, manifest_available],
            dim=-1,
        ).bool()
        return {
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "api_emb": api_emb,
            "graph_emb": graph_emb,
            "manifest_emb": manifest_emb,
            "api_available": api_available,
            "graph_available": graph_available,
            "manifest_available": manifest_available,
            "modality_alive": modality_alive,
        }


__all__ = [
    "ApiSequenceEncoder",
    "ManifestEncoder",
    "TriModalEncoderBackbone",
]
