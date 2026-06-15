from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv
from torch_geometric.utils import softmax


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def safe_mean_pool(node_emb: torch.Tensor, batch: Optional[torch.Tensor],
                   num_graphs: int) -> torch.Tensor:
    if node_emb.numel() == 0:
        return node_emb.new_zeros((num_graphs, node_emb.size(-1)))
    if batch is None:
        pooled = node_emb.mean(dim=0, keepdim=True)
        return pooled if num_graphs == 1 else pooled.expand(num_graphs, -1).contiguous()
    out = node_emb.new_zeros((num_graphs, node_emb.size(-1)))
    cnt = node_emb.new_zeros((num_graphs, 1))
    out.index_add_(0, batch, node_emb)
    cnt.index_add_(0, batch, torch.ones((node_emb.size(0), 1),
                                        device=node_emb.device, dtype=node_emb.dtype))
    return out / cnt.clamp_min(1.0)


def _recover_truncated_sensitive_mask(data, keep_local_parts) -> Optional[torch.Tensor]:
    sensitive_mask = getattr(data, "sensitive_mask", None)
    if sensitive_mask is None or not isinstance(sensitive_mask, torch.Tensor):
        return None
    if sensitive_mask.numel() != data.x.size(0):
        return None

    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
    if batch.numel() == 0:
        return sensitive_mask.new_empty((0,), dtype=torch.bool)

    num_graphs = int(batch.max().item()) + 1
    graph_sizes = torch.bincount(batch, minlength=num_graphs)
    graph_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=data.x.device)
    graph_offsets[1:] = graph_sizes.cumsum(dim=0)

    parts = []
    for gid, local_idx in enumerate(keep_local_parts):
        if local_idx is not None and local_idx.numel() > 0:
            parts.append(graph_offsets[gid] + local_idx.to(data.x.device))
    if not parts:
        return sensitive_mask.new_empty((0,), dtype=torch.bool)

    keep_global = torch.cat(parts, dim=0).long()
    return sensitive_mask.to(data.x.device)[keep_global].bool()


def masked_mean_pool(
    node_emb: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    node_mask: Optional[torch.Tensor],
    fallback: torch.Tensor,
) -> torch.Tensor:
    if node_emb.numel() == 0 or node_mask is None or node_mask.numel() != node_emb.size(0):
        return fallback

    mask = node_mask.to(device=node_emb.device, dtype=node_emb.dtype).view(-1, 1)
    out = node_emb.new_zeros((num_graphs, node_emb.size(-1)))
    cnt = node_emb.new_zeros((num_graphs, 1))
    out.index_add_(0, batch, node_emb * mask)
    cnt.index_add_(0, batch, mask)
    pooled = out / cnt.clamp_min(1.0)
    return torch.where(cnt > 0, pooled, fallback)


class SensitiveAwareReadout(nn.Module):
    """Graph readout that keeps global context while emphasizing risky nodes."""

    def __init__(self, dim: int, dropout: float = 0.1, use_sensitive_hint: bool = True):
        super().__init__()
        self.use_sensitive_hint = bool(use_sensitive_hint)
        hidden = max(dim // 2, 32)
        self.attn = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.mix = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        node_emb: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
        sensitive_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean_all = safe_mean_pool(node_emb, batch, num_graphs)

        if self.use_sensitive_hint and sensitive_mask is not None and sensitive_mask.numel() == node_emb.size(0):
            sensitive_bool = sensitive_mask.to(node_emb.device).bool().view(-1)
            sens = sensitive_bool.to(node_emb.dtype).view(-1, 1)
        else:
            sens = node_emb.new_zeros((node_emb.size(0), 1))
            sensitive_bool = None

        mean_sensitive = masked_mean_pool(
            node_emb,
            batch,
            num_graphs,
            sensitive_bool,
            fallback=mean_all,
        )

        scores = self.attn(torch.cat([node_emb, sens], dim=-1)).view(-1)
        if self.use_sensitive_hint:
            scores = scores + 0.5 * sens.view(-1)
        weights = softmax(scores, batch, num_nodes=num_graphs).view(-1, 1)
        attn_pool = node_emb.new_zeros((num_graphs, node_emb.size(-1)))
        attn_pool.index_add_(0, batch, node_emb * weights)

        return self.mix(torch.cat([mean_all, mean_sensitive, attn_pool], dim=-1))


def _filter_edges(edge_index: torch.Tensor, keep_global: torch.Tensor,
                  num_nodes_old: int) -> torch.Tensor:
    if keep_global.numel() == 0 or edge_index.numel() == 0:
        return edge_index.new_empty((2, 0))
    device = edge_index.device
    mapping = torch.full((num_nodes_old,), -1, dtype=torch.long, device=device)
    mapping[keep_global] = torch.arange(keep_global.numel(), device=device)
    src, dst = mapping[edge_index[0]], mapping[edge_index[1]]
    valid = (src >= 0) & (dst >= 0)
    return torch.stack([src[valid], dst[valid]], dim=0)


def truncate_per_graph(data, max_nodes: int, use_behavior_hint: bool = False):
    """Sensitive-node-priority truncation (vectorised)."""
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

    num_nodes_total = data.x.size(0)
    if num_nodes_total == 0:
        return data.x, data.edge_index, batch, []

    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    device = data.x.device
    graph_sizes = torch.bincount(batch, minlength=num_graphs)

    if graph_sizes.max().item() <= max_nodes:
        keep_local_parts = [torch.arange(graph_sizes[g].item(), device=device)
                            for g in range(num_graphs)]
        return data.x, data.edge_index, batch, keep_local_parts

    sensitive_mask = getattr(data, "sensitive_mask", None)
    if use_behavior_hint and sensitive_mask is not None and sensitive_mask.numel() == num_nodes_total:
        sensitive_nodes = sensitive_mask.bool()
    else:
        sensitive_nodes = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)

    api_aligned_nodes = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    method_api_edge_index = getattr(data, "method_api_edge_index", None)
    if use_behavior_hint and (
        isinstance(method_api_edge_index, torch.Tensor)
        and method_api_edge_index.ndim == 2
        and method_api_edge_index.size(0) == 2
        and method_api_edge_index.numel() > 0
    ):
        src = method_api_edge_index[0].long()
        valid_edge = (src >= 0) & (src < num_nodes_total)

        dst = method_api_edge_index[1].long()
        api_sensitive = getattr(data, "api_sensitive_mask", None)
        api_types = getattr(data, "api_type_ids", None)
        if (
            isinstance(api_sensitive, torch.Tensor)
            and isinstance(api_types, torch.Tensor)
            and api_sensitive.numel() == api_types.numel()
            and api_sensitive.numel() > 0
        ):
            valid_dst = (dst >= 0) & (dst < api_sensitive.numel())
            relevance = torch.zeros_like(valid_edge)
            if valid_dst.any():
                relevant_api = (
                    (api_sensitive.to(device=device).float() > 0.5)
                    & (api_types.to(device=device).long() > 0)
                )
                relevance[valid_dst] = relevant_api[dst[valid_dst].to(device=device)]
            valid_edge = valid_edge & relevance

        valid_src = src[valid_edge]
        if valid_src.numel() > 0:
            api_aligned_nodes[valid_src.unique()] = True

    # Alignment-aware truncation: keep method nodes with explicit API events.
    # Otherwise method-API supervision is often truncated before cross-attention.
    priority = torch.full((num_nodes_total,), 2, dtype=torch.long, device=device)
    priority[sensitive_nodes | api_aligned_nodes] = 1
    priority[sensitive_nodes & api_aligned_nodes] = 0

    graph_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    graph_offsets[1:] = graph_sizes.cumsum(dim=0)
    local_idx = torch.arange(num_nodes_total, device=device) - graph_offsets[batch]

    max_local = int(local_idx.max().item()) + 1
    sort_key = batch.long() * 3 * max_local + priority * max_local + local_idx
    sorted_indices = sort_key.argsort()

    sorted_batch = batch[sorted_indices]
    sorted_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    sorted_sizes = torch.bincount(sorted_batch, minlength=num_graphs)
    sorted_offsets[1:] = sorted_sizes.cumsum(dim=0)
    within_graph_rank = torch.arange(num_nodes_total, device=device) - sorted_offsets[sorted_batch]
    keep_in_sorted = within_graph_rank < max_nodes
    keep_mask = torch.zeros(num_nodes_total, dtype=torch.bool, device=device)
    keep_mask[sorted_indices[keep_in_sorted]] = True
    keep_global = torch.where(keep_mask)[0]
    x = data.x[keep_global]
    edge_index = _filter_edges(data.edge_index, keep_global, num_nodes_total)
    new_batch = batch[keep_global]
    keep_local_parts = []
    for gid in range(num_graphs):
        gid_mask = keep_mask & (batch == gid)
        gid_global = torch.where(gid_mask)[0]
        gid_local = gid_global - graph_offsets[gid]
        keep_local_parts.append(gid_local)
    return x, edge_index, new_batch, keep_local_parts
# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────

def _empty_graph_forward(data, batch, keep_local_parts, out_dim: int):
    """Shared empty-graph early-return for GAT / GCN encoders."""
    num_graphs = int(getattr(data, "num_graphs", 1))
    return (
        data.x.new_zeros((0, out_dim)),
        data.x.new_zeros((num_graphs, out_dim)),
        batch,
        keep_local_parts,
    )


def _num_graphs_from_batch(batch, data) -> int:
    """Return the number of graphs from a PyG batch tensor."""
    if batch is not None and batch.numel() > 0:
        return int(batch.max().item()) + 1
    return int(getattr(data, "num_graphs", 1))
class GraphEncoderGAT(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 heads: int = 4, num_layers: int = 2, max_nodes: int = 2048,
                 use_behavior_hint: bool = False):
        super().__init__()
        if hidden % heads != 0 or out_dim % heads != 0:
            raise ValueError(
                f"hidden ({hidden}) and out_dim ({out_dim}) must be divisible by heads ({heads})")
        num_layers = max(int(num_layers), 1)
        self.max_nodes = max_nodes
        self.use_behavior_hint = bool(use_behavior_hint)
        self.out_dim = int(out_dim)
        self.in_proj = nn.Linear(in_dim, hidden)
        layers = []
        norms = []
        if num_layers == 1:
            layers.append(GATv2Conv(hidden, out_dim // heads, heads=heads, dropout=0.1))
            norms.append(nn.LayerNorm(out_dim))
        else:
            for _ in range(num_layers - 1):
                layers.append(GATv2Conv(hidden, hidden // heads, heads=heads, dropout=0.1))
                norms.append(nn.LayerNorm(hidden))
            layers.append(GATv2Conv(hidden, out_dim // heads, heads=heads, dropout=0.1))
            norms.append(nn.LayerNorm(out_dim))
        self.gat_layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norms)
        self.readout = SensitiveAwareReadout(out_dim, use_sensitive_hint=self.use_behavior_hint)

    def forward(self, data):
        x, edge_index, batch, keep_local_parts = truncate_per_graph(data, self.max_nodes, self.use_behavior_hint)
        if x.numel() == 0:
            return _empty_graph_forward(data, batch, keep_local_parts, self.out_dim)
        h = self.in_proj(x)
        for conv, norm in zip(self.gat_layers, self.norm_layers):
            h = norm(F.elu(conv(h, edge_index)))
        num_graphs = _num_graphs_from_batch(batch, data)
        sensitive_mask = _recover_truncated_sensitive_mask(data, keep_local_parts)
        graph_emb = self.readout(h, batch, num_graphs, sensitive_mask)
        return h, graph_emb, batch, keep_local_parts
class GraphEncoderGCN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128,
                 max_nodes: int = 2048, use_behavior_hint: bool = False):
        super().__init__()
        self.max_nodes = max_nodes
        self.use_behavior_hint = bool(use_behavior_hint)
        self.in_proj = nn.Linear(in_dim, hidden)
        self.gcn1 = GCNConv(hidden, hidden)
        self.gcn2 = GCNConv(hidden, out_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(out_dim)
        self.readout = SensitiveAwareReadout(out_dim, use_sensitive_hint=self.use_behavior_hint)
        self._out_dim = out_dim
    def forward(self, data):
        x, edge_index, batch, keep_local_parts = truncate_per_graph(data, self.max_nodes, self.use_behavior_hint)
        if x.numel() == 0:
            return _empty_graph_forward(data, batch, keep_local_parts, self._out_dim)
        h = self.in_proj(x)
        h = self.norm1(F.elu(self.gcn1(h, edge_index)))
        h = self.norm2(F.elu(self.gcn2(h, edge_index)))
        num_graphs = _num_graphs_from_batch(batch, data)
        sensitive_mask = _recover_truncated_sensitive_mask(data, keep_local_parts)
        graph_emb = self.readout(h, batch, num_graphs, sensitive_mask)
        return h, graph_emb, batch, keep_local_parts
