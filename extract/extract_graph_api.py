#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build graph + API-sequence features from APK files.

Each output file is:
  {out_root}/{split}/{sha256}.pt

The .pt content is a top-level payload with sample-level observable metadata
and a ``dex_list`` containing one dict per successfully parsed DEX:
  {
    "dex_name": str,

    # Graph branch
    "call_x": Float16Tensor [M, 515 + optional graph-lite behavior hints],
    "call_edge_index": Int32Tensor [2, E],
    "call_sensitive_mask": UInt8Tensor [M],
    "method_spans": Int32Tensor [M, 2],

    # API semantic branch
    "api_ids": LongTensor [T],
    "api_type_ids": UInt8Tensor [T],
    "api_sensitive_mask": UInt8Tensor [T],
    "api_method_index": Int32Tensor [T],
    "api_in_graph_mask": UInt8Tensor [T],
    "method_api_edge_index": Int32Tensor [2, K],
    "api_category_counts": Float16Tensor [C],
    "api_semantic_category_counts": Float16Tensor [12],
    "graph_semantic_category_counts": Float16Tensor [12],

    "meta": {...}
  }

Design:
  API branch: ordered framework API behavior sequence.
  Graph branch: sensitive method call graph with lightweight method behavior hints.
  Alignment: method_api_edge_index anchors API events to graph method nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from androguard.core.analysis.analysis import Analysis
except Exception:  # pragma: no cover - androguard version compatibility
    Analysis = None
from androguard.core.dex import DEX

from fusion.semantic_categories import (
    SEMANTIC_CATEGORIES,
    api_semantic_counts_from_type_ids,
    graph_semantic_counts_from_method_api_edges,
)
from fusion.quality import OBSERVABLE_SCHEMA_VERSION, graph_structure_stats

warnings.filterwarnings("ignore")
logging.getLogger("androguard").setLevel(logging.CRITICAL)
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.disable("androguard")
except Exception:
    pass


FRAMEWORK_PREFIXES = (
    # DEX style
    "Landroid/",
    "Ljava/",
    "Ljavax/",
    "Ldalvik/",
    "Lorg/apache/http/",
    "Lokhttp3/",
    "Lcom/google/android/",

    # Dot style fallback
    "android.",
    "java.",
    "javax.",
    "dalvik.",
    "org.apache.http.",
    "okhttp3.",
    "com.google.android.",
)

DEFAULT_DROP_PREFIXES = (
    "Landroidx/",
    "Landroid/support/",
    "androidx.",
    "android.support.",
)

API_CATEGORY_NAMES = [
    "other",
    "telephony",
    "sms",
    "location",
    "contacts_content",
    "camera_media",
    "network",
    "runtime_exec",
    "reflection",
    "dynamic_loading",
    "file_io",
    "package_info",
    "crypto",
    "webview",
    "system_settings",
    "account",
]

API_CATEGORY_TO_ID = {name: i for i, name in enumerate(API_CATEGORY_NAMES)}


def safe_mkdir(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(obj: Any, path: Path) -> None:
    path = Path(path)
    safe_mkdir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def list_dex_entries(apk_path: Path) -> List[str]:
    with zipfile.ZipFile(apk_path, "r") as zf:
        entries = [
            name for name in zf.namelist()
            if re.fullmatch(r"classes(?:\d*)\.dex", os.path.basename(name))
        ]
    return sorted(entries, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))


def method_sig(m: Any) -> str:
    try:
        return canonical_method_sig(m.get_class_name(), m.get_name(), m.get_descriptor())
    except Exception:
        return str(m)


def _build_analysis(dvm: DEX, raw_bytes: bytes):
    if Analysis is None:
        return None
    try:
        dx = Analysis(dvm)
    except TypeError:
        dx = Analysis()
        try:
            dx.add(dvm)
        except Exception:
            pass
    try:
        dx.create_xref()
    except Exception:
        pass
    return dx


def _iter_method_instructions(m: Any):
    try:
        return list(m.get_instructions())
    except Exception:
        pass
    try:
        code = m.get_code()
        if code is None:
            return []
        return list(code.get_bc().get_instructions())
    except Exception:
        return []


def _iter_basic_blocks(ma: Any):
    try:
        blocks = ma.get_basic_blocks()
        if hasattr(blocks, "get"):
            return list(blocks.get())
        return list(blocks)
    except Exception:
        return []


def parse_invoke_targets_from_method(m: Any) -> List[Tuple[str, str, str]]:
    targets: List[Tuple[str, str, str]] = []
    pat = re.compile(r"(\[*L[^;]+;)->([^\s(]+)(\([^)]*\)\S*)")

    for ins in _iter_method_instructions(m):
        try:
            name = str(ins.get_name()).lower()
        except Exception:
            name = ""
        if not name.startswith("invoke"):
            continue

        try:
            output = str(ins.get_output())
        except Exception:
            output = str(ins)

        match = pat.search(output)
        if match:
            targets.append((match.group(1), match.group(2), match.group(3)))

    return targets


def is_sensitive_callee(class_name: str, method_name: str) -> bool:
    full = f"{normalize_dex_class_name(class_name)}#{method_name}".lower()
    sensitive_keywords = (
        "telephony",
        "sms",
        "sendtextmessage",
        "location",
        "contacts",
        "contentresolver",
        "camera",
        "mediarecorder",
        "audio",
        "socket",
        "urlconnection",
        "http",
        "runtime#exec",
        "processbuilder",
        "reflect",
        "class#forname",
        "dexclassloader",
        "pathclassloader",
        "loadclass",
        "getinstalledpackages",
        "getinstalledapplications",
        "packagemanager",
        "javax.crypto",
        "messagedigest",
        "webview",
        "settings",
        "account",
    )
    return any(k in full for k in sensitive_keywords)


def _instruction_opcode(ins: Any) -> int:
    for attr in ("get_op_value", "get_opcode"):
        fn = getattr(ins, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
    try:
        return int(getattr(ins, "op_value"))
    except Exception:
        return 0


def _block_histogram(instructions: Sequence[Any], vocab_size: int) -> np.ndarray:
    hist = np.zeros((vocab_size,), dtype=np.float32)
    for ins in instructions:
        hist[_instruction_opcode(ins) % vocab_size] += 1.0
    total = float(hist.sum())
    if total > 0:
        hist /= total
    return hist


def build_method_local_cfg_embedding(
    ma: Any,
    m: Any,
    vocab_size: int = 256,
) -> Tuple[np.ndarray, Tuple[int, int], bool]:
    blocks = _iter_basic_blocks(ma)
    block_hists: List[np.ndarray] = []
    num_edges = 0
    starts: List[int] = []
    ends: List[int] = []

    if blocks:
        for block in blocks:
            instructions = _iter_method_instructions(block)
            if not instructions:
                try:
                    instructions = list(block.get_instructions())
                except Exception:
                    instructions = []
            block_hists.append(_block_histogram(instructions, vocab_size))

            children = getattr(block, "childs", None)
            if children is None:
                children = getattr(block, "child", [])
            try:
                num_edges += len(children)
            except Exception:
                pass

            for attr, bucket in (("get_start", starts), ("get_end", ends)):
                fn = getattr(block, attr, None)
                if callable(fn):
                    try:
                        bucket.append(int(fn()))
                    except Exception:
                        pass
    else:
        instructions = _iter_method_instructions(m)
        block_hists.append(_block_histogram(instructions, vocab_size))

    if not block_hists:
        block_hists.append(np.zeros((vocab_size,), dtype=np.float32))

    h = np.stack(block_hists, axis=0)
    mean_hist = h.mean(axis=0)
    max_hist = h.max(axis=0)

    num_blocks = len(block_hists)
    num_instr = 0
    for ins in _iter_method_instructions(m):
        num_instr += 1

    stats = np.asarray(
        [
            min(np.log1p(num_blocks) / np.log1p(64), 1.0),
            min(np.log1p(num_edges) / np.log1p(128), 1.0),
            min(np.log1p(num_instr) / np.log1p(2048), 1.0),
        ],
        dtype=np.float32,
    )

    emb = np.concatenate([mean_hist, max_hist, stats], axis=0).astype(np.float32)
    span = (
        int(min(starts)) if starts else 0,
        int(max(ends)) if ends else int(num_instr),
    )
    sensitive = any(is_sensitive_callee(cls, name) for cls, name, _ in parse_invoke_targets_from_method(m))
    return emb, span, sensitive


def build_graph_lite_behavior_hints(
    events: Sequence["ApiEvent"],
    is_sensitive_method: bool,
    count_norm: int = 32,
) -> np.ndarray:
    """Return four lightweight method-level behavior hints for graph nodes.

    The hints intentionally avoid copying fine-grained API identity into the
    graph branch. They only describe behavior intensity and sensitivity:
      1) whether this method calls a sensitive framework API
      2) log-normalized API event count
      3) ratio of sensitive API events
      4) ratio of non-other semantic API categories
    """
    n = len(events)
    if n <= 0:
        return np.asarray([float(is_sensitive_method), 0.0, 0.0, 0.0], dtype=np.float32)

    sensitive_count = sum(1 for e in events if e.sensitive)
    semantic_count = sum(1 for e in events if e.category_id != API_CATEGORY_TO_ID["other"])
    denom = float(max(n, 1))
    count_base = max(int(count_norm), 1)

    return np.asarray(
        [
            float(is_sensitive_method),
            min(np.log1p(n) / np.log1p(count_base), 1.0),
            sensitive_count / denom,
            semantic_count / denom,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ApiEvent:
    old_method_idx: int
    token: str
    category_id: int
    sensitive: bool


def canonical_method_sig(class_name: str, method_name: str, desc: str) -> str:
    """Canonical DEX method key used for both method definitions and invoke targets."""
    cls = str(class_name or "").strip()
    name = str(method_name or "").strip()
    descriptor = str(desc or "").strip()
    return f"{cls}->{name}{descriptor}"


def canonical_method_sig_from_method(m: Any) -> str:
    try:
        return canonical_method_sig(m.get_class_name(), m.get_name(), m.get_descriptor())
    except Exception:
        return method_sig(m)


def normalize_dex_class_name(class_name: str) -> str:
    s = str(class_name or "").strip()
    while s.startswith("["):
        s = s[1:]
    if s.startswith("L") and s.endswith(";"):
        s = s[1:-1]
    return s.replace("/", ".")


def canonical_api_token(
    class_name: str,
    method_name: str,
    desc: str,
    include_descriptor: bool = False,
) -> str:
    cls = normalize_dex_class_name(class_name)
    name = str(method_name or "").strip() or "<init>"
    if include_descriptor:
        return f"{cls}#{name}{desc}"
    return f"{cls}#{name}"


def stable_hash_id(token: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="little", signed=False)
    return 2 + (value % num_buckets)  # reserve 0=pad, 1=unk


def categorize_api(class_name: str, method_name: str) -> int:
    cls = normalize_dex_class_name(class_name)
    name = str(method_name or "")
    full = f"{cls}#{name}".lower()

    if "telephony" in full and "sms" not in full:
        return API_CATEGORY_TO_ID["telephony"]
    if "sms" in full or "sendtextmessage" in full:
        return API_CATEGORY_TO_ID["sms"]
    if "location" in full or "gps" in full:
        return API_CATEGORY_TO_ID["location"]
    if "contacts" in full or "contentresolver" in full or "provider" in full:
        return API_CATEGORY_TO_ID["contacts_content"]
    if "camera" in full or "mediarecorder" in full or "audio" in full:
        return API_CATEGORY_TO_ID["camera_media"]
    if (
        "java.net" in full
        or "javax.net" in full
        or "android.net" in full
        or "okhttp" in full
        or "http" in full
        or "socket" in full
        or "urlconnection" in full
    ):
        return API_CATEGORY_TO_ID["network"]
    if "runtime#exec" in full or "processbuilder" in full:
        return API_CATEGORY_TO_ID["runtime_exec"]
    if "reflect" in full or "class#forname" in full:
        return API_CATEGORY_TO_ID["reflection"]
    if "dexclassloader" in full or "pathclassloader" in full or "loadclass" in full:
        return API_CATEGORY_TO_ID["dynamic_loading"]
    if "java.io" in full or "java.nio" in full or "environment" in full:
        return API_CATEGORY_TO_ID["file_io"]
    if "packagemanager" in full or "getinstalledpackages" in full or "getinstalledapplications" in full:
        return API_CATEGORY_TO_ID["package_info"]
    if "javax.crypto" in full or "java.security" in full or "messagedigest" in full:
        return API_CATEGORY_TO_ID["crypto"]
    if "webview" in full:
        return API_CATEGORY_TO_ID["webview"]
    if "settings" in full or "systemproperties" in full:
        return API_CATEGORY_TO_ID["system_settings"]
    if "account" in full:
        return API_CATEGORY_TO_ID["account"]
    return API_CATEGORY_TO_ID["other"]


def should_keep_api(
    class_name: str,
    framework_only: bool = True,
    include_prefixes: Sequence[str] = FRAMEWORK_PREFIXES,
    drop_prefixes: Sequence[str] = DEFAULT_DROP_PREFIXES,
) -> bool:
    cls = str(class_name or "").strip()
    normalized = normalize_dex_class_name(cls)

    if any(cls.startswith(p) or normalized.startswith(p) for p in drop_prefixes):
        return False
    if not framework_only:
        return True
    return any(cls.startswith(p) or normalized.startswith(p) for p in include_prefixes)


def select_api_events(events: List[ApiEvent], max_events: int) -> List[ApiEvent]:
    if max_events <= 0 or len(events) <= max_events:
        return events

    sensitive_idx = [i for i, e in enumerate(events) if e.sensitive]
    semantic_idx = [
        i for i, e in enumerate(events)
        if e.category_id != API_CATEGORY_TO_ID["other"] and not e.sensitive
    ]

    sensitive_set = set(sensitive_idx)
    semantic_set = set(semantic_idx)
    other_idx = [
        i for i, _ in enumerate(events)
        if i not in sensitive_set and i not in semantic_set
    ]

    chosen: List[int] = []
    for pool in (sensitive_idx, semantic_idx, other_idx):
        if len(chosen) >= max_events:
            break
        remain = max_events - len(chosen)
        if len(pool) <= remain:
            chosen.extend(pool)
        elif remain > 0:
            pos = np.linspace(0, len(pool) - 1, num=remain, dtype=np.int64)
            chosen.extend([pool[int(i)] for i in pos])

    chosen = sorted(set(chosen))[:max_events]
    return [events[i] for i in chosen]


def _rank_methods_for_fallback(
    events_by_method: List[List[ApiEvent]],
    max_methods: int,
) -> Set[int]:
    scores = []
    for i, events in enumerate(events_by_method):
        sens = sum(1 for e in events if e.sensitive)
        semantic = sum(1 for e in events if e.category_id != API_CATEGORY_TO_ID["other"])
        scores.append((sens * 1000 + semantic * 10 + len(events), -i, i))
    scores.sort(reverse=True)
    keep = [i for _, _, i in scores[:max_methods]]
    return set(keep)


def _empty_dex_result(
    dim: int,
    reason: str,
    use_graph_behavior_hints: bool = False,
) -> Dict[str, Any]:
    return {
        "call_x": torch.empty((0, dim), dtype=torch.float16),
        "call_edge_index": torch.empty((2, 0), dtype=torch.int32),
        "call_sensitive_mask": torch.empty((0,), dtype=torch.uint8),
        "method_spans": torch.empty((0, 2), dtype=torch.int32),
        "api_ids": torch.empty((0,), dtype=torch.long),
        "api_type_ids": torch.empty((0,), dtype=torch.uint8),
        "api_sensitive_mask": torch.empty((0,), dtype=torch.uint8),
        "api_method_index": torch.empty((0,), dtype=torch.int32),
        "api_in_graph_mask": torch.empty((0,), dtype=torch.uint8),
        "method_api_edge_index": torch.empty((2, 0), dtype=torch.int32),
        "api_category_counts": torch.zeros((len(API_CATEGORY_NAMES),), dtype=torch.float16),
        "api_semantic_category_counts": torch.zeros((len(SEMANTIC_CATEGORIES),), dtype=torch.float16),
        "graph_semantic_category_counts": torch.zeros((len(SEMANTIC_CATEGORIES),), dtype=torch.float16),
        "observable_metadata": {
            "api_parse_ok": True,
            "api_parse_error": "",
            "api_event_count_raw": 0,
            "api_event_count_kept": 0,
            "api_truncated": False,
            "api_truncation_ratio": 0.0,
            "api_known_type_count": 0,
            "api_unknown_type_count": 0,
            "dex_parse_ok": True,
            "dex_file_count": 1,
            "dex_parse_success_ratio": 1.0,
            "class_count": 0,
            "method_count": 0,
            "graph_parse_ok": True,
            "graph_parse_error": "",
            "graph_timeout": False,
            "graph_node_count_raw": 0,
            "graph_edge_count_raw": 0,
            "graph_isolated_node_count": 0,
            "graph_largest_component_ratio": 0.0,
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        },
        "meta": {
            "empty_reason": reason,
            "graph": {
                "num_methods_all": 0,
                "num_methods_sensitive_seed": 0,
                "num_methods_kept": 0,
                "num_edges_kept": 0,
                "local_method_emb_dim": int(dim),
                "sensitive_hops": 0,
                "fallback_used": False,
                "max_methods_per_dex": 0,
                "invoke_total": 0,
                "invoke_internal_hit": 0,
                "invoke_internal_hit_rate": 0.0,
                "graph_behavior_hints": bool(use_graph_behavior_hints),
                "graph_behavior_hint_names": [
                    "sensitive_method_flag",
                    "log_api_event_count",
                    "sensitive_api_ratio",
                    "semantic_api_ratio",
                ] if use_graph_behavior_hints else [],
            },
            "api": {
                "num_api_events_raw": 0,
                "num_api_events": 0,
                "api_truncated": False,
                "num_api_events_in_graph": 0,
                "api_in_graph_ratio": 0.0,
                "num_unique_api_tokens": 0,
                "num_unique_api_ids": 0,
                "hash_collision_estimate": 0,
                "num_api_buckets": 0,
                "num_api_categories": len(API_CATEGORY_NAMES),
                "api_category_names": API_CATEGORY_NAMES,
                "semantic_category_names": list(SEMANTIC_CATEGORIES),
                "api_category_counts_source": "empty_dex",
                "graph_semantic_category_counts_source": "empty_dex",
                "hash_collision_scope": "empty_dex",
                "api_event_scope": "",
                "framework_only": True,
                "include_descriptor": False,
                "max_api_events_per_dex": 0,
                "max_api_events_per_method": 0,
                "representation": "empty_dex",
                "id_reserved": {"pad": 0, "unk": 1},
                "category_method": "heuristic_v1_prefix_keyword",
            },
            "stored_dtypes": {
                "call_x": "float16",
                "call_edge_index": "int32",
                "call_sensitive_mask": "uint8",
                "method_spans": "int32",
                "api_ids": "int64",
                "api_type_ids": "uint8",
                "api_sensitive_mask": "uint8",
                "api_method_index": "int32",
                "api_in_graph_mask": "uint8",
                "method_api_edge_index": "int32",
                "api_category_counts": "float16",
                "api_semantic_category_counts": "float16",
                "graph_semantic_category_counts": "float16",
            },
        },
    }


def build_graph_api_for_dex(
    dvm: DEX,
    raw_bytes: bytes,
    vocab_size: int = 256,
    keep_method_names: bool = False,
    keep_api_tokens: bool = False,
    sensitive_hops: int = 1,
    max_methods_per_dex: int = 4096,
    fallback_max_methods: int = 512,
    num_api_buckets: int = 8192,
    max_api_events_per_dex: int = 1024,
    max_api_events_per_method: int = 32,
    api_event_scope: str = "all_methods",
    framework_only: bool = True,
    include_descriptor: bool = False,
    fallback_policy: str = "api_rich",
    use_graph_behavior_hints: bool = True,
) -> Dict[str, Any]:
    dx = _build_analysis(dvm, raw_bytes)
    graph_hint_dim = 4 if use_graph_behavior_hints else 0
    base_dim = int(vocab_size) * 2 + 3
    out_dim = base_dim + graph_hint_dim

    method_list: List[Any] = []
    ma_list: List[Any] = []
    method_to_idx: Dict[str, int] = {}

    for ma in dx.get_methods():
        is_ext_fn = getattr(ma, "is_external", None)
        if callable(is_ext_fn):
            try:
                if bool(is_ext_fn()):
                    continue
            except Exception:
                pass

        try:
            m = ma.get_method()
        except Exception:
            m = getattr(ma, "method", None)

        if m is None:
            continue

        sig = canonical_method_sig_from_method(m)
        if sig in method_to_idx:
            continue

        method_to_idx[sig] = len(method_list)
        method_list.append(m)
        ma_list.append(ma)

    m_all = len(method_list)
    dim = out_dim
    if m_all == 0:
        return _empty_dex_result(dim, "no_methods", use_graph_behavior_hints)

    method_embs: List[np.ndarray] = []
    method_spans: List[Tuple[int, int]] = []
    sensitive_seed: Set[int] = set()
    method_names: List[str] = []
    events_by_method: List[List[ApiEvent]] = []

    for idx, (ma, m) in enumerate(zip(ma_list, method_list)):
        emb, span, is_sensitive = build_method_local_cfg_embedding(
            ma=ma,
            m=m,
            vocab_size=vocab_size,
        )
        if is_sensitive:
            sensitive_seed.add(idx)

        cur_events: List[ApiEvent] = []
        for cls, name, desc in parse_invoke_targets_from_method(m):
            if not should_keep_api(cls, framework_only=framework_only):
                continue

            token = canonical_api_token(cls, name, desc, include_descriptor=include_descriptor)
            cat_id = categorize_api(cls, name)
            cur_events.append(
                ApiEvent(
                    old_method_idx=idx,
                    token=token,
                    category_id=cat_id,
                    sensitive=is_sensitive_callee(cls, name),
                )
            )

        if max_api_events_per_method > 0 and len(cur_events) > max_api_events_per_method:
            cur_events = select_api_events(cur_events, max_api_events_per_method)

        if use_graph_behavior_hints:
            hints = build_graph_lite_behavior_hints(
                cur_events,
                is_sensitive_method=is_sensitive,
                count_norm=max_api_events_per_method,
            )
            emb = np.concatenate([emb, hints], axis=0).astype(np.float32)

        method_embs.append(emb)
        method_spans.append(span)
        method_names.append(canonical_method_sig_from_method(m))
        events_by_method.append(cur_events)

    dim = int(method_embs[0].shape[0])

    invoke_total = 0
    invoke_internal_hit = 0

    edges_full: Set[Tuple[int, int]] = set()
    neighbors_undirected: Dict[int, Set[int]] = {i: set() for i in range(m_all)}

    for u, m in enumerate(method_list):
        for cls, name, desc in parse_invoke_targets_from_method(m):
            invoke_total += 1
            v = method_to_idx.get(canonical_method_sig(cls, name, desc))
            if v is not None:
                invoke_internal_hit += 1
                edges_full.add((u, v))
                neighbors_undirected[u].add(v)
                neighbors_undirected[v].add(u)

    invoke_internal_hit_rate = invoke_internal_hit / max(invoke_total, 1)

    kept_nodes: Set[int] = set(sensitive_seed)
    frontier: Set[int] = set(sensitive_seed)

    for _ in range(max(0, int(sensitive_hops))):
        nxt: Set[int] = set()
        for u in frontier:
            nxt.update(neighbors_undirected.get(u, set()))
        nxt -= kept_nodes
        kept_nodes.update(nxt)
        frontier = nxt

    fallback_used = False
    if not kept_nodes:
        fallback_used = True
        if fallback_policy == "empty_dex":
            return _empty_dex_result(dim, "no_sensitive_seed", use_graph_behavior_hints)
        if fallback_policy == "all_capped":
            kept_nodes = set(range(min(m_all, max(1, int(fallback_max_methods)))))
        else:
            kept_nodes = _rank_methods_for_fallback(events_by_method, max(1, int(fallback_max_methods)))

    if max_methods_per_dex > 0 and len(kept_nodes) > max_methods_per_dex:
        sens = sorted(i for i in kept_nodes if i in sensitive_seed)
        rest = sorted(
            (i for i in kept_nodes if i not in sensitive_seed),
            key=lambda i: (len(events_by_method[i]), -i),
            reverse=True,
        )
        kept_nodes = set((sens + rest)[:max_methods_per_dex])

    kept_sorted = sorted(kept_nodes)
    old2new = {old: new for new, old in enumerate(kept_sorted)}

    x_np = np.stack([method_embs[i] for i in kept_sorted], axis=0).astype(np.float16)
    spans_np = np.asarray([method_spans[i] for i in kept_sorted], dtype=np.int32)
    sensitive_mask_np = np.asarray(
        [1 if i in sensitive_seed else 0 for i in kept_sorted],
        dtype=np.uint8,
    )

    kept_edges = sorted(
        (old2new[u], old2new[v])
        for u, v in edges_full
        if u in old2new and v in old2new
    )

    if kept_edges:
        edge_index = torch.from_numpy(np.asarray(kept_edges, dtype=np.int32).T.copy())
    else:
        edge_index = torch.empty((2, 0), dtype=torch.int32)

    scope = str(api_event_scope or "all_methods").lower()
    if scope not in {"all_methods", "graph_methods"}:
        raise ValueError(f"Unsupported api_event_scope: {api_event_scope}")

    api_source_indices = range(m_all) if scope == "all_methods" else kept_sorted

    api_events: List[ApiEvent] = []
    for old_idx in api_source_indices:
        api_events.extend(events_by_method[old_idx])

    num_api_events_raw = len(api_events)
    api_events = select_api_events(api_events, int(max_api_events_per_dex))
    num_api_events_kept = len(api_events)

    unique_tokens = {e.token for e in api_events}
    unique_ids = {stable_hash_id(e.token, num_api_buckets) for e in api_events}
    hash_collision_estimate = max(0, len(unique_tokens) - len(unique_ids))

    api_ids = torch.tensor(
        [stable_hash_id(e.token, num_api_buckets) for e in api_events],
        dtype=torch.long,
    )
    api_type_ids = torch.tensor([e.category_id for e in api_events], dtype=torch.uint8)
    api_sensitive_mask = torch.tensor([1 if e.sensitive else 0 for e in api_events], dtype=torch.uint8)

    api_method_index = torch.tensor(
        [old2new.get(e.old_method_idx, -1) for e in api_events],
        dtype=torch.int32,
    )
    api_in_graph_mask = (api_method_index >= 0).to(torch.uint8)

    valid_edges = [
        (int(mi), j)
        for j, mi in enumerate(api_method_index.tolist())
        if mi >= 0
    ]

    if valid_edges:
        method_api_edge_index = torch.tensor(valid_edges, dtype=torch.int32).t().contiguous()
    else:
        method_api_edge_index = torch.empty((2, 0), dtype=torch.int32)

    num_api_events_in_graph = int(api_in_graph_mask.sum().item())
    api_in_graph_ratio = num_api_events_in_graph / max(num_api_events_kept, 1)

    counts = np.zeros((len(API_CATEGORY_NAMES),), dtype=np.float32)
    for e in api_events:
        counts[int(e.category_id)] += 1.0

    total = float(counts.sum())
    if total > 0:
        counts /= total

    api_semantic_counts = api_semantic_counts_from_type_ids(api_type_ids.long())
    graph_semantic_counts = graph_semantic_counts_from_method_api_edges(
        api_type_ids.long(),
        method_api_edge_index.long(),
    )
    isolated_nodes, largest_component_ratio = graph_structure_stats(
        edge_index.long(),
        len(kept_sorted),
    )
    class_names: set[str] = set()
    for method in method_list:
        try:
            class_names.add(str(method.get_class_name() or ""))
        except Exception:
            pass
    class_count = len(class_names)
    known_type_count = int((api_type_ids.long() > 0).sum().item())

    result: Dict[str, Any] = {
        "call_x": torch.from_numpy(x_np).to(torch.float16),
        "call_edge_index": edge_index.contiguous(),
        "call_sensitive_mask": torch.from_numpy(sensitive_mask_np).to(torch.uint8),
        "method_spans": torch.from_numpy(spans_np).to(torch.int32),

        "api_ids": api_ids,
        "api_type_ids": api_type_ids,
        "api_sensitive_mask": api_sensitive_mask,
        "api_method_index": api_method_index,
        "api_in_graph_mask": api_in_graph_mask,
        "method_api_edge_index": method_api_edge_index,
        "api_category_counts": torch.from_numpy(counts).to(torch.float16),
        "api_semantic_category_counts": api_semantic_counts.to(torch.float16),
        "graph_semantic_category_counts": graph_semantic_counts.to(torch.float16),
        "observable_metadata": {
            "api_parse_ok": True,
            "api_parse_error": "",
            "api_event_count_raw": int(num_api_events_raw),
            "api_event_count_kept": int(num_api_events_kept),
            "api_truncated": bool(num_api_events_raw > num_api_events_kept),
            "api_truncation_ratio": (
                float(1.0 - num_api_events_kept / num_api_events_raw)
                if num_api_events_raw > 0
                else 0.0
            ),
            "api_known_type_count": known_type_count,
            "api_unknown_type_count": int(num_api_events_kept - known_type_count),
            "dex_parse_ok": True,
            "dex_file_count": 1,
            "dex_parse_success_ratio": 1.0,
            "class_count": int(class_count),
            "method_count": int(m_all),
            "graph_parse_ok": True,
            "graph_parse_error": "",
            "graph_timeout": False,
            "graph_node_count_raw": int(len(kept_sorted)),
            "graph_edge_count_raw": int(edge_index.shape[1]),
            "graph_isolated_node_count": int(isolated_nodes),
            "graph_largest_component_ratio": float(largest_component_ratio),
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        },

        "meta": {
            "graph": {
                "num_methods_all": int(m_all),
                "num_methods_sensitive_seed": int(len(sensitive_seed)),
                "num_methods_kept": int(len(kept_sorted)),
                "num_edges_kept": int(edge_index.shape[1]),
                "local_method_emb_dim": int(dim),
                "sensitive_hops": int(sensitive_hops),
                "fallback_used": bool(fallback_used),
                "max_methods_per_dex": int(max_methods_per_dex),
                "invoke_total": int(invoke_total),
                "invoke_internal_hit": int(invoke_internal_hit),
                "invoke_internal_hit_rate": float(invoke_internal_hit_rate),
                "graph_behavior_hints": bool(use_graph_behavior_hints),
                "graph_behavior_hint_names": [
                    "sensitive_method_flag",
                    "log_api_event_count",
                    "sensitive_api_ratio",
                    "semantic_api_ratio",
                ] if use_graph_behavior_hints else [],
            },
            "api": {
                "num_api_events_raw": int(num_api_events_raw),
                "num_api_events": int(num_api_events_kept),
                "api_truncated": bool(num_api_events_raw > num_api_events_kept),
                "num_api_events_in_graph": int(num_api_events_in_graph),
                "api_in_graph_ratio": float(api_in_graph_ratio),
                "num_unique_api_tokens": int(len(unique_tokens)),
                "num_unique_api_ids": int(len(unique_ids)),
                "hash_collision_estimate": int(hash_collision_estimate),
                "num_api_buckets": int(num_api_buckets),
                "num_api_categories": int(len(API_CATEGORY_NAMES)),
                "api_category_names": API_CATEGORY_NAMES,
                "semantic_category_names": list(SEMANTIC_CATEGORIES),
                "api_category_counts_source": "kept_events_after_truncation",
                "api_semantic_category_counts_source": "api_type_id_to_12d_semantic_taxonomy",
                "graph_semantic_category_counts_source": "method_api_edge_index_aligned_api_events",
                "hash_collision_scope": "per_dex_after_truncation", 
                "api_event_scope": scope,
                "framework_only": bool(framework_only),
                "include_descriptor": bool(include_descriptor),
                "max_api_events_per_dex": int(max_api_events_per_dex),
                "max_api_events_per_method": int(max_api_events_per_method),
                "representation": "ordered framework API behavior sequence with stable hashed ids",
                "id_reserved": {"pad": 0, "unk": 1},
                "category_method": "heuristic_v1_prefix_keyword",
            },
            "stored_dtypes": {
                "call_x": "float16",
                "call_edge_index": "int32",
                "call_sensitive_mask": "uint8",
                "method_spans": "int32",
                "api_ids": "int64",
                "api_type_ids": "uint8",
                "api_sensitive_mask": "uint8",
                "api_method_index": "int32",
                "api_in_graph_mask": "uint8",
                "method_api_edge_index": "int32",
                "api_category_counts": "float16",
                "api_semantic_category_counts": "float16",
                "graph_semantic_category_counts": "float16",
            },
        },
    }

    if keep_method_names:
        result["call_method_names"] = [method_names[i] for i in kept_sorted]

    if keep_api_tokens:
        result["api_tokens"] = [e.token for e in api_events]

    return result


def process_apk(
    apk_path: Path,
    out_dir: Path,
    split: str,
    cfg: Dict[str, Any],
) -> Tuple[bool, str]:
    safe_mkdir(out_dir)

    sha = sha256_file(apk_path)
    out_path = out_dir / f"{sha}.pt"

    if cfg["resume"] and out_path.exists():
        return True, ""

    try:
        dex_entries = list_dex_entries(apk_path)
        if not dex_entries:
            return False, "no classes*.dex"

        results: List[Dict[str, Any]] = []
        dex_errors: list[str] = []

        with zipfile.ZipFile(apk_path, "r") as zf:
            for entry in dex_entries:
                dex_name = os.path.basename(entry)

                try:
                    dex_bytes = zf.read(entry)
                    dvm = DEX(dex_bytes)

                    item = build_graph_api_for_dex(
                        dvm=dvm,
                        raw_bytes=dex_bytes,
                        vocab_size=cfg["vocab_size"],
                        keep_method_names=cfg["keep_method_names"],
                        keep_api_tokens=cfg["keep_api_tokens"],
                        sensitive_hops=cfg["sensitive_hops"],
                        max_methods_per_dex=cfg["max_methods_per_dex"],
                        fallback_max_methods=cfg["fallback_max_methods"],
                        num_api_buckets=cfg["num_api_buckets"],
                        max_api_events_per_dex=cfg["max_api_events_per_dex"],
                        max_api_events_per_method=cfg["max_api_events_per_method"],
                        api_event_scope=cfg["api_event_scope"],
                        framework_only=cfg["framework_only"],
                        include_descriptor=cfg["include_descriptor"],
                        fallback_policy=cfg["fallback_policy"],
                        use_graph_behavior_hints=cfg["use_graph_behavior_hints"],
                    )

                    item["dex_name"] = dex_name
                    item["meta"].update({
                        "sha256_apk": sha,
                        "apk_name": apk_path.name,
                        "dex_name": dex_name,
                        "split": split,
                        "representation": {
                            "graph_branch": "sensitive_method_call_graph",
                            "api_branch": "framework_api_behavior_sequence",
                            "alignment_anchor": "method_api_edge_index",
                        },
                    })

                    results.append(item)

                except Exception as exc:
                    dex_errors.append(f"{dex_name}: {type(exc).__name__}: {exc}")
                    print(f"[WARN] DEX failed: {dex_name} | {exc}", file=sys.stderr, flush=True)

        if not results:
            return False, "all dex entries failed"

        observable_parts = [
            item.get("observable_metadata", {})
            for item in results
            if isinstance(item, dict)
        ]
        raw_api = sum(int(meta.get("api_event_count_raw", 0)) for meta in observable_parts)
        kept_api = sum(int(meta.get("api_event_count_kept", 0)) for meta in observable_parts)
        known_api = sum(int(meta.get("api_known_type_count", 0)) for meta in observable_parts)
        graph_nodes = sum(int(meta.get("graph_node_count_raw", 0)) for meta in observable_parts)
        graph_edges = sum(int(meta.get("graph_edge_count_raw", 0)) for meta in observable_parts)
        graph_isolated = sum(int(meta.get("graph_isolated_node_count", 0)) for meta in observable_parts)
        dex_success_ratio = len(results) / max(len(dex_entries), 1)
        observable_metadata = {
            "api_parse_ok": bool(results),
            "api_parse_error": "; ".join(dex_errors),
            "api_event_count_raw": raw_api,
            "api_event_count_kept": kept_api,
            "api_truncated": kept_api < raw_api,
            "api_truncation_ratio": float(1.0 - kept_api / raw_api) if raw_api > 0 else 0.0,
            "api_known_type_count": known_api,
            "api_unknown_type_count": max(0, kept_api - known_api),
            "dex_parse_ok": bool(results),
            "dex_file_count": len(dex_entries),
            "dex_parse_success_ratio": dex_success_ratio,
            "class_count": sum(int(meta.get("class_count", 0)) for meta in observable_parts),
            "method_count": sum(int(meta.get("method_count", 0)) for meta in observable_parts),
            "graph_parse_ok": bool(results),
            "graph_parse_error": "; ".join(dex_errors),
            "graph_timeout": False,
            "graph_node_count_raw": graph_nodes,
            "graph_edge_count_raw": graph_edges,
            "graph_isolated_node_count": graph_isolated,
            "graph_largest_component_ratio": (
                max(
                    float(meta.get("graph_largest_component_ratio", 0.0))
                    * int(meta.get("graph_node_count_raw", 0))
                    for meta in observable_parts
                )
                / max(graph_nodes, 1)
                if observable_parts
                else 0.0
            ),
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        }
        payload = {
            "dex_list": results,
            "observable_metadata": observable_metadata,
            "direct_build_meta": {
                "pt_schema_version": 4,
                "schema_version": OBSERVABLE_SCHEMA_VERSION,
                "dex_file_count": len(dex_entries),
                "dex_success_ratio": dex_success_ratio,
            },
        }
        atomic_torch_save(payload, out_path)
        return True, ""

    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def collect_apks(apk_root: Path, splits: Sequence[str]) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []

    for split in splits:
        split_dir = apk_root / split
        if not split_dir.exists():
            continue

        for path in sorted(split_dir.glob("*.apk")):
            items.append((split, path))

    return items


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")

    return cfg


def parse_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    data = raw.get("data", {})
    hp = raw.get("hyperparameters", {})
    graph = raw.get("graph", {})
    api = raw.get("api", {})
    storage = raw.get("storage", {})
    execution = raw.get("execution", {})

    cfg = {
        "apk_root": Path(data["apk_root"]),
        "out_root": Path(data["out_root"]),
        "splits": list(data.get("splits", ["train", "val", "test"])),

        "vocab_size": int(hp.get("vocab_size", 256)),

        "sensitive_hops": int(graph.get("sensitive_hops", 1)),
        "max_methods_per_dex": int(graph.get("max_methods_per_dex", 4096)),
        "fallback_max_methods": int(graph.get("fallback_max_methods", 512)),
        "fallback_policy": str(graph.get("fallback_policy", "api_rich")),
        "use_graph_behavior_hints": bool(graph.get("use_behavior_hints", True)),

        "num_api_buckets": int(api.get("num_hash_buckets", 8192)),
        "max_api_events_per_dex": int(api.get("max_events_per_dex", 1024)),
        "max_api_events_per_method": int(api.get("max_events_per_method", 32)),
        "api_event_scope": str(api.get("event_scope", "all_methods")),
        "framework_only": bool(api.get("framework_only", True)),
        "include_descriptor": bool(api.get("include_descriptor", False)),

        "keep_method_names": bool(storage.get("keep_method_names", False)),
        "keep_api_tokens": bool(storage.get("keep_api_tokens", False)),

        "workers": int(execution.get("workers", 1)),
        "resume": bool(execution.get("resume", True)),
    }

    if cfg["vocab_size"] <= 0:
        raise ValueError("hyperparameters.vocab_size must be positive")
    if cfg["sensitive_hops"] < 0:
        raise ValueError("graph.sensitive_hops must be >= 0")
    if cfg["max_methods_per_dex"] <= 0:
        raise ValueError("graph.max_methods_per_dex must be positive")
    if cfg["fallback_max_methods"] <= 0:
        raise ValueError("graph.fallback_max_methods must be positive")
    if cfg["fallback_policy"] not in {"api_rich", "all_capped", "empty_dex"}:
        raise ValueError("graph.fallback_policy must be one of: api_rich, all_capped, empty_dex")

    if cfg["num_api_buckets"] <= 0:
        raise ValueError("api.num_hash_buckets must be positive")
    if cfg["max_api_events_per_dex"] <= 0:
        raise ValueError("api.max_events_per_dex must be positive")
    if cfg["max_api_events_per_method"] <= 0:
        raise ValueError("api.max_events_per_method must be positive")
    if cfg["api_event_scope"] not in {"all_methods", "graph_methods"}:
        raise ValueError("api.event_scope must be one of: all_methods, graph_methods")

    if cfg["workers"] <= 0:
        raise ValueError("execution.workers must be >= 1")

    return cfg


def _worker(job: Tuple[str, str, str, Dict[str, Any]]) -> Tuple[bool, str, str, str]:
    split, apk_path_str, out_root_str, cfg = job
    apk_path = Path(apk_path_str)
    out_dir = Path(out_root_str) / split
    ok, err = process_apk(apk_path, out_dir, split, cfg)
    return ok, split, apk_path.name, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract graph + API sequence .pt features")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config(load_config(Path(args.config)))
    safe_mkdir(cfg["out_root"])

    items = collect_apks(cfg["apk_root"], cfg["splits"])
    if not items:
        print("No APKs found. Check apk_root and splits.")
        return

    print("Graph+API extraction config:")
    printable = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"Found {len(items)} APKs")

    jobs = [
        (split, str(apk_path), str(cfg["out_root"]), cfg)
        for split, apk_path in items
    ]

    ok = 0
    fail = 0
    failed: List[Dict[str, str]] = []

    if cfg["workers"] == 1:
        iterator = tqdm(jobs, desc="Build graph+API .pt", unit="apk")
        for job in iterator:
            succ, split, name, err = _worker(job)
            if succ:
                ok += 1
            else:
                fail += 1
                failed.append({"split": split, "apk": name, "reason": err})
    else:
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as ex:
            future_map = {ex.submit(_worker, job): job for job in jobs}

            for fut in tqdm(as_completed(future_map), total=len(jobs), desc="Build graph+API .pt", unit="apk"):
                job = future_map[fut]

                try:
                    succ, split, name, err = fut.result()
                except Exception as exc:
                    succ, split, name = False, job[0], Path(job[1]).name
                    err = f"{type(exc).__name__}: {exc}"

                if succ:
                    ok += 1
                else:
                    fail += 1
                    failed.append({"split": split, "apk": name, "reason": err})

    print(f"done. ok={ok}, fail={fail}")

    if failed:
        failed_path = cfg["out_root"] / "failed_graph_api.json"
        failed_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"failed list -> {failed_path}")


if __name__ == "__main__":
    main()
