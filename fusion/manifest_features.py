from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch

from fusion.quality import OBSERVABLE_SCHEMA_VERSION
import yaml


ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

DEFAULT_CATEGORIES = [
    "network",
    "sms",
    "location",
    "contacts",
    "storage",
    "telephony",
    "camera_media",
    "receiver",
    "component_exposure",
    "dynamic_loading",
    "crypto",
    "system_settings",
]

CATEGORY_KEYWORDS = {
    "network": ["internet", "network", "wifi", "connectivity", "http", "socket"],
    "sms": ["sms", "mms", "wap_push"],
    "location": ["location", "gps"],
    "contacts": ["contacts", "accounts", "profile", "calendar"],
    "storage": ["storage", "external_storage", "media", "download"],
    "telephony": ["phone", "call", "telephony", "read_phone_state", "imei"],
    "camera_media": ["camera", "record_audio", "microphone", "audio", "video", "image"],
    "receiver": ["boot_completed", "package_added", "package_removed", "sms_received", "battery", "receiver"],
    "component_exposure": ["exported", "browsable", "launcher", "main"],
    "dynamic_loading": ["query_all_packages", "request_install_packages", "dex", "package"],
    "crypto": ["keystore", "credential", "biometric", "fingerprint"],
    "system_settings": ["settings", "system_alert", "write_settings", "notification", "admin"],
}


@dataclass
class ManifestRecord:
    sid: str
    apk_name: str = ""
    sha256: str = ""
    permissions: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    services: list[str] = field(default_factory=list)
    receivers: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    intent_actions: list[str] = field(default_factory=list)
    intent_categories: list[str] = field(default_factory=list)
    uses_features: list[str] = field(default_factory=list)
    min_sdk: int = 0
    target_sdk: int = 0
    debuggable: bool = False
    exported_component_count: int = 0
    component_count: int = 0
    parse_error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "sid": self.sid,
            "apk_name": self.apk_name,
            "sha256": self.sha256,
            "permissions": sorted(set(self.permissions)),
            "activities": sorted(set(self.activities)),
            "services": sorted(set(self.services)),
            "receivers": sorted(set(self.receivers)),
            "providers": sorted(set(self.providers)),
            "intent_actions": sorted(set(self.intent_actions)),
            "intent_categories": sorted(set(self.intent_categories)),
            "uses_features": sorted(set(self.uses_features)),
            "min_sdk": int(self.min_sdk or 0),
            "target_sdk": int(self.target_sdk or 0),
            "debuggable": bool(self.debuggable),
            "exported_component_count": int(self.exported_component_count or 0),
            "component_count": int(self.component_count or 0),
            "parse_error": self.parse_error,
        }


def _lower_tokens(values: Iterable[Any]) -> list[str]:
    out = []
    for value in values or []:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text.lower())
    return out


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _android_attr(elem, name: str, default: str | None = None):
    if elem is None:
        return default
    return elem.attrib.get(ANDROID_NS + name, elem.attrib.get(name, default))


def _xml_components_and_intents(manifest_xml) -> tuple[dict[str, list[str]], list[str], list[str], int]:
    components = {"activity": [], "service": [], "receiver": [], "provider": []}
    actions: list[str] = []
    categories: list[str] = []
    exported_count = 0
    if manifest_xml is None:
        return components, actions, categories, exported_count

    app = manifest_xml.find("application")
    if app is None:
        return components, actions, categories, exported_count

    for tag in components:
        for elem in app.findall(tag):
            name = _android_attr(elem, "name")
            if name:
                components[tag].append(str(name))
            intent_filters = elem.findall("intent-filter")
            exported_raw = _android_attr(elem, "exported")
            exported = str(exported_raw).lower() == "true" if exported_raw is not None else bool(intent_filters)
            if exported:
                exported_count += 1
            for intent_filter in intent_filters:
                for action in intent_filter.findall("action"):
                    action_name = _android_attr(action, "name")
                    if action_name:
                        actions.append(str(action_name))
                for category in intent_filter.findall("category"):
                    category_name = _android_attr(category, "name")
                    if category_name:
                        categories.append(str(category_name))
    return components, actions, categories, exported_count


def extract_manifest_record(apk_path: str | Path, sid: str | None = None) -> ManifestRecord:
    apk_path = Path(apk_path)
    rec = ManifestRecord(sid=(sid or apk_path.stem).lower(), apk_name=apk_path.name)
    try:
        from androguard.core.apk import APK

        apk = APK(str(apk_path))
        rec.permissions = _lower_tokens(apk.get_permissions() or [])
        rec.activities = _lower_tokens(apk.get_activities() or [])
        rec.services = _lower_tokens(apk.get_services() or [])
        rec.receivers = _lower_tokens(apk.get_receivers() or [])
        rec.providers = _lower_tokens(apk.get_providers() or [])
        rec.uses_features = _lower_tokens(apk.get_features() or [])
        rec.min_sdk = _safe_int(apk.get_min_sdk_version())
        rec.target_sdk = _safe_int(apk.get_target_sdk_version())
        try:
            rec.debuggable = bool(apk.is_debuggable())
        except Exception:
            rec.debuggable = False

        manifest_xml = None
        try:
            manifest_xml = apk.get_android_manifest_xml()
        except Exception:
            manifest_xml = None
        xml_components, actions, categories, exported_count = _xml_components_and_intents(manifest_xml)
        rec.intent_actions = _lower_tokens(actions)
        rec.intent_categories = _lower_tokens(categories)
        rec.exported_component_count = int(exported_count)

        if not rec.activities:
            rec.activities = _lower_tokens(xml_components["activity"])
        if not rec.services:
            rec.services = _lower_tokens(xml_components["service"])
        if not rec.receivers:
            rec.receivers = _lower_tokens(xml_components["receiver"])
        if not rec.providers:
            rec.providers = _lower_tokens(xml_components["provider"])
        rec.component_count = len(rec.activities) + len(rec.services) + len(rec.receivers) + len(rec.providers)
    except Exception as exc:
        rec.parse_error = f"{type(exc).__name__}: {exc}"
    return rec


def category_counts_from_strings(values: Iterable[str], categories: list[str] | None = None) -> torch.Tensor:
    categories = categories or DEFAULT_CATEGORIES
    counts = torch.zeros((len(categories),), dtype=torch.float32)
    lowered = [str(v).lower() for v in values or []]
    for idx, category in enumerate(categories):
        keywords = CATEGORY_KEYWORDS.get(category, [category])
        for value in lowered:
            if any(k in value for k in keywords):
                counts[idx] += 1.0
    return counts


def category_counts_from_record(record: dict[str, Any], categories: list[str] | None = None) -> torch.Tensor:
    categories = categories or DEFAULT_CATEGORIES
    values: list[str] = []
    for key in (
        "permissions",
        "intent_actions",
        "intent_categories",
        "uses_features",
        "activities",
        "services",
        "receivers",
        "providers",
    ):
        values.extend(record.get(key) or [])
    counts = category_counts_from_strings(values, categories)
    if "component_exposure" in categories:
        counts[categories.index("component_exposure")] += float(record.get("exported_component_count", 0) or 0)
    if "receiver" in categories:
        counts[categories.index("receiver")] += float(len(record.get("receivers") or []))
    return counts


# Normalisation divisors for manifest stats.  These are derived from coarse upper
# bounds of the training-distribution so each stat lands roughly in [0, 1].
# Revisit if the data source changes substantially (e.g. different app stores,
# much larger apps).
_STAT_NORM_PERMISSIONS = 6.0
_STAT_NORM_COMPONENT_TYPE = 5.0
_STAT_NORM_SDK = 35.0
_STAT_NORM_TOTAL_COMPONENTS = 80.0


def manifest_stats_from_record(record: dict[str, Any]) -> torch.Tensor:
    component_count = float(record.get("component_count", 0) or 0)
    exported_count = float(record.get("exported_component_count", 0) or 0)
    stats = torch.tensor(
        [
            math.log1p(len(record.get("permissions") or [])) / _STAT_NORM_PERMISSIONS,
            math.log1p(len(record.get("activities") or [])) / _STAT_NORM_COMPONENT_TYPE,
            math.log1p(len(record.get("services") or [])) / _STAT_NORM_COMPONENT_TYPE,
            math.log1p(len(record.get("receivers") or [])) / _STAT_NORM_COMPONENT_TYPE,
            math.log1p(len(record.get("providers") or [])) / _STAT_NORM_COMPONENT_TYPE,
            math.log1p(len(record.get("uses_features") or [])) / _STAT_NORM_COMPONENT_TYPE,
            min(float(record.get("min_sdk", 0) or 0) / _STAT_NORM_SDK, 1.0),
            min(float(record.get("target_sdk", 0) or 0) / _STAT_NORM_SDK, 1.0),
            1.0 if record.get("debuggable") else 0.0,
            min(exported_count / max(component_count, 1.0), 1.0),
            min(component_count / _STAT_NORM_TOTAL_COMPONENTS, 1.0),
        ],
        dtype=torch.float32,
    )
    return torch.nan_to_num(stats, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def build_manifest_vocab(
    records: Iterable[dict[str, Any]],
    max_permissions: int = 128,
    max_intents: int = 64,
    max_features: int = 32,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    perm_counter: Counter[str] = Counter()
    intent_counter: Counter[str] = Counter()
    feature_counter: Counter[str] = Counter()
    for rec in records:
        perm_counter.update(_lower_tokens(rec.get("permissions") or []))
        intent_counter.update(_lower_tokens(rec.get("intent_actions") or []))
        intent_counter.update(_lower_tokens(rec.get("intent_categories") or []))
        feature_counter.update(_lower_tokens(rec.get("uses_features") or []))
    return {
        "categories": list(categories or DEFAULT_CATEGORIES),
        "permission_vocab": [k for k, _ in perm_counter.most_common(max_permissions)],
        "intent_vocab": [k for k, _ in intent_counter.most_common(max_intents)],
        "feature_vocab": [k for k, _ in feature_counter.most_common(max_features)],
    }


def validate_manifest_vocab(
    vocab: dict[str, Any],
    *,
    require_train_metadata: bool = False,
    allow_empty: bool = False,
) -> None:
    categories = list(vocab.get("categories") or [])
    if categories != list(DEFAULT_CATEGORIES):
        raise ValueError("Manifest vocab categories must match the fixed robust semantic taxonomy")

    if require_train_metadata:
        metadata = vocab.get("metadata") or {}
        if metadata.get("source_split") != "train" or metadata.get("leakage_guard") != "train_only":
            raise ValueError("Manifest vocab must be built from the train split with leakage_guard=train_only")

    if allow_empty:
        return

    has_vocab = any(
        bool(vocab.get(key))
        for key in ("permission_vocab", "intent_vocab", "feature_vocab")
    )
    if not has_vocab:
        raise ValueError(
            "Manifest vocab is empty. Build it from the train split, or set allow_empty_vocab=true only for debugging."
        )


def load_manifest_vocab(
    path: str | Path,
    *,
    require_train_metadata: bool = False,
    allow_empty: bool = False,
) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        vocab = yaml.safe_load(f) or {}
    vocab.setdefault("categories", list(DEFAULT_CATEGORIES))
    vocab.setdefault("permission_vocab", [])
    vocab.setdefault("intent_vocab", [])
    vocab.setdefault("feature_vocab", [])
    validate_manifest_vocab(
        vocab,
        require_train_metadata=require_train_metadata,
        allow_empty=allow_empty,
    )
    return vocab


def save_manifest_vocab(vocab: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vocab, f, sort_keys=False, allow_unicode=False)


def vectorize_manifest_record(
    record: dict[str, Any],
    vocab: dict[str, Any],
    manifest_dim: int = 256,
) -> dict[str, Any]:
    categories = list(vocab.get("categories") or DEFAULT_CATEGORIES)
    permission_vocab = list(vocab.get("permission_vocab") or [])
    intent_vocab = list(vocab.get("intent_vocab") or [])
    feature_vocab = list(vocab.get("feature_vocab") or [])

    perm_index = {v: i for i, v in enumerate(permission_vocab)}
    intent_index = {v: i for i, v in enumerate(intent_vocab)}
    feature_index = {v: i for i, v in enumerate(feature_vocab)}

    permissions = _lower_tokens(record.get("permissions") or [])
    intents = _lower_tokens(record.get("intent_actions") or []) + _lower_tokens(record.get("intent_categories") or [])
    features = _lower_tokens(record.get("uses_features") or [])

    perm_vec = torch.zeros((len(permission_vocab),), dtype=torch.float32)
    perm_ids = []
    for item in permissions:
        idx = perm_index.get(item)
        if idx is not None:
            perm_vec[idx] = 1.0
            perm_ids.append(idx + 1)

    intent_vec = torch.zeros((len(intent_vocab),), dtype=torch.float32)
    intent_ids = []
    for item in intents:
        idx = intent_index.get(item)
        if idx is not None:
            intent_vec[idx] = 1.0
            intent_ids.append(idx + 1)

    feature_vec = torch.zeros((len(feature_vocab),), dtype=torch.float32)
    for item in features:
        idx = feature_index.get(item)
        if idx is not None:
            feature_vec[idx] = 1.0

    category_counts = category_counts_from_record(record, categories)
    component_values: list[str] = []
    for key in ("activities", "services", "receivers", "providers"):
        component_values.extend(record.get(key) or [])
    component_category_counts = category_counts_from_strings(component_values, categories)
    if "component_exposure" in categories:
        component_category_counts[categories.index("component_exposure")] += float(
            record.get("exported_component_count", 0) or 0
        )
    if "receiver" in categories:
        component_category_counts[categories.index("receiver")] += float(
            len(record.get("receivers") or [])
        )
    permission_category_map = torch.stack(
        [category_counts_from_strings([token], categories) for token in permission_vocab],
        dim=0,
    ) if permission_vocab else torch.zeros((0, len(categories)), dtype=torch.float32)
    intent_category_map = torch.stack(
        [category_counts_from_strings([token], categories) for token in intent_vocab],
        dim=0,
    ) if intent_vocab else torch.zeros((0, len(categories)), dtype=torch.float32)
    category_norm = category_counts / category_counts.sum().clamp_min(1.0)
    stats = manifest_stats_from_record(record)
    parts = [perm_vec, intent_vec, feature_vec, category_norm, stats]
    raw_dim = sum(int(p.numel()) for p in parts)
    if raw_dim > manifest_dim:
        raise ValueError(
            "manifest_dim is too small for the selected Manifest vocab layout: "
            f"manifest_dim={manifest_dim}, required={raw_dim} "
            f"(permissions={len(permission_vocab)}, intents={len(intent_vocab)}, "
            f"features={len(feature_vocab)}, categories={len(categories)}, stats={int(stats.numel())})"
        )
    manifest_x = torch.cat([p.float().view(-1) for p in parts], dim=0)
    if manifest_x.numel() < manifest_dim:
        manifest_x = torch.cat([manifest_x, torch.zeros((manifest_dim - manifest_x.numel(),), dtype=torch.float32)])

    parse_error = str(record.get("parse_error") or "")
    has_manifest = not parse_error and (
        len(permissions)
        + len(intents)
        + len(features)
        + int(record.get("component_count", 0) or 0)
        + int(record.get("exported_component_count", 0) or 0)
        + int(record.get("min_sdk", 0) or 0)
        + int(record.get("target_sdk", 0) or 0)
        + int(bool(record.get("debuggable")))
        > 0
    )
    coverage_values = []
    coverage_meta = {}
    for name, tokens, index in (
        ("permission", permissions, perm_index),
        ("intent", intents, intent_index),
        ("feature", features, feature_index),
    ):
        unique_tokens = set(tokens)
        coverage = (
            len(unique_tokens.intersection(index)) / len(unique_tokens)
            if unique_tokens
            else 1.0
        )
        coverage_meta[f"{name}_vocab_coverage"] = float(coverage)
        if unique_tokens:
            coverage_values.append(float(coverage))
    if has_manifest and coverage_values:
        # Parser success contributes the base score. Vocabulary coverage
        # modulates quality but must not mark a valid, OOV-heavy Manifest as
        # unavailable.
        q_manifest = 0.5 + 0.5 * (sum(coverage_values) / len(coverage_values))
    else:
        q_manifest = 1.0 if has_manifest else 0.0
    manifest_vocab_coverage = (
        float(sum(coverage_values) / len(coverage_values))
        if coverage_values
        else 1.0
    )

    return {
        "manifest_x": manifest_x,
        "manifest_permission_ids": torch.tensor(sorted(set(perm_ids)), dtype=torch.long),
        "manifest_intent_ids": torch.tensor(sorted(set(intent_ids)), dtype=torch.long),
        "manifest_category_counts": category_counts.float(),
        "manifest_component_category_counts": component_category_counts.float(),
        "manifest_permission_category_map": permission_category_map.float(),
        "manifest_intent_category_map": intent_category_map.float(),
        "manifest_stats": stats.float(),
        "q_manifest": torch.tensor([q_manifest], dtype=torch.float32),
        # Clean extraction never fabricates synthetic degradation metadata.
        "pert_manifest": torch.tensor([0.0], dtype=torch.float32),
        "observable_metadata": {
            "manifest_parse_ok": not bool(parse_error),
            "manifest_parse_error": parse_error,
            "manifest_has_content": bool(has_manifest),
            "manifest_vocab_coverage": manifest_vocab_coverage,
            "manifest_permission_count": len(set(permissions)),
            "manifest_component_count": int(record.get("component_count", 0) or 0),
            "manifest_intent_count": len(set(intents)),
            "schema_version": OBSERVABLE_SCHEMA_VERSION,
        },
        "manifest_meta": {
            "apk_name": record.get("apk_name", ""),
            "sha256": record.get("sha256", ""),
            "permissions": permissions,
            "intent_actions": _lower_tokens(record.get("intent_actions") or []),
            "intent_categories": _lower_tokens(record.get("intent_categories") or []),
            "uses_features": features,
            "component_count": int(record.get("component_count", 0) or 0),
            "exported_component_count": int(record.get("exported_component_count", 0) or 0),
            "min_sdk": int(record.get("min_sdk", 0) or 0),
            "target_sdk": int(record.get("target_sdk", 0) or 0),
            "debuggable": bool(record.get("debuggable")),
            "parse_error": parse_error,
            "quality_score": float(q_manifest),
            **coverage_meta,
        },
        "manifest_permission_dim": len(permission_vocab),
        "manifest_intent_dim": len(intent_vocab),
        "manifest_feature_dim": len(feature_vocab),
    }


def read_manifest_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    records = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = str(rec.get("sid") or rec.get("sha256") or "").lower()
            if sid:
                records[sid] = rec
    return records
