import csv
import random
import os
import json
from collections import defaultdict
from datetime import datetime
from math import floor
import copy


# ================================================================
#  配置
# ================================================================

CONFIG = {
    "csv_path": "/Users/tsing/Downloads/latest_with-added-date.csv",
    "output_dir": "./dataset_split",
    "seed": 42,

    "train_size": 10000,
    "val_size": 2000,
    "test_size": 1400,

    "train_years": [2020,2021,2022],
    "val_years": [2023],
    "test_years": [2024],

    "vt_configs": {
        "strict":  {"malware_min": 10, "benign_max": 0},
        "relaxed": {"malware_min": 4,  "benign_max": 0},
    },
    "active_vt_config": "relaxed",

    "benign_mode": "mixed",
    "benign_play_ratio": 0.7,

    "min_apk_size": 50000,
    "max_apk_size": 100_000_000,

    "one_version_per_pkg": True,
}

# 固定 seed offset，避免 hash randomization
SEED_OFFSETS = {"train": 0, "val": 1, "test": 2}


# ================================================================
#  最大余数法（一次性补齐，无循环）
# ================================================================

def largest_remainder_alloc(pools, total):
    """
    最大余数法分配 quota
    overflow 按剩余容量比例再分配（只看容量 > 0，不排除 capped）
    """
    pool_total = sum(pools.values())
    if pool_total == 0:
        return {k: 0 for k in pools}

    achievable = min(total, pool_total)

    exact = {k: achievable * v / pool_total for k, v in pools.items()}
    floored = {k: floor(v) for k, v in exact.items()}

    remainder = achievable - sum(floored.values())
    fractional = {k: exact[k] - floored[k] for k in exact}
    ranked = sorted(fractional, key=lambda k: fractional[k], reverse=True)

    quotas = dict(floored)
    for i in range(int(remainder)):
        quotas[ranked[i]] += 1

    # Cap + overflow 再分配
    overflow = 0
    for k in quotas:
        if quotas[k] > pools[k]:
            overflow += quotas[k] - pools[k]
            quotas[k] = pools[k]

    if overflow > 0:
        remaining_capacity = {
            k: pools[k] - quotas[k]
            for k in quotas
            if pools[k] - quotas[k] > 0
        }
        if remaining_capacity:
            extra = _lra_inner(remaining_capacity, overflow)
            for k, v in extra.items():
                quotas[k] += v

    return quotas


def _lra_inner(pools, total):
    """内部最大余数法，用于 overflow 再分配"""
    pool_total = sum(pools.values())
    if pool_total == 0:
        return {k: 0 for k in pools}

    achievable = min(total, pool_total)
    exact = {k: achievable * v / pool_total for k, v in pools.items()}
    floored = {k: min(floor(v), pools[k]) for k, v in exact.items()}

    remainder = int(achievable - sum(floored.values()))
    fractional = {k: exact[k] - floored[k] for k in exact if floored[k] < pools[k]}
    ranked = sorted(fractional, key=lambda k: fractional[k], reverse=True)

    quotas = dict(floored)
    for i in range(min(remainder, len(ranked))):
        if quotas[ranked[i]] < pools[ranked[i]]:
            quotas[ranked[i]] += 1

    return quotas
# ================================================================
#  扫描 + 筛选
# ================================================================

def parse_added_date(added_date_str):
    try:
        return datetime.strptime(added_date_str.strip(), "%Y-%m-%d %H:%M:%S.%f")
    except (ValueError, AttributeError):
        return None


def extract_pkg_name(row):
    pkg = row.get("pkg_name", "").strip()
    if not pkg:
        pkg = f"__unknown_{row['sha256']}"
    return pkg


def scan_and_filter(cfg):
    vt_cfg = cfg["vt_configs"][cfg["active_vt_config"]]
    all_years = set(cfg["train_years"] + cfg["val_years"] + cfg["test_years"])

    benign = defaultdict(list)
    malware = defaultdict(list)
    total = 0
    b_count = 0
    m_count = 0

    print(f"扫描 {cfg['csv_path']} ...")
    print(f"VT 配置: {cfg['active_vt_config']} -> {vt_cfg}")

    with open(cfg["csv_path"], "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 2_000_000 == 0:
                print(f"  {total // 1_000_000}M 行 | 良性池 {b_count} | 恶意池 {m_count}")

            try:
                vt = int(row.get("vt_detection", -1))
                apk_size = int(row.get("apk_size", 0))
            except (ValueError, TypeError):
                continue

            if apk_size < cfg["min_apk_size"] or apk_size > cfg["max_apk_size"]:
                continue

            added_dt = parse_added_date(row.get("added", ""))
            if not added_dt:
                continue
            year = added_dt.year
            if year not in all_years:
                continue

            sha = row.get("sha256", "").strip()
            if not sha:
                continue

            pkg = extract_pkg_name(row)
            markets = row.get("markets", "").lower()
            market_tag = "play" if "play" in markets else "other"

            if vt <= vt_cfg["benign_max"]:
                benign[year].append((sha, pkg, market_tag, added_dt))
                b_count += 1
            elif vt >= vt_cfg["malware_min"]:
                malware[year].append((sha, pkg, market_tag, added_dt))
                m_count += 1

    print(f"\n扫描完成: {total} 行")
    _print_pool_stats(benign, malware, all_years)
    return benign, malware


def _print_pool_stats(benign, malware, all_years):
    print(f"\n{'年份':>6} {'良性':>10} {'恶意':>10}")
    print("-" * 28)
    for y in sorted(all_years):
        print(f"{y:>6} {len(benign[y]):>10} {len(malware[y]):>10}")


# ================================================================
#  pkg_name 去泄漏 + 每 pkg 每 split 只保留一个版本
# ================================================================

def deduplicate_by_pkg(benign, malware, cfg):
    split_map = {}
    for y in cfg["train_years"]:
        split_map[y] = "train"
    for y in cfg["val_years"]:
        split_map[y] = "val"
    for y in cfg["test_years"]:
        split_map[y] = "test"

    priority = {"test": 0, "val": 1, "train": 2}

    # 收集每个 pkg 出现在哪些 split
    pkg_splits = defaultdict(set)
    for year, items in list(benign.items()) + list(malware.items()):
        sp = split_map.get(year)
        if not sp:
            continue
        for sha, pkg, market, dex_dt in items:
            pkg_splits[pkg].add(sp)

    pkg_owner = {}
    cross_leak = 0
    for pkg, splits in pkg_splits.items():
        if len(splits) > 1 and not pkg.startswith("__unknown_"):
            cross_leak += 1
        pkg_owner[pkg] = min(splits, key=lambda s: priority[s])

    print(f"\n跨 split 冲突 pkg: {cross_leak}")

    def dedup_items(data_by_year, label_name):
        # 过滤不属于当前 split 的
        filtered = defaultdict(list)
        removed = 0
        for year, items in data_by_year.items():
            sp = split_map.get(year)
            for item in items:
                sha, pkg, market, dex_dt = item
                if pkg_owner.get(pkg) == sp:
                    filtered[year].append(item)
                else:
                    removed += 1

        if cfg["one_version_per_pkg"]:
            # 每个 (pkg, split) 只保留 added_date 最新的版本
            split_pkg_best = defaultdict(dict)
            for year, items in filtered.items():
                sp = split_map.get(year)
                for item in items:
                    sha, pkg, market, dex_dt = item
                    existing = split_pkg_best[sp].get(pkg)
                    if existing is None or dex_dt > existing[1][3]:
                        split_pkg_best[sp][pkg] = (year, item)

            deduped = defaultdict(list)
            kept = 0
            for sp, pkg_map in split_pkg_best.items():
                for pkg, (year, item) in pkg_map.items():
                    deduped[year].append(item)
                    kept += 1

            total_before = sum(len(v) for v in filtered.values())
            print(f"  {label_name}: 跨 split -{removed}, "
                  f"同 split 去重 -{total_before - kept}, 保留 {kept}")
            return deduped
        else:
            print(f"  {label_name}: 跨 split -{removed}")
            return filtered

    clean_benign = dedup_items(benign, "良性")
    clean_malware = dedup_items(malware, "恶意")

    all_years = set(cfg["train_years"] + cfg["val_years"] + cfg["test_years"])
    print(f"\n去泄漏后:")
    _print_pool_stats(clean_benign, clean_malware, all_years)

    return clean_benign, clean_malware


# ================================================================
#  Benign 混合采样（合并池兜底）
# ================================================================
def sample_benign_mixed(pool, target, play_ratio, rng):
    """
    按 play/other 比例采样，deficit 按剩余容量比例补齐
    最后合并池兜底，保证返回 min(target, len(pool)) 条
    """
    play_items = [x for x in pool if x[2] == "play"]
    other_items = [x for x in pool if x[2] != "play"]

    n_play_want = int(target * play_ratio)
    n_other_want = target - n_play_want

    n_play = min(n_play_want, len(play_items))
    n_other = min(n_other_want, len(other_items))

    # deficit 按剩余容量比例补齐（不偏向任何一边）
    deficit = target - n_play - n_other
    if deficit > 0:
        play_left = len(play_items) - n_play
        other_left = len(other_items) - n_other
        total_left = play_left + other_left

        if total_left > 0:
            from_play = min(round(deficit * play_left / total_left), play_left)
            from_other = min(deficit - from_play, other_left)
            n_play += from_play
            n_other += from_other

    sampled_play = rng.sample(range(len(play_items)), n_play) if n_play > 0 else []
    sampled_other = rng.sample(range(len(other_items)), n_other) if n_other > 0 else []
    sampled = [play_items[i] for i in sampled_play] + [other_items[i] for i in sampled_other]
    need = min(target, len(pool)) - len(sampled)
    if need > 0:
        used_shas = {s[0] for s in sampled}  # sha256 天然唯一
        remaining = [x for x in pool if x[0] not in used_shas]
        if remaining:
            sampled += rng.sample(remaining, min(need, len(remaining)))

    return sampled

# ================================================================
#  采样（保留 pkg 字段）
# ================================================================

def sample_split(benign, malware, years, target_size, cfg, seed):
    """
    返回: [(sha256, label, year, pkg_name, market_tag), ...]
    """
    rng = random.Random(seed)
    half = target_size // 2

    # 恶意
    m_pools = {y: len(malware.get(y, [])) for y in years}
    m_quotas = largest_remainder_alloc(m_pools, half)

    malware_sampled = []
    for y in years:
        pool = malware.get(y, [])
        q = m_quotas.get(y, 0)
        if q > 0 and pool:
            chosen = rng.sample(pool, q)
            malware_sampled.extend(
                [(sha, 1, y, pkg, market) for sha, pkg, market, dex_dt in chosen]
            )

    # 良性
    b_pools = {y: len(benign.get(y, [])) for y in years}
    b_quotas = largest_remainder_alloc(b_pools, half)

    benign_sampled = []
    for y in years:
        pool = benign.get(y, [])
        q = b_quotas.get(y, 0)
        if q > 0 and pool:
            if cfg["benign_mode"] == "mixed":
                chosen = sample_benign_mixed(pool, q, cfg["benign_play_ratio"], rng)
            else:
                play_only = [x for x in pool if x[2] == "play"]
                chosen = rng.sample(play_only, min(q, len(play_only)))
            benign_sampled.extend(
                [(sha, 0, y, pkg, market) for sha, pkg, market, dex_dt in chosen]
            )

    # 配平
    actual_half = min(len(benign_sampled), len(malware_sampled))
    if actual_half < half:
        print(f"  ⚠️  目标 {half}x2, 实际 {actual_half}x2 "
              f"(良性 {len(benign_sampled)}, 恶意 {len(malware_sampled)})")

    rng.shuffle(benign_sampled)
    rng.shuffle(malware_sampled)
    samples = benign_sampled[:actual_half] + malware_sampled[:actual_half]
    rng.shuffle(samples)

    return samples


# ================================================================
#  Bias check（读 cfg 而非全局 CONFIG）
# ================================================================

def bias_check(samples, split_name, cfg):
    benign_sources = defaultdict(int)
    malware_count = 0
    benign_count = 0

    for sha, label, year, pkg, market in samples:
        if label == 0:
            benign_count += 1
            benign_sources[market] += 1
        else:
            malware_count += 1

    print(f"\n[Bias Check] {split_name}:")
    print(f"  恶意: {malware_count}, 良性: {benign_count}")
    if benign_count > 0:
        for source in sorted(benign_sources):
            count = benign_sources[source]
            pct = 100 * count / benign_count
            print(f"  良性 '{source}': {count} ({pct:.1f}%)")

        play_pct = benign_sources.get("play", 0) / benign_count
        target_pct = cfg["benign_play_ratio"]
        drift = abs(play_pct - target_pct)
        if drift > 0.1:
            print(f"  ⚠️  play 占比 {play_pct:.1%} 偏离目标 {target_pct:.0%} 超过 10%")
        else:
            print(f"  ✅ play 占比 {play_pct:.1%}, 接近目标 {target_pct:.0%}")


# ================================================================
#  保存（含 pkg_name + market）
# ================================================================

def save_split(samples, filepath):
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "label", "year", "pkg_name", "market"])
        for sha, label, year, pkg, market in samples:
            writer.writerow([sha, label, year, pkg, market])


def save_metadata(cfg, splits):
    meta = {
        "seed": cfg["seed"],
        "seed_offsets": SEED_OFFSETS,
        "vt_config": cfg["active_vt_config"],
        "vt_params": cfg["vt_configs"][cfg["active_vt_config"]],
        "one_version_per_pkg": cfg["one_version_per_pkg"],
        "benign_mode": cfg["benign_mode"],
        "benign_play_ratio": cfg["benign_play_ratio"],
        "train_years": cfg["train_years"],
        "val_years": cfg["val_years"],
        "test_years": cfg["test_years"],
        "apk_size_range": [cfg["min_apk_size"], cfg["max_apk_size"]],
        "stats": {name: len(data) for name, data in splits.items()},
    }
    path = os.path.join(cfg["output_dir"], "split_metadata.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n元信息: {path}")


# ================================================================
#  主流程
# ================================================================

def run_split(vt_config_name="strict"):
    cfg = copy.deepcopy(CONFIG)
    cfg["active_vt_config"] = vt_config_name

    suffix = f"_{vt_config_name}" if vt_config_name != "strict" else ""
    out_dir = cfg["output_dir"]
    cfg["output_dir"] = f"{out_dir}{suffix}"
    # cfg["output_dir"] = f"dataset_split{suffix}"
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  VT: {vt_config_name} | pkg 单版本: {cfg['one_version_per_pkg']}")
    print(f"  输出: {cfg['output_dir']}")
    print(f"{'='*50}")

    # 1. 扫描
    benign, malware = scan_and_filter(cfg)

    # 2. 去泄漏
    benign, malware = deduplicate_by_pkg(benign, malware, cfg)

    # 3. 采样
    split_configs = [
        ("train", cfg["train_years"], cfg["train_size"]),
        ("val",   cfg["val_years"],   cfg["val_size"]),
        ("test",  cfg["test_years"],  cfg["test_size"]),
    ]

    splits = {}
    for name, years, size in split_configs:
        print(f"\n--- {name} ---")
        seed = cfg["seed"] + SEED_OFFSETS[name]
        data = sample_split(benign, malware, years, size, cfg, seed)
        splits[name] = data
        print(f"  结果: {len(data)} 条")

    # 4. sha256 交叉检查
    sha_sets = {n: {s[0] for s in d} for n, d in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sha_sets[a] & sha_sets[b]
        assert len(overlap) == 0, f"{a}/{b} sha256 交叉 {len(overlap)} 条!"
    print("\n✅ sha256 无交叉")

    # 5. pkg_name 交叉检查（额外验证）
    pkg_sets = {n: {s[3] for s in d} for n, d in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = pkg_sets[a] & pkg_sets[b]
        if overlap:
            # 过滤掉 __unknown_ 开头的
            real_overlap = {p for p in overlap if not p.startswith("__unknown_")}
            if real_overlap:
                print(f"  ⚠️  {a}/{b} pkg_name 交叉 {len(real_overlap)} 个!")
            else:
                print(f"  ✅ {a}/{b} pkg_name 无交叉（{len(overlap)} 个 unknown 忽略）")
        else:
            print(f"  ✅ {a}/{b} pkg_name 无交叉")

    # 6. Bias check
    for name, data in splits.items():
        bias_check(data, name, cfg)

    # 7. 年份分布
    all_years = sorted(set(cfg["train_years"] + cfg["val_years"] + cfg["test_years"]))
    print(f"\n{'':>8} ", end="")
    for y in all_years:
        print(f"{y:>8}", end="")
    print(f"{'总计':>8}")
    print("-" * (10 + 8 * (len(all_years) + 1)))

    for name in ["train", "val", "test"]:
        data = splits[name]
        year_counts = defaultdict(int)
        for _, _, year, _, _ in data:
            year_counts[year] += 1
        print(f"{name:>8} ", end="")
        for y in all_years:
            print(f"{year_counts.get(y, 0):>8}", end="")
        print(f"{len(data):>8}")

    # 8. 保存
    for name, data in splits.items():
        save_split(data, os.path.join(cfg["output_dir"], f"{name}.csv"))
    save_metadata(cfg, splits)

    print(f"\n✅ {cfg['output_dir']}/ 完成")


def main():
    # run_split("strict")
    print("\n\n")
    run_split("relaxed")

    print(f"\n{'='*50}")
    print("  dataset_split/         <- 主实验 (vt >= 10)")
    print("  dataset_split_relaxed/ <- 附录   (vt >= 4)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()