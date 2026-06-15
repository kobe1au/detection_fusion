#!/usr/bin/env python3
"""
AndroZoo APK 批量下载器（按 split 分组 + 每组一个 tqdm 进度条）
- 断点续传（progress_file 每行一个 sha256）
- 并发下载（ThreadPoolExecutor）
- 429 读取 Retry-After
- 403/401/400 打印响应片段
- zip magic 检查，避免写入 HTML/错误页
- 可选 sha256 校验
"""

import csv
import os
import sys
import time
import hashlib
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

_thread_local = threading.local()

def get_session():
    if getattr(_thread_local, "session", None) is None:
        s = requests.Session()
        # 连接池调大，适配并发
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=0)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "androzoo-downloader/1.1", "Connection": "keep-alive"})
        _thread_local.session = s
    return _thread_local.session

# ================================================================
#  配置
# ================================================================

CONFIG = {
    # 从环境变量读取，避免泄露
    "api_key": os.environ.get("ANDROZOO_API_KEY", ""),

    # split CSV 所在目录
   "split_dir": r"D:/code/detection",

    #APK 下载到哪里（会自动建立 train/val/test 子目录）
   "output_dir": r"D:/src",

    #断点续传：已完成 sha 记录文件（每行一个 sha256）
    "progress_file": r"D:/src/download_progress.txt",

    # split CSV 所在目录
    # "split_dir": r"/root/autodl-tmp/dataset/dataset_split_relaxed",

    # # APK 下载到哪里（会自动建立 train/val/test 子目录）
    # "output_dir": r"/root/autodl-tmp/dataset/apks_relaxed",

    # # 断点续传：已完成 sha 记录文件（每行一个 sha256）
    # "progress_file": r"/root/autodl-tmp/dataset/apks_relaxed/download_progress.txt",

    # 要下载哪些 split（可选 train / val / test）
    "splits": ["train"],

    # 并发下载线程数（AndroZoo 限流，别太高）
    "max_workers": 8,

    # 单个文件下载超时（秒）
    "timeout": 300,

    # 失败重试次数
    "max_retries": 3,

    # 重试间隔（秒）
    "retry_delay": 5,

    # 是否校验 sha256
    "verify_sha256": True,
}

ANDROZOO_URL = "https://androzoo.uni.lu/api/download"

_progress_lock = threading.Lock()


# ================================================================
#  工具函数
# ================================================================
_progress_buffer = []
_progress_flush_every = 50  # 可调

def append_progress(progress_file: str, sha256: str) -> None:
    global _progress_buffer
    with _progress_lock:
        _progress_buffer.append(sha256)
        if len(_progress_buffer) >= _progress_flush_every:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write("\n".join(_progress_buffer) + "\n")
            _progress_buffer.clear()

def flush_progress(progress_file: str) -> None:
    global _progress_buffer
    with _progress_lock:
        if _progress_buffer:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write("\n".join(_progress_buffer) + "\n")
            _progress_buffer.clear()

def sha256_of_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def load_progress(progress_file: str) -> set[str]:
    done = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                sha = line.strip()
                if sha:
                    done.add(sha.lower())
    return done


def load_sha_list_by_split(split_dir: str, splits: list[str]) -> dict[str, list[dict]]:
    """
    返回：
      {
        "train": [{"sha256":..., "label":..., "split":"train"}, ...],
        "val":   [...],
        "test":  [...]
      }
    同时在全局去重（同一个 sha 不会在多个 split 重复提交）
    """
    by_split: dict[str, list[dict]] = {s: [] for s in splits}
    seen = set()

    for split_name in splits:
        csv_path = os.path.join(split_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  {csv_path} 不存在，跳过")
            continue

        count = 0
        with open(csv_path, "r",  encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
               # print(row)
                sha = row["sha256"].strip().lower()
                if not sha or sha in seen:
                    continue
                seen.add(sha)
                by_split[split_name].append({
                    "sha256": sha,
                    "split": split_name,
                    "label": int(row.get("label", -1)),
                })
                count += 1
        print(f"  {split_name}: {count} 个 sha256")

    total = sum(len(v) for v in by_split.values())
    print(f"  去重后总计: {total} 个待下载")
    return by_split


# ================================================================
#  下载单个 APK
# ================================================================

def download_one(
    sha256: str,
    output_dir: str,
    api_key: str,
    timeout: int,
    max_retries: int,
    retry_delay: int,
    verify: bool,
) -> tuple[str, str, str]:
    out_path = os.path.join(output_dir, f"{sha256}.apk")
    tmp_path = out_path + ".tmp"

    # 已存在：直接跳过（不 hash）
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return (sha256, "skip", "exists")
    elif os.path.exists(out_path):
        # size==0 的坏文件
        try:
            os.remove(out_path)
        except OSError:
            pass

    # 清理残留 tmp
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    session = get_session()

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(
                ANDROZOO_URL,
                params={"apikey": api_key, "sha256": sha256},
                timeout=timeout,
                stream=True,
            ) as resp:

                # 429：按 Retry-After 等待（并确保 resp 被关闭）
                if resp.status_code == 429:
                    ra = resp.headers.get("Retry-After")
                    wait = int(ra) if (ra and ra.isdigit()) else (retry_delay * attempt * 2)
                    time.sleep(wait)
                    continue

                if resp.status_code in (400, 401, 403):
                    txt = resp.text[:200].replace("\n", " ")
                    return (sha256, "fail", f"{resp.status_code} {txt}")

                if resp.status_code == 404:
                    return (sha256, "fail", "404 Not Found")

                resp.raise_for_status()

                # 下载时边写边 hash（只对新下载校验）
                h = hashlib.sha256() if verify else None
                first4 = b""

                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):  # 4MB
                        if not chunk:
                            continue
                        if not first4:
                            first4 = chunk[:4]
                        f.write(chunk)
                        if h is not None:
                            h.update(chunk)

                # magic 检查
                if first4 != b"PK\x03\x04":
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                    return (sha256, "fail", "not an APK (zip magic mismatch)")

                # sha256 校验
                if verify:
                    actual = h.hexdigest().lower()
                    if actual != sha256:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        if attempt < max_retries:
                            time.sleep(retry_delay)
                            continue
                        return (sha256, "fail", f"sha256 mismatch: {actual[:12]}...")

                shutil.move(tmp_path, out_path)
                size_mb = os.path.getsize(out_path) / 1024 / 1024
                return (sha256, "ok", f"{size_mb:.1f}MB")

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return (sha256, "fail", "timeout")

        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            return (sha256, "fail", str(e)[:120])

    return (sha256, "fail", "retry exhausted")


# ================================================================
#  按 split 下载（每组一个 tqdm）
# ================================================================

def download_split(split_name: str, items: list[dict], cfg: dict, done: set[str]) -> list[dict]:
    """
    下载某个 split 的所有样本，返回该 split 的失败列表
    """
    out_dir = os.path.join(cfg["output_dir"], split_name)
    os.makedirs(out_dir, exist_ok=True)

    # 过滤已完成
    todo = [x for x in items if x["sha256"] not in done]
    total = len(todo)
    if total == 0:
        print(f"\n✅ {split_name}: 全部已完成（0 待下载）")
        return []

    stats = defaultdict(int)
    failed: list[dict] = []

    start = time.time()
    pbar = tqdm(total=total, desc=f"{split_name}", unit="apk", dynamic_ncols=True)

    with ThreadPoolExecutor(max_workers=cfg["max_workers"]) as executor:
        futures = {
            executor.submit(
                download_one,
                x["sha256"],
                out_dir,
                cfg["api_key"],
                cfg["timeout"],
                cfg["max_retries"],
                cfg["retry_delay"],
                cfg["verify_sha256"],
            ): x
            for x in todo
        }

        completed = 0
        for fut in as_completed(futures):
            completed += 1
            sha, status, msg = fut.result()
            stats[status] += 1

            if status in ("ok", "skip"):
                append_progress(cfg["progress_file"], sha)
                done.add(sha)  # 让同一轮里其它 split（如重复）也能跳过
            else:
                failed.append({"sha256": sha, "split": split_name, "reason": msg})

            # tqdm 更新
            elapsed = time.time() - start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0

            pbar.update(1)
            pbar.set_postfix({
                "ok": stats["ok"],
                "skip": stats["skip"],
                "fail": stats["fail"],
                "eta_min": int(eta / 60),
                "last": status,
            })

            # 失败就打印一条
            if status == "fail":
                tqdm.write(f"[{split_name}] fail {sha[:12]}... | {msg}")

    pbar.close()

    print(f"✅ {split_name}: ok={stats['ok']} skip={stats['skip']} fail={stats['fail']}")
    return failed


def batch_download(cfg: dict) -> None:
    print("=" * 50)
    print("  AndroZoo APK 批量下载器（split 分组 + tqdm）")
    print("=" * 50)

    if not cfg["api_key"]:
        print("\n❌ 请先设置环境变量 ANDROZOO_API_KEY")
        print("   PowerShell: $env:ANDROZOO_API_KEY='你的key'")
        print("   CMD: set ANDROZOO_API_KEY=你的key")
        sys.exit(1)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["progress_file"]), exist_ok=True)

    print("\n加载 split CSV...")
    by_split = load_sha_list_by_split(cfg["split_dir"], cfg["splits"])
    total = sum(len(v) for v in by_split.values())
    if total == 0:
        print("没有需要下载的样本")
        return

    print("\n加载断点续传进度...")
    done = load_progress(cfg["progress_file"])
    print(f"  已完成记录: {len(done)}")

    all_failed: list[dict] = []
    for split_name in cfg["splits"]:
        items = by_split.get(split_name, [])
        if not items:
            continue
        failed = download_split(split_name, items, cfg, done)
        all_failed.extend(failed)

    # 保存失败列表
    if all_failed:
        fail_path = os.path.join(cfg["output_dir"], "download_failed.csv")
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sha256", "split", "reason"])
            writer.writeheader()
            writer.writerows(all_failed)
        print(f"\n⚠️  失败列表: {fail_path} ({len(all_failed)} 条)")
    else:
        print("\n✅ 全部下载完成，无失败项")

    flush_progress(cfg["progress_file"])


if __name__ == "__main__":
    batch_download(CONFIG)
