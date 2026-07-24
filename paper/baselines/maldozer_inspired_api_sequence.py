from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from fusion.constants import VALIDATION_HOLDOUT_FRACTION
from paper.baselines.common import (
    binary_metrics,
    concat_long_from_sources,
    enforce_formal_split_completeness,
    load_pt,
    read_label_csv,
    set_reproducible_seed,
    validation_selection_indices,
    write_json,
    write_predictions,
)


class ApiSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[torch.Tensor],
        labels: np.ndarray,
        sha256: list[str],
    ) -> None:
        self.sequences = sequences
        self.labels = labels.astype(np.int64)
        self.sha256 = sha256

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx], int(self.labels[idx])


def load_api_sequences(
    pt_dir: str | Path,
    csv_path: str | Path,
    *,
    max_len: int,
    vocab_size: int,
    show_progress: bool = True,
) -> tuple[ApiSequenceDataset, Any, list[dict[str, str]]]:
    frame = read_label_csv(csv_path)
    sequences: list[torch.Tensor] = []
    labels: list[int] = []
    kept_rows = []
    failures: list[dict[str, str]] = []
    max_token = int(vocab_size) - 1
    rows = list(frame.itertuples(index=False))
    iterator = tqdm(
        rows,
        desc=f"load {Path(csv_path).stem}",
        unit="apk",
        leave=False,
        disable=not show_progress,
    )
    for row in iterator:
        sha = str(getattr(row, "sha256"))
        try:
            payload = load_pt(pt_dir, sha)
            seq = concat_long_from_sources(payload, "api_ids").long()
            seq = seq[: int(max_len)]
            if seq.numel() == 0:
                # Keep an unavailable API sequence as padding-only input. It
                # must not masquerade as the real hash bucket zero token.
                seq = torch.zeros((1,), dtype=torch.long)
            else:
                seq = seq.clamp(min=0, max=max_token) + 1  # 0 is padding.
            sequences.append(seq)
            labels.append(int(getattr(row, "label")))
            kept_rows.append(row)
        except Exception as exc:  # noqa: BLE001
            failures.append({"sha256": sha, "error": f"{type(exc).__name__}: {exc}"})
    if not sequences:
        raise RuntimeError(f"No loadable samples from {csv_path}")
    kept_frame = frame[frame["sha256"].astype(str).isin([str(getattr(row, "sha256")) for row in kept_rows])].copy()
    return ApiSequenceDataset(sequences, np.asarray(labels, dtype=np.int64), [str(getattr(row, "sha256")) for row in kept_rows]), kept_frame, failures


def collate_api_batch(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.as_tensor([item[1] for item in batch], dtype=torch.long)
    max_len = max(int(item[0].numel()) for item in batch)
    x = torch.zeros((len(batch), max_len), dtype=torch.long)
    for idx, (seq, _label) in enumerate(batch):
        x[idx, : seq.numel()] = seq
    return x, labels


class ApiSequenceCNN(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        emb_dim: int = 128,
        channels: int = 128,
        kernels: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size) + 1, int(emb_dim), padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(int(emb_dim), int(channels), kernel_size=int(kernel), padding=int(kernel) // 2)
            for kernel in kernels
        )
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(channels) * len(kernels), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x).transpose(1, 2)
        pooled = []
        mask = (x != 0).unsqueeze(1)
        for conv in self.convs:
            h = F.relu(conv(emb))
            h = h.masked_fill(~mask, float("-inf"))
            h = h.amax(dim=-1)
            h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
            pooled.append(h)
        features = self.dropout(torch.cat(pooled, dim=-1))
        return self.classifier(features)


def subset_api_dataset(
    dataset: ApiSequenceDataset, indices: list[int]
) -> ApiSequenceDataset:
    index_array = np.asarray(indices, dtype=np.int64)
    return ApiSequenceDataset(
        [dataset.sequences[int(index)] for index in index_array],
        dataset.labels[index_array],
        [dataset.sha256[int(index)] for index in index_array],
    )


def predict(
    model: nn.Module,
    dataset: ApiSequenceDataset,
    *,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
    desc: str = "predict",
) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_api_batch)
    probs = []
    model.eval()
    with torch.no_grad():
        for x, _labels in tqdm(loader, desc=desc, leave=False, disable=not show_progress):
            logits = model(x.to(device))
            prob = torch.softmax(logits.float(), dim=-1)[:, 1]
            probs.append(prob.cpu().numpy())
    return np.concatenate(probs, axis=0)


def evaluate_split(
    model: nn.Module,
    pt_dir: str | Path,
    csv_path: str | Path,
    out_dir: Path,
    split_name: str,
    *,
    max_len: int,
    vocab_size: int,
    batch_size: int,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, Any]:
    dataset, frame, failures = load_api_sequences(
        pt_dir,
        csv_path,
        max_len=max_len,
        vocab_size=vocab_size,
        show_progress=show_progress,
    )
    enforce_formal_split_completeness(
        split_name,
        num_eval=len(dataset),
        failures=failures,
    )
    prob = predict(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        show_progress=show_progress,
        desc=f"predict {split_name}",
    )
    metrics = binary_metrics(dataset.labels, prob)
    metrics["num_failed"] = len(failures)
    metrics["csv"] = str(csv_path)
    metrics["pt_dir"] = str(pt_dir)
    write_predictions(out_dir / f"predictions_{split_name}.csv", frame, prob)
    if failures:
        write_json(out_dir / f"failures_{split_name}.json", {"failures": failures})
    return metrics


def train_model(
    model: nn.Module,
    train_ds: ApiSequenceDataset,
    val_ds: ApiSequenceDataset,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    show_progress: bool = True,
) -> dict[str, Any]:
    counts = np.bincount(train_ds.labels, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = torch.as_tensor(weights / weights.mean(), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_api_batch)
    best_state = None
    best_macro = -1.0
    history = []
    model.to(device)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        iterator = tqdm(loader, desc=f"train {epoch}", leave=False, disable=not show_progress)
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(y.numel())
            total_count += int(y.numel())
            iterator.set_postfix(loss=f"{float(loss.item()):.4f}")
        val_prob = predict(
            model,
            val_ds,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
            desc=f"val {epoch}",
        )
        val_metrics = binary_metrics(val_ds.labels, val_prob)
        macro = float(val_metrics["macro_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_count, 1),
                "val_macro_f1": macro,
                "val_auc": val_metrics.get("auc"),
            }
        )
        print(
            f"epoch={epoch} train_loss={history[-1]['train_loss']:.4f} "
            f"val_macro_f1={macro:.4f} val_auc={val_metrics.get('auc')}",
            flush=True,
        )
        if macro > best_macro:
            best_macro = macro
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_macro_f1": best_macro, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a MalDozer-inspired API sequence CNN baseline.")
    parser.add_argument("--train-pt-dir", required=True)
    parser.add_argument("--val-pt-dir", required=True)
    parser.add_argument("--test-pt-dir", required=True)
    parser.add_argument("--train-csv", default="labels/train.csv")
    parser.add_argument("--val-csv", default="labels/val.csv")
    parser.add_argument("--test-csv", default="labels/test.csv")
    parser.add_argument("--extra-test-csv", action="append", default=[])
    parser.add_argument("--out-dir", default="paper/outputs/maldozer_inspired")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=VALIDATION_HOLDOUT_FRACTION,
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()
    show_progress = not bool(args.no_progress)
    set_reproducible_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_ds, _train_frame, train_failures = load_api_sequences(
        args.train_pt_dir,
        args.train_csv,
        max_len=int(args.max_len),
        vocab_size=int(args.vocab_size),
        show_progress=show_progress,
    )
    val_full_ds, val_frame, val_failures = load_api_sequences(
        args.val_pt_dir,
        args.val_csv,
        max_len=int(args.max_len),
        vocab_size=int(args.vocab_size),
        show_progress=show_progress,
    )
    enforce_formal_split_completeness(
        "train",
        num_eval=len(train_ds),
        failures=train_failures,
    )
    enforce_formal_split_completeness(
        "validation",
        num_eval=len(val_full_ds),
        failures=val_failures,
    )
    selection_indices, validation_split = validation_selection_indices(
        val_frame,
        calibration_fraction=float(args.validation_fraction),
        seed=int(args.seed),
    )
    val_selection_ds = subset_api_dataset(val_full_ds, selection_indices)
    enforce_formal_split_completeness(
        "validation_selection",
        num_eval=len(val_selection_ds),
        failures=[],
    )
    model = ApiSequenceCNN(
        vocab_size=int(args.vocab_size),
        emb_dim=int(args.emb_dim),
        channels=int(args.channels),
        dropout=float(args.dropout),
    )
    train_summary = train_model(
        model,
        train_ds,
        val_selection_ds,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        device=device,
        show_progress=show_progress,
    )
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "args": vars(args),
            "train_summary": train_summary,
        },
        out_dir / "best_maldozer_inspired.pt",
    )
    model.to(device)

    summary: dict[str, Any] = {
        "method": "MalDozer-inspired API sequence CNN baseline",
        "train": {
            "num_eval": len(train_ds),
            "num_failed": len(train_failures),
            **train_summary,
            "validation_selection_size": len(val_selection_ds),
            "validation_split": {
                key: value
                for key, value in validation_split.items()
                if key not in {"selection_indices", "calibration_indices"}
            },
        },
        "validation": evaluate_split(
            model,
            args.val_pt_dir,
            args.val_csv,
            out_dir,
            "val",
            max_len=int(args.max_len),
            vocab_size=int(args.vocab_size),
            batch_size=int(args.batch_size),
            device=device,
            show_progress=show_progress,
        ),
        "test": evaluate_split(
            model,
            args.test_pt_dir,
            args.test_csv,
            out_dir,
            "test",
            max_len=int(args.max_len),
            vocab_size=int(args.vocab_size),
            batch_size=int(args.batch_size),
            device=device,
            show_progress=show_progress,
        ),
        "extra_eval": {},
        "val_num_failed": len(val_failures),
    }
    for csv_path in args.extra_test_csv:
        name = Path(csv_path).stem
        summary["extra_eval"][name] = evaluate_split(
            model,
            args.test_pt_dir,
            csv_path,
            out_dir,
            name,
            max_len=int(args.max_len),
            vocab_size=int(args.vocab_size),
            batch_size=int(args.batch_size),
            device=device,
            show_progress=show_progress,
        )
    write_json(out_dir / "summary.json", summary)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
