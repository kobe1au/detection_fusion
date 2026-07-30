class ArchitectureConstants:
    HEAD_HIDDEN_DIMS = [256, 128]
    HEAD_DROPOUT_RATES = [0.3, 0.2]

    GATE_HIDDEN_DIM = 128


class GateConstants:
    EPS = 1e-8


class AvailabilityIndex:
    """Exact model-facing fusion input: one hard availability bit per branch."""

    API_ALIVE = 0
    GRAPH_ALIVE = 1
    MANIFEST_ALIVE = 2

    BASE_DIM = 3


VALIDATION_HOLDOUT_FRACTION = 0.60

class TriModalConfigDefaults:
    """Shared implementation defaults, not the identity of the paper method.

    Runnable experiments must select an explicit method YAML.  In particular,
    CARE-Droid has a separate closed configuration and lifecycle in
    ``fusion.care_train``. Keeping shared architecture and comparison-method
    defaults here avoids duplicating them across the experiment catalogue
    without making this mapping an implicit paper method.
    """

    CONFIG = {
        "data": {
            "max_failed_ratio": 0.0,
            "max_api_events_per_sample": 2048,
            "strict_split_integrity": True,
            "strict_partition_isolation": True,
            "allow_pt_superset": True,
        },
        "train": {
            "deterministic": True,
            "strict_deterministic": False,
            "device": "auto",
            "pin_memory": False,
            "allow_pyg_pin_memory": False,
            "persistent_workers": True,
            "use_amp": True,
            "eta_min": 0.000001,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "grad_accum_steps": 1,
            "label_smoothing": 0.0,
        },
        # Comparison-method identity. Current comparison runs always train
        # their own model; no cross-method checkpoint reuse is supported.
        "encoder_stage": {
            "mode": "fit",
            "protocol_id": "comparison_method_specific_stage1_v1",
            "strict_identity": True,
        },
        "model": {
            "num_classes": 2,
            "fusion_mode": "discount_probability",
            "max_nodes_gnn": 12288,
            "api_encoder": {
                "type": "transformer",
                "num_hash_buckets": 8192,
                "type_vocab_size": 16,
                "emb_dim": 128,
                "hidden_dim": 256,
                "dropout": 0.15,
                "layers": 2,
                "heads": 4,
                "max_seq_len": 2048,
            },
            "graph_encoder": {
                "type": "gatv2",
                "emb_dim": 128,
                "hidden": 128,
                "heads": 4,
                "layers": 2,
                "use_behavior_hint": False,
                "drop_extracted_behavior_hints": True,
            },
            "manifest_encoder": {
                "in_dim": 256,
                "emb_dim": 128,
                "hidden_dim": 256,
                "dropout": 0.1,
                "category_dim": 12,
                "stats_dim": 11,
                "permission_dim": 128,
                "intent_dim": 64,
                "feature_dim": 32,
            },
            "gate": {
                "hidden_dim": 128,
                "detach": True,
            },
        },
        "fusion": {
            "mode": "discount_probability",
            "evidence_activation": "softplus",
            "use_hard_alive_mask": True,
            "force_fp32_decision": True,
            "min_discount": 1.0e-6,
        },
        "loss": {
            "branch_aux_weight": 0.25,
            "evidential_loss_weight": 0.05,
            "branch_aux_weights": {
                "api": 1.0,
                "graph": 1.0,
                "manifest": 1.0,
            },
            # Clean Stage-1 supervises every available branch equally.
            "auxiliary_weight_mode": "alive_masked_uniform",
            "evidential": {
                "anneal_epochs": 10,
                "branches": ["api", "graph", "manifest"],
                "class_weight": "balanced",
            },
        },
        "calibration": {
            "validation_fraction": VALIDATION_HOLDOUT_FRACTION,
            "split_seed": 42,
            "stratified_group_split": True,
        },
        "classification_threshold": {
            "enabled": False,
        },
        "eval": {
            "run_test": True,
            "run_robust_test": True,
            "perturb_strengths": [0.1, 0.3, 0.5, 0.7, 0.9],
            "perturb_tests": [
                "clean",
                # Canonical five-point controlled-degradation curves.
                "api_event_dropout",
                "graph_sparsify",
                "manifest_permission_mask",
                "api_missing",
                "graph_missing",
                "manifest_missing",
            ],
        },
    }
