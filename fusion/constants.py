class ArchitectureConstants:
    HEAD_HIDDEN_DIMS = [256, 128]
    HEAD_DROPOUT_RATES = [0.3, 0.2]
    HEAD_DROPOUT = 0.2

    GATE_HIDDEN_DIM = 128
    GATE_INIT_BIAS = 0.0
    GATE_JOINT_INIT_BIAS = 0.5

    MODALITY_ALIVE_THRESHOLD = 0.01


class QualityConstants:
    API_COUNT_NORM = 128.0
    API_DIVERSITY_SCALE = 2.0
    API_COUNT_WEIGHT = 0.35
    API_DIVERSITY_WEIGHT = 0.25
    API_COVERAGE_WEIGHT = 0.25
    API_TYPE_WEIGHT = 0.15

    GRAPH_NODE_NORM = 32.0
    GRAPH_NODE_WEIGHT = 0.35
    GRAPH_EDGE_WEIGHT = 0.35
    GRAPH_FEATURE_WEIGHT = 0.30

    ALIGN_NODE_COVER_WEIGHT = 0.5
    ALIGN_API_COVER_WEIGHT = 0.5

class GateConstants:
    NUM_BRANCHES = 4
    UNIFORM_BRANCH_WEIGHT = 0.25
    EPS = 1e-8


class EvidenceIndex:
    API_INTEGRITY = 0
    GRAPH_INTEGRITY = 1
    MANIFEST_INTEGRITY = 2
    CODE_INTEGRITY = 3
    API_GRAPH_ANCHOR_SUPPORT = 4
    MANIFEST_CODE_SUPPORT = 5
    MANIFEST_TO_CODE_CONFLICT = 6
    CODE_TO_MANIFEST_CONFLICT = 7
    API_ALIVE = 8
    GRAPH_ALIVE = 9
    MANIFEST_ALIVE = 10
    API_ENCODER_COVERAGE = 11
    GRAPH_ENCODER_COVERAGE = 12

    BASE_DIM = 13

    # Compatibility aliases for old gate/loss code. They point only to
    # observable integrity/support fields and never to synthetic pert_*.
    R_API = API_INTEGRITY
    R_GRAPH = GRAPH_INTEGRITY
    R_MANIFEST = MANIFEST_INTEGRITY
    Q_ALIGN = API_GRAPH_ANCHOR_SUPPORT


class TriModalConfigDefaults:
    """Stable defaults for the lean observable-reliability pipeline.

    YAML experiment files should contain paths, experiment names, and the few
    mechanism switches being studied. Architecture defaults and invariant
    safety settings live here so the paper configuration does not look like a
    pile of unrelated knobs.
    """

    CONFIG = {
        "data": {
            "max_failed_ratio": 0.0,
            "max_api_events_per_sample": 2048,
            "graph_semantic_source": "alignment",
            "strict_split_integrity": True,
            "strict_partition_isolation": True,
            "allow_pt_superset": True,
        },
        "train": {
            "tuning_mode": False,
            "deterministic": True,
            "strict_deterministic": False,
            "device": "auto",
            "min_delta": 0.0001,
            "pin_memory": False,
            "allow_pyg_pin_memory": False,
            "persistent_workers": True,
            "use_amp": True,
            "eta_min": 0.000001,
            "weight_decay": 0.01,
            "grad_clip": 1.0,
            "grad_accum_steps": 1,
            "label_smoothing": 0.0,
            "checkpoint_metric": "clean_macro_f1",
        },
        "model": {
            "num_classes": 2,
            "fusion_mode": "discount_probability",
            "joint_emb_dim": 128,
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
                "account_for_encoder_budget": True,
            },
            "manifest_encoder": {
                "enabled": True,
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
                "use_consistency_evidence": True,
                "use_conflict_evidence": True,
                "use_perturbation_evidence": False,
                "apply_alive_mask": True,
            },
        },
        "fusion": {
            "mode": "discount_probability",
            "detach_discount": True,
            "detach_confidence_proxy": True,
            "use_confidence_proxy": True,
            "use_reliability_discount": True,
            "reliability_discount_exponent": 0.5,
            "branch_competence_prior": {
                "enabled": True,
                "metric": "macro_f1",
                "normalization": "best",
                "min_value": 0.5,
            },
            "visible_integrity_modifier": {
                "enabled": True,
                "beta": 1.0,
                "min_value": 0.5,
                "min_reference": 1.0e-6,
            },
            "weight_sharpening_gamma": 1.0,
            "use_support_discount": True,
            "use_conflict_discount": True,
            "use_hard_alive_mask": True,
            "acceptance_aggregation": "product",
            "force_fp32_decision": True,
            "min_discount": 1.0e-6,
            "fallback": "uniform",
            "confidence_proxy": {
                "type": "entropy_margin",
                "temperature_api": 1.0,
                "temperature_graph": 1.0,
                "temperature_manifest": 1.0,
                "temperature_joint": 1.0,
            },
            "reliability_calibration": {
                "enabled": True,
                "hidden_dim": 16,
                "missing_relation_support": 0.0,
                "use_relation_evidence": True,
                "apply_alive_mask": True,
                "loss": "bce",
                "weight": 1.0,
            },
            "probability_calibration": {
                "enabled": True,
                "weight": 1.0,
            },
            "support_factor": {
                "manifest_support_base": 0.5,
                "code_anchor_base": 0.5,
            },
            "conflict_factor": {
                "min_value": 0.05,
            },
        },
        "loss": {
            "branch_aux_weight": 0.25,
            "reliability_weighted_aux": True,
            "min_aux_weight": 0.2,
            "detach_reliability_for_aux": True,
            "reliability_calibration_weight": 0.0,
            "probability_calibration_weight": 0.0,
        },
        "calibration": {
            "enabled": True,
            "validation_fraction": 0.5,
            "split_seed": 42,
            "stratified_group_split": True,
            "epochs": 20,
            "patience": 4,
            "min_delta": 0.00001,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
            "include_robust_val": False,
        },
        "selective_prediction": {
            "enabled": True,
            "target_coverage": 0.95,
        },
        "robust": {
            "train_aug": True,
            "perturb_prob": 0.5,
            "perturb_strengths": [0.1, 0.3, 0.5],
        },
        "eval": {
            "run_test": True,
            "run_robust_test": True,
            "robust_val": {"enabled": False},
            "perturb_strengths": [0.1, 0.3, 0.5, 0.7, 0.9],
            "perturb_tests": [
                "clean",
                "api_degraded",
                "graph_degraded",
                "api_graph_degraded",
                "manifest_degraded",
                "all_degraded",
                "api_missing",
                "graph_missing",
                "manifest_missing",
            ],
        },
    }
