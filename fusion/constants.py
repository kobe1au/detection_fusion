class ArchitectureConstants:
    HEAD_HIDDEN_DIMS = [256, 128]
    HEAD_DROPOUT_RATES = [0.3, 0.2]

    GATE_HIDDEN_DIM = 128


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

class GateConstants:
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
            },
        },
        "fusion": {
            "mode": "discount_probability",
            "detach_discount": True,
            "detach_confidence_proxy": True,
            "use_confidence_proxy": True,
            "use_reliability_discount": True,
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
            },
            "reliability_calibration": {
                "enabled": True,
                "method": "monotonic_correctness",
                "use_model_visibility": True,
                "use_embedding_density": False,
                "embedding_density_variance_shrinkage": 0.10,
                "embedding_density_reference_quantile": 0.95,
                "embedding_density_min_class_samples": 8,
                "use_prediction_margin": True,
                "use_predicted_class_feature": True,
                "objective_weights": {
                    "clean": 0.50,
                    "completeness": 0.25,
                    "semantic": 0.25,
                },
                "require_all_objective_families": False,
                "apply_alive_mask": True,
                "loss": "bce",
                "weight": 1.0,
            },
            "routing": {
                "enabled": False,
                "mode": "learned",
                "calibration_weight": 1.0,
                "train_end_to_end": False,
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "route_oracle_loss_weight": 0.0,
                "route_oracle_temperature": 1.0,
                "subset_oracle_loss_weight": 0.0,
                "subset_oracle_temperature": 1.0,
                "group_robust_objective": {
                    "enabled": False,
                    "taxonomy": "perturb_type_v1",
                    "soft_worst_weight": 0.0,
                    "temperature": 0.1,
                    "apply_to": ["routing_distribution"],
                },
                "route_conflict_enabled": True,
                "risk_conflict_enabled": True,
                "risk_mode": "learned",
                "risk_loss_weight": 1.0,
                "risk_loss": "bce",
                "initial_risk": 0.10,
                "final_temperature_scaling": False,
                "acceptance_score_mode": "product",
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
            "branch_aux_weights": {
                "api": 1.0,
                "graph": 1.0,
                "manifest": 1.0,
            },
            # The auxiliary branch loss is weighted by observable integrity,
            # not by the fitted branch-correctness calibrator. The explicit
            # mode makes availability masking an independently testable choice.
            "auxiliary_weight_mode": "integrity",
            "min_aux_weight": 0.2,
            "detach_reliability_for_aux": True,
            "reliability_calibration_weight": 0.0,
            "probability_calibration_weight": 0.0,
            "evidential": {
                "anneal_epochs": 10,
                "evidence_activation": "softplus",
                "branches": ["api", "graph", "manifest"],
                "class_weight": "balanced",
            },
        },
        "calibration": {
            "enabled": True,
            "validation_fraction": 0.5,
            "conformal_fraction": 0.5,
            "split_seed": 42,
            "stratified_group_split": True,
            # Route-only pairwise completeness views are opt-in because they
            # add three full transformed copies per configured strength.
            "include_pairwise_completeness_views": False,
            "epochs": 30,
            "lr": 0.001,
            "weight_decay": 0.0,
            "grad_clip": 5.0,
        },
        "classification_threshold": {
            "enabled": False,
            "objective": "macro_f1",
            "selection_rule": "macro_f1_unconstrained_v1",
        },
        "selective_prediction": {
            "enabled": True,
            "mode": "conformal",
            "threshold_score": "msp",
            "class_conditional": True,
            "target_coverage": 0.90,
            "use_raw_conflict": False,
            "min_calibration_malware": 1,
            "require_feasible": False,
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
                "api_semantic_corrupted",
                "graph_semantic_corrupted",
                "manifest_semantic_corrupted",
                "all_semantic_corrupted",
                "api_missing",
                "graph_missing",
                "manifest_missing",
            ],
        },
    }
