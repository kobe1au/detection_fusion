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


# Legacy three-role split defaults retained only for comparison methods that
# still exercise the generic post-hoc implementation.  The proposed
# competence-anchored method does not consume these values: it requires the
# fixed schema-v2 75/25 model-selection/decision-calibration assignment.
VALIDATION_HOLDOUT_FRACTION = 0.60
CONFORMAL_WITHIN_HOLDOUT_FRACTION = 5.0 / 12.0

class TriModalConfigDefaults:
    """Shared implementation defaults, not the identity of the paper method.

    Runnable experiments must select an explicit method YAML.  In particular,
    ``competence_anchored_fusion.yaml`` replaces the legacy fusion,
    calibration, loss, and validation sections below with its closed schema.
    Keeping shared architecture and comparison-method defaults here avoids
    duplicating them across the experiment catalogue without making this
    mapping an implicit "main method".
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
        },
        # Stage-1 is a separately versioned, reusable artifact.  Post-hoc I1,
        # I2 and I3 changes must not silently retrain or mutate the encoders.
        "encoder_stage": {
            "mode": "fit",
            "protocol_id": "neutral_alive_uniform_clean_stage1_v2",
            "checkpoint_path": None,
            "expected_sha256": None,
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
            },
        },
        "fusion": {
            "mode": "discount_probability",
            "evidence_activation": "softplus",
            "use_i1_reliability": True,
            "use_hard_alive_mask": True,
            "force_fp32_decision": True,
            "min_discount": 1.0e-6,
            "reliability_calibration": {
                "enabled": True,
                "method": "monotonic_correctness",
                "use_evidential_certainty": True,
                "use_prediction_margin": True,
                "use_api_observed_support": True,
                "use_predicted_class_intercept": True,
                "scenario_objective_weights": {
                    "clean": 0.50,
                    "perturb": 0.50,
                },
                "loss": "bce",
            },
            "routing": {
                "enabled": False,
                "mode": "learned",
                "posthoc_refine": True,
                "prediction_loss_weight": 1.0,
                "risk_conflict_enabled": True,
                "risk_mode": "learned",
                "risk_loss_weight": 1.0,
                "risk_loss": "bce",
                "initial_risk": 0.10,
                "final_temperature_scaling": False,
            },
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
            "enabled": True,
            "validation_fraction": VALIDATION_HOLDOUT_FRACTION,
            "conformal_fraction": CONFORMAL_WITHIN_HOLDOUT_FRACTION,
            "split_seed": 42,
            "stratified_group_split": True,
            # Post-hoc I1/I2 fitting uses a compact, explicitly declared set
            # of representative mechanisms. Evaluation owns its independent
            # five-strength stress suite below.
            "fit_perturbations": [
                "api_event_dropout",
                "graph_sparsify",
                "manifest_permission_mask",
            ],
            "perturb_strengths": [0.3, 0.5, 0.7],
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
        "eval": {
            "run_test": True,
            "run_robust_test": True,
            "perturb_strengths": [0.1, 0.3, 0.5, 0.7, 0.9],
            "perturb_tests": [
                "clean",
                # Canonical five-point curves. The middle three strengths are
                # used by post-hoc fitting; 0.1 and 0.9 audit extrapolation.
                "api_event_dropout",
                "graph_sparsify",
                "manifest_permission_mask",
                "api_missing",
                "graph_missing",
                "manifest_missing",
            ],
        },
    }
