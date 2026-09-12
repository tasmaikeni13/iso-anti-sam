"""
Search Space and Protocol Definitions for Hyperparameter Sweeps across
AdamW, Vanilla SAM, and CARVE on FineWeb-Edu.
"""

SWEEP_CONFIG_125M = {
    "model_scale": "125M",
    "total_tokens": 3_000_000_000,
    "seq_len": 2048,
    "global_batch_size_tokens": 524_288, # 256 sequences * 2048
    "gradient_accumulation_steps": 64,  # micro-batch 4 * 64 on 1 GPU, or 8 on 8 GPUs
    "micro_batch_size": 4,
    "warmup_ratio": 0.02,
    "max_steps": 5722,                  # 3B / 524,288
    "optimizers": {
        "adamw": {
            "lr": [3e-4, 6e-4, 1e-3],
            "weight_decay": [0.01, 0.1],
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "fused": True
        },
        "sam": {
            "lr": [3e-4, 6e-4],
            "rho": [0.01, 0.02, 0.05, 0.10],
            "weight_decay": 0.01,
            "adaptive": [False, True],
            "base_optimizer": "adamw"
        },
        "carve": {
            "lr": [3e-4, 6e-4, 1e-3],
            "rho_0": [0.005, 0.01, 0.02, 0.05],
            "coherence_floor": [0.0, 0.02],
            "schedule": ["cosine_decay", "constant"],
            "weight_decay": 0.01,
            "base_optimizer": "adamw"
        }
    }
}

SWEEP_CONFIG_350M = {
    "model_scale": "350M",
    "total_tokens": 7_000_000_000,
    "seq_len": 2048,
    "global_batch_size_tokens": 1_048_576, # 512 sequences * 2048
    "gradient_accumulation_steps": 128,    # micro-batch 4 * 128 on 1 GPU, or 16 on 8 GPUs
    "micro_batch_size": 4,
    "warmup_ratio": 0.02,
    "max_steps": 6675,                     # 7B / 1,048,576
    "optimizers": {
        "adamw": {
            "lr": [2e-4, 4e-4, 8e-4],
            "weight_decay": [0.01, 0.1],
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "fused": True
        },
        "sam": {
            "lr": [2e-4, 4e-4],
            "rho": [0.01, 0.02, 0.05],
            "weight_decay": 0.01,
            "adaptive": False,
            "base_optimizer": "adamw"
        },
        "carve": {
            "lr": [2e-4, 4e-4, 8e-4],
            "rho_0": [0.005, 0.01, 0.02, 0.04],
            "coherence_floor": [0.0, 0.01],
            "schedule": ["cosine_decay", "constant"],
            "weight_decay": 0.01,
            "base_optimizer": "adamw"
        }
    }
}
