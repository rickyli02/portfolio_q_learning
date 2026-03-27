"""Feature builders and context masking for optional conditional inputs."""

from src.features.base_features import FeatureSpec, build_model_input, total_input_dim
from src.features.context_features import ContextBundle
from src.features.masking import (
    apply_context_mask,
    make_empty_mask,
    make_full_mask,
    random_context_dropout,
    validate_context_pair,
)

__all__ = [
    "FeatureSpec",
    "build_model_input",
    "total_input_dim",
    "ContextBundle",
    "apply_context_mask",
    "make_empty_mask",
    "make_full_mask",
    "random_context_dropout",
    "validate_context_pair",
]
