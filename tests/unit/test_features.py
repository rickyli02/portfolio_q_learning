"""Unit tests for src/features/: masking utilities, ContextBundle, and build_model_input."""

import pytest
import torch

from src.features.masking import (
    apply_context_mask,
    make_empty_mask,
    make_full_mask,
    random_context_dropout,
    validate_context_pair,
)
from src.features.context_features import ContextBundle
from src.features.base_features import FeatureSpec, build_model_input, total_input_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(batch_size: int = 4, context_dim: int = 6) -> torch.Tensor:
    return torch.randn(batch_size, context_dim)


def _make_bool_mask(batch_size: int = 4, context_dim: int = 6, fill: bool = True) -> torch.Tensor:
    return torch.full((batch_size, context_dim), fill, dtype=torch.bool)


# ---------------------------------------------------------------------------
# masking.py — validate_context_pair
# ---------------------------------------------------------------------------


def test_validate_ok():
    ctx = _make_context()
    mask = _make_bool_mask()
    validate_context_pair(ctx, mask)  # should not raise


def test_validate_shape_mismatch_raises():
    ctx = torch.randn(4, 6)
    mask = torch.ones(4, 5, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        validate_context_pair(ctx, mask)


def test_validate_wrong_dtype_raises():
    ctx = torch.randn(4, 6)
    mask = torch.ones(4, 6, dtype=torch.float32)  # not bool
    with pytest.raises(ValueError, match="dtype"):
        validate_context_pair(ctx, mask)


# ---------------------------------------------------------------------------
# masking.py — apply_context_mask
# ---------------------------------------------------------------------------


def test_apply_mask_full():
    ctx = torch.ones(3, 4)
    mask = make_full_mask((3, 4))
    out = apply_context_mask(ctx, mask)
    assert torch.allclose(out, ctx)


def test_apply_mask_empty():
    ctx = torch.ones(3, 4)
    mask = make_empty_mask((3, 4))
    out = apply_context_mask(ctx, mask)
    assert torch.all(out == 0)


def test_apply_mask_partial():
    ctx = torch.ones(2, 4)
    mask = torch.tensor([[True, False, True, False],
                          [False, True, False, True]])
    out = apply_context_mask(ctx, mask)
    expected = torch.tensor([[1., 0., 1., 0.],
                              [0., 1., 0., 1.]])
    assert torch.allclose(out, expected)


def test_apply_mask_preserves_shape():
    ctx = torch.randn(8, 10)
    mask = make_full_mask((8, 10))
    assert apply_context_mask(ctx, mask).shape == (8, 10)


# ---------------------------------------------------------------------------
# masking.py — make_full_mask / make_empty_mask
# ---------------------------------------------------------------------------


def test_make_full_mask_all_true():
    mask = make_full_mask((5, 3))
    assert mask.dtype == torch.bool
    assert mask.all()


def test_make_empty_mask_all_false():
    mask = make_empty_mask((5, 3))
    assert mask.dtype == torch.bool
    assert not mask.any()


def test_masks_correct_shape():
    shape = (4, 7)
    assert make_full_mask(shape).shape == shape
    assert make_empty_mask(shape).shape == shape


# ---------------------------------------------------------------------------
# masking.py — random_context_dropout
# ---------------------------------------------------------------------------


def test_dropout_zero_prob_unchanged():
    ctx = torch.randn(4, 8)
    mask = make_full_mask((4, 8))
    new_ctx, new_mask = random_context_dropout(ctx, mask, drop_prob=0.0)
    assert torch.allclose(new_ctx, ctx)
    assert new_mask.all()


def test_dropout_high_prob_reduces_present():
    torch.manual_seed(0)
    ctx = torch.randn(32, 16)
    mask = make_full_mask((32, 16))
    _, new_mask = random_context_dropout(ctx, mask, drop_prob=0.9)
    # Most features should now be missing
    assert new_mask.float().mean() < 0.5


def test_dropout_never_reveals_already_missing():
    ctx = torch.randn(4, 8)
    mask = make_empty_mask((4, 8))  # all missing
    new_ctx, new_mask = random_context_dropout(ctx, mask, drop_prob=0.5)
    assert not new_mask.any()
    assert torch.all(new_ctx == 0)


def test_dropout_invalid_prob_raises():
    ctx = torch.randn(2, 4)
    mask = make_full_mask((2, 4))
    with pytest.raises(ValueError, match="drop_prob"):
        random_context_dropout(ctx, mask, drop_prob=1.0)


def test_dropout_output_shape():
    ctx = torch.randn(6, 10)
    mask = make_full_mask((6, 10))
    new_ctx, new_mask = random_context_dropout(ctx, mask, drop_prob=0.3)
    assert new_ctx.shape == (6, 10)
    assert new_mask.shape == (6, 10)


def test_dropout_reproducible_with_generator():
    ctx = torch.randn(4, 8)
    mask = make_full_mask((4, 8))
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, m1 = random_context_dropout(ctx, mask, drop_prob=0.5, generator=g1)
    _, m2 = random_context_dropout(ctx, mask, drop_prob=0.5, generator=g2)
    assert torch.equal(m1, m2)


# ---------------------------------------------------------------------------
# ContextBundle
# ---------------------------------------------------------------------------


def test_bundle_all_present():
    values = torch.randn(4, 6)
    bundle = ContextBundle.all_present(values)
    assert bundle.mask.all()
    assert bundle.dim == 6
    assert bundle.batch_size == 4


def test_bundle_all_missing():
    bundle = ContextBundle.all_missing((4, 6))
    assert not bundle.mask.any()
    assert torch.all(bundle.values == 0)
    assert bundle.is_fully_missing()


def test_bundle_masked_values_zeros_missing():
    values = torch.ones(3, 4)
    mask = torch.tensor([[True, False, True, False]] * 3)
    bundle = ContextBundle(values=values, mask=mask)
    out = bundle.masked_values()
    assert out[:, 0].all()
    assert not out[:, 1].any()


def test_bundle_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape"):
        ContextBundle(
            values=torch.randn(4, 6),
            mask=torch.ones(4, 5, dtype=torch.bool),
        )


def test_bundle_is_fully_present():
    bundle = ContextBundle.all_present(torch.randn(2, 3))
    assert bundle.is_fully_present()
    assert not bundle.is_fully_missing()


def test_bundle_is_fully_missing():
    bundle = ContextBundle.all_missing((2, 3))
    assert bundle.is_fully_missing()
    assert not bundle.is_fully_present()


def test_bundle_with_dropout_shape():
    bundle = ContextBundle.all_present(torch.randn(8, 12))
    dropped = bundle.with_dropout(0.4)
    assert dropped.values.shape == (8, 12)
    assert dropped.mask.shape == (8, 12)


def test_bundle_to_device():
    bundle = ContextBundle.all_present(torch.randn(3, 4))
    cpu_bundle = bundle.to("cpu")
    assert cpu_bundle.values.device.type == "cpu"
    assert cpu_bundle.mask.device.type == "cpu"


def test_bundle_unbatched():
    values = torch.randn(5)
    bundle = ContextBundle.all_present(values)
    assert bundle.dim == 5
    assert bundle.batch_size is None


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------


def test_feature_spec_defaults():
    spec = FeatureSpec(name="wealth", dim=3)
    assert spec.optional is False
    assert spec.dim == 3


def test_feature_spec_invalid_dim_raises():
    with pytest.raises(ValueError, match="dim"):
        FeatureSpec(name="bad", dim=0)


# ---------------------------------------------------------------------------
# build_model_input
# ---------------------------------------------------------------------------


def test_build_no_context_returns_base():
    base = torch.randn(4, 8)
    out = build_model_input(base, context=None)
    assert torch.equal(out, base)


def test_build_with_full_context():
    base = torch.randn(4, 8)
    context = ContextBundle.all_present(torch.randn(4, 6))
    out = build_model_input(base, context)
    assert out.shape == (4, 14)


def test_build_with_empty_context_zeros_context_dims():
    base = torch.ones(4, 8)
    context = ContextBundle.all_missing((4, 6))
    out = build_model_input(base, context)
    assert out.shape == (4, 14)
    # Base part is untouched
    assert torch.allclose(out[:, :8], base)
    # Context part is all zeros
    assert torch.all(out[:, 8:] == 0)


def test_build_partial_context():
    base = torch.ones(2, 4)
    values = torch.ones(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    context = ContextBundle(values=values, mask=mask)
    out = build_model_input(base, context)
    assert out.shape == (2, 7)
    expected_ctx = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    assert torch.allclose(out[:, 4:], expected_ctx)


def test_build_batch_size_mismatch_raises():
    base = torch.randn(4, 8)
    context = ContextBundle.all_present(torch.randn(3, 6))  # different B
    with pytest.raises(ValueError, match="batch size"):
        build_model_input(base, context)


def test_build_unbatched():
    base = torch.randn(8)
    context = ContextBundle.all_present(torch.randn(4))
    out = build_model_input(base, context)
    assert out.shape == (12,)


# ---------------------------------------------------------------------------
# total_input_dim
# ---------------------------------------------------------------------------


def test_total_input_dim_no_context():
    spec = FeatureSpec("base", dim=8)
    assert total_input_dim(spec, None) == 8


def test_total_input_dim_with_context():
    base_spec = FeatureSpec("base", dim=8)
    ctx_spec = FeatureSpec("ctx", dim=4, optional=True)
    assert total_input_dim(base_spec, ctx_spec) == 12
