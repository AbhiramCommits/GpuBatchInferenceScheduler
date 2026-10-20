"""pytest suite for the gpuinfer bindings.

GPU-dependent tests are skipped when no CUDA device is present.
"""

import gpuinfer
import numpy as np
import pytest

requires_gpu = pytest.mark.skipif(
    not gpuinfer.cuda_available(), reason="no CUDA GPU available"
)

gpu = pytest.mark.gpu

rng = np.random.default_rng(7)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 32, 32, 32),
        (2, 64, 48, 32),
        (3, 17, 33, 25),  # non-multiples of the 32x32 tile
        (1, 128, 64, 96),
    ],
)
@requires_gpu
@gpu
def test_gpu_batched_gemm_matches_numpy(shape):
    batch, m, n, k = shape
    a = rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32)
    got = gpuinfer.batched_gemm(a, b)
    expected = np.matmul(a, b)
    assert got.shape == (batch, m, n)
    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-4)


def _gemm_tolerance(a, b, k):
    """Worst-case float32 dot-product error scale: 8 * u * K * max|A| * max|B|.

    u is binary32 unit roundoff. Keeping the tolerance proportional to K catches
    tile/indexing bugs (O(1) error) while allowing legitimate accumulation error.
    """
    u = np.finfo(np.float32).eps / 2.0
    return 8.0 * u * k * float(np.abs(a).max()) * float(np.abs(b).max())


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (64, 1, 1, 1),
        (1, 31, 33, 1),
        (64, 1, 31, 33),
        (1, 31, 1, 257),
        (64, 31, 1, 257),
        (1, 33, 31, 1),
        (64, 33, 31, 1),
        (1, 257, 33, 31),
        (64, 257, 33, 31),
        (1, 1, 257, 1000),
        (64, 1, 1, 1000),
        (1, 1000, 1, 31),
        (64, 1000, 1, 31),
        (1, 1, 1000, 33),
        (1, 33, 33, 1000),
        (1, 1000, 1000, 1),
        (1, 257, 257, 257),
    ],
)
@requires_gpu
@gpu
def test_gpu_batched_gemm_matches_float64_reference(shape):
    batch, m, n, k = shape
    a = rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32)
    got = gpuinfer.batched_gemm(a, b)
    expected = np.matmul(a.astype(np.float64), b.astype(np.float64))
    tol = _gemm_tolerance(a, b, k)
    assert got.shape == (batch, m, n)
    assert np.isfinite(got).all()
    err = np.abs(got.astype(np.float64) - expected)
    assert err.max() <= tol, f"max abs err {err.max()} > tol {tol} for {shape}"


@requires_gpu
@gpu
def test_gpu_batched_gemm_noncontiguous_input():
    batch, m, n, k = 2, 64, 48, 32
    a = rng.uniform(-1.0, 1.0, (batch, m, k)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (batch, k, n)).astype(np.float32)
    a_view = np.ascontiguousarray(a.transpose(0, 2, 1)).transpose(0, 2, 1)
    assert not a_view.flags["C_CONTIGUOUS"]
    got = gpuinfer.batched_gemm(a_view, b)
    expected = np.matmul(a, b)
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-4)


def test_batched_gemm_rejects_float64():
    a = rng.uniform(-1.0, 1.0, (1, 32, 32))
    b = rng.uniform(-1.0, 1.0, (1, 32, 32))
    with pytest.raises(ValueError):
        gpuinfer.batched_gemm(a, b)


def test_batched_gemm_rejects_bad_dtype():
    a = rng.uniform(-1.0, 1.0, (1, 32, 32))
    b = rng.uniform(-1.0, 1.0, (1, 32, 32)).astype(np.float32)
    with pytest.raises(ValueError):
        gpuinfer.batched_gemm(a, b)


def test_batched_gemm_rejects_bad_ndim():
    a = rng.uniform(-1.0, 1.0, (32, 32)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (32, 32)).astype(np.float32)
    with pytest.raises(ValueError):
        gpuinfer.batched_gemm(a, b)


def test_batched_gemm_rejects_shape_mismatch():
    a = rng.uniform(-1.0, 1.0, (2, 32, 32)).astype(np.float32)
    b = rng.uniform(-1.0, 1.0, (3, 32, 32)).astype(np.float32)
    with pytest.raises(ValueError):
        gpuinfer.batched_gemm(a, b)
    b = rng.uniform(-1.0, 1.0, (2, 64, 32)).astype(np.float32)
    with pytest.raises(ValueError):
        gpuinfer.batched_gemm(a, b)


def test_submit_job_validates_arguments():
    with pytest.raises(ValueError):
        gpuinfer.submit_job(0, 32, 32)
    with pytest.raises(ValueError):
        gpuinfer.submit_job(32, 32, 32, batch=0)


def test_submit_job_returns_id():
    job_id = gpuinfer.submit_job(32, 32, 16, batch=1, priority=1)
    assert isinstance(job_id, int) and job_id >= 0


def test_scheduler_lifecycle_and_stats():
    sched = gpuinfer.Scheduler(max_batch=4, workers=1, impl="tiled")
    sched.start()
    stats = sched.stats()
    assert stats["running"] is True
    for key in (
        "submitted",
        "completed",
        "queued",
        "failed",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "throughput_jobs_per_s",
        "mean_gpu_utilization",
    ):
        assert key in stats
    sched.stop()
    assert sched.stats()["running"] is False


def test_scheduler_rejects_bad_impl():
    with pytest.raises(ValueError):
        gpuinfer.Scheduler(impl="bogus")
