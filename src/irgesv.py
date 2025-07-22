import os
import ctypes

import numpy as np

import jax
from jax._src.typing import ArrayLike, Array, DTypeLike
from jax._src.numpy.util import promote_dtypes_inexact, ensure_arraylike
from jax._src.lax.linalg import _check_solve_shapes
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pathlib import Path

# Load the shared library with the FFI target definitions
current_file = Path(__file__).resolve()
SHARED_LIBRARY = f"{current_file.parent.parent}/build/libirgesv_cuda.so"
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)
jax.ffi.register_ffi_target("gesv_cuda", jax.ffi.pycapsule(library.GesvFfi),
                            platform="CUDA")

def _irgesv_impl(a, b, dtype):
    out_type = jax.ShapeDtypeStruct(b.shape, a.dtype)
    niter_type = jax.ShapeDtypeStruct((b.shape[0], 1), jnp.int32)
    info_out_type = jax.ShapeDtypeStruct((b.shape[0], 1), jnp.int32)
    # Check for invalid inputs that previously would have led to a batched 1D solve:
    def impl(target_name):
        return lambda _a, _b: jax.ffi.ffi_call(
        target_name,
        (out_type, niter_type, info_out_type),
        input_layouts=((0,2,1),(0,2,1), ()),
        output_layouts=((0,2,1),(0,1), (0,1)),
        vmap_method="broadcast_all",
        )(_a, _b, jnp.array(1, dtype=dtype))
    return jax.lax.platform_dependent(a, b, cuda=impl("gesv_cuda"))

def irgesv(a: ArrayLike, b: ArrayLike, dtype: DTypeLike) -> Array:
    """
    Solves a system of linear equations `Ax = b` using iterative refinement
    with dynamic precision in the inner solver loop.

    This function wraps an implementation of CUSOLVEs `gesv`
    a linear solver (`_irgesv_impl`) that employs iterative refinement to 
    improve the accuracy of the solution, working with a lower-precision solve 
    and refining in a higher precision. It supports batched inputs and handles both single
    right-hand side vectors and multiple RHS columns.

    Parameters
    ----------
    a : ArrayLike
        The coefficient matrix `A`, of shape (..., M, M), where `...` is
        an optional batch dimension. Only a single batch dimension is supported.
    b : ArrayLike
        The right-hand side `b`, of shape (..., M) or (..., M, K), where `K`
        is the number of right-hand sides. Automatically reshaped to match `a`
        as needed.
    dtype : DTypeLike
        The target floating-point precision to use in the inner refinement loop
        (e.g., `jnp.float32`, `jnp.float32`, `jnp.float64`).

    Returns
    -------
    x : jax.Array
        The solution to the system `Ax = b`, with shape matching `b`.
    niter : jax.Array
        The number of refinement iterations used per system, with shape matching
        the batch dimensions.
    info : jax.Array
        Error status of the inner solver. 
        If iter is
        <0 : iterative refinement has failed, main precision (Inputs/Outputs precision) factorization has been performed
        -1 : taking into account machine parameters, n, nrhs, it is a priori not worth working in lower precision
        -2 : overflow of an entry when moving from main to lower precision
        -3 : failure during the factorization
        -5 : overflow occurred during computation
        -50: solver stopped the iterative refinement after reaching maximum allowed iterations
        >0 : iter is a number of iterations solver performed to reach convergence criteria

    Raises
    ------
    ValueError
        If more than one batch dimension is provided or if input shapes are
        incompatible for linear solving.

    Notes
    -----
    - Internally promotes `a` and `b` to a common inexact dtype.
    - Automatically handles input reshaping for vector and single-system cases.
    - Designed for use with JAX and supports batched linear solves with shared
      refinement configuration.

    Example
    -------
    >>> A = jnp.array([[3.0, 2.0], [1.0, 4.0]])
    >>> b = jnp.array([5.0, 6.0])
    >>> x, res, niters = irgesv(A, b, dtype=jnp.float32)
    """
    a, b = ensure_arraylike("gesv", a, b)
    a, b = promote_dtypes_inexact(a, b)
    _check_solve_shapes(a, b)
    if a.ndim>3:
        raise ValueError("Only a single batch dimension is supported right now, a=[i, m, m]; got a={a.shape}")
    no_batch = a.ndim==2
    if no_batch:
        # b.shape == [m] -> b.shape == [1, m]
        # b.shape == [m,1] -> b.shape == [1, m, 1]
        a = jnp.expand_dims(a, axis=0)
        b = jnp.expand_dims(b, axis=0)
    single_system = (a.ndim == b.ndim + 1)
    if single_system:
        # b.shape == [..., m] -> b.shape == [..., m, 1]
        b = jnp.expand_dims(b, axis=-1)        
    result = _irgesv_impl(a, b, dtype)
    if no_batch:
        # res.shape == [1, *shape] -> res.shape == [*shape]
        result = tuple(jnp.squeeze(_res, axis=0) for _res in result)
    if single_system:
        # x.shape == [..., n, 1] -> res.shape == [..., n]
        result = (jnp.squeeze(result[0], axis=-1), result[1], result[2])
    return result