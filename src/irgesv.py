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
SHARED_LIBRARY = f"{current_file.parent.parent}/build/libhello_world_cuda.so"
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