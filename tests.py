import pytest
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
print(jax.devices('gpu'))
from src.irgesv import irgesv
#Run with pytest -v -s tests.py

def assert_ffi_solve(a, b, dtype_low):
    x_ffi, niter, info = irgesv(a, b, dtype_low)
    if a.ndim == 3 and b.ndim == 2 and a.shape[0] == b.shape[0]:
        x_linalg_solve = jnp.linalg.solve(a, jnp.expand_dims(b, axis=-1)).squeeze(-1)
    else:
        x_linalg_solve = jnp.linalg.solve(a, b)
    assert jnp.allclose(x_ffi, x_linalg_solve, atol=1e-7)

def general_test(shape_a, shape_b, dtype_main, dtype_low, seed):
    key = jax.random.key(seed)
    key_a, key_b = jax.random.split(key,2)
    dtype_main = jnp.float64
    a = jax.random.normal(key_a, shape=shape_a, dtype=dtype_main)
    b = jax.random.normal(key_b, shape=shape_b, dtype=dtype_main)
    assert_ffi_solve(a, b, dtype_low)

def get_shapes_a_b(batch,n,nrhs):
    if batch>0:
        shape_a = [batch,]
        shape_b = [batch,]
    else:
        shape_a = []
        shape_b = []
    
    if nrhs>0:
        shape_a.extend([n,n])
        shape_b.extend([n,nrhs])
    else:
        shape_a.extend([n,n])
        shape_b.extend([n,])
    return shape_a, shape_b

@pytest.mark.parametrize("batch", (0,1,3))
@pytest.mark.parametrize("n", (2,3,7))
@pytest.mark.parametrize("nrhs", (0,1,3))
@pytest.mark.parametrize("dtype_low", (jnp.float64, jnp.float32, jnp.float16, jnp.bfloat16))
def test_all_f64(batch:int, n:int, nrhs:int, dtype_low):
    n = 10
    seed=1234
    dtype_main = jnp.float64
    shape_a, shape_b = get_shapes_a_b(batch, n, nrhs)
    general_test(shape_a, shape_b, dtype_main, dtype_low, seed)

@pytest.mark.parametrize("batch", (0,1,3))
@pytest.mark.parametrize("n", (2,3,7))
@pytest.mark.parametrize("nrhs", (0,1,3))
@pytest.mark.parametrize("dtype_low", (jnp.float32, jnp.float16, jnp.bfloat16))
def test_all_f32(batch:int, n:int, nrhs:int, dtype_low):
    n = 10
    seed=1234
    dtype_main = jnp.float32
    shape_a, shape_b = get_shapes_a_b(batch, n, nrhs)
    general_test(shape_a, shape_b, dtype_main, dtype_low, seed)
