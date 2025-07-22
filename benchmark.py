import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
jax.config.update("jax_enable_x64", True)

import subprocess
# Step 1: Check if JAX detects any GPU
gpu_devices = jax.devices("gpu")
if not gpu_devices:
    raise SystemError("No GPU found (jax.devices did not detect any GPU)")

# Step 2: Try to get GPU name via nvidia-smi
try:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        encoding="utf-8"
    )
    # Get name of the first GPU
    gpu_name = output.strip().split('\n')[0]
except Exception as e:
    gpu_name = f"Error querying GPU name: {e}"
print(gpu_devices)
print(gpu_name)

import jax.numpy as jnp
import numpy as np
from src.irgesv import irgesv
import time
from functools import partial
import matplotlib.pyplot as plt
def examine_jaxpr(closed_jaxpr):
    jaxpr = closed_jaxpr.jaxpr
    print("invars:", jaxpr.invars)
    print("outvars:", jaxpr.outvars)
    print("constvars:", jaxpr.constvars)
    for eqn in jaxpr.eqns:
        print("equation:", eqn.invars, eqn.primitive, eqn.outvars, eqn.params)
        print()
    print("jaxpr:", jaxpr)

def benchmark_gesv_vs_jax_solve(solver, n, dtype_main, num_trials=25, seed=1234):
    shape_a = (1, n, n)
    shape_b = (1, n, 1)
    key = jax.random.key(seed)
    a = []
    b = []

    # Benchmark solver
    times = []
    for trial in range(num_trials+1):        
        key, key_a, key_b = jax.random.split(key, 3)
        a = jax.random.normal(key_a, shape=shape_a, dtype=dtype_main)
        da = jnp.expand_dims(jnp.squeeze(jnp.abs(jnp.sum(a, axis=-1))),axis=0)
        a = a + jnp.diag(da)
        b = jax.random.normal(key_b, shape=shape_b, dtype=dtype_main)
        a_mat = jax.device_put(a, device=jax.devices("gpu")[0]).block_until_ready()
        b_mat = jax.device_put(b, device=jax.devices("gpu")[0]).block_until_ready()
        start = time.time()
        result = solver(a_mat, b_mat)
        out = result[0].block_until_ready()
        if trial>0:
            times.append(time.time() - start)
        print(jnp)
        if len(result)>2:
            print(f"niter: {result[1]}")
    return times
# Example usage:
if __name__ == "__main__":

    times_irgesv_total, times_linalg_solve_total = [], []
    matrix_sizes = []
    ntrials = 5
    type_main = jnp.float64
    type_low = jnp.float32
    print(jax.lax.Precision.HIGHEST)
    solver_irgesv = jax.jit(partial(irgesv, dtype=type_low))
    solver_linalg_solve = jax.jit(jnp.linalg.solve)
    for power in np.arange(2, 15):  # Adjust range as needed
        n = 2**power
        print(f"Size {n}x{n}")
        matrix_sizes.append(n)
        times_irgesv = benchmark_gesv_vs_jax_solve(solver_irgesv, n, type_main, num_trials=ntrials)
        times_linalg_solve = benchmark_gesv_vs_jax_solve(solver_linalg_solve, n, type_main, num_trials=ntrials)
        times_linalg_solve_total.append(times_linalg_solve)
        times_irgesv_total.append(times_irgesv)
        print(f"jax.linalg.solve avg time: {np.mean(np.array(times_linalg_solve)):.6f} s")
        print(f"irgesv FFI solver avg time: {np.mean(np.array(times_irgesv)):.6f} s")
    # Calculate averages and variances
    avg_times_irgesv = [np.mean(times) for times in times_irgesv_total]
    var_times_irgesv = [np.var(times) for times in times_irgesv_total]
    avg_times_linalg_solve = [np.mean(times) for times in times_linalg_solve_total]
    var_times_linalg_solve = [np.var(times) for times in times_linalg_solve_total]

    # Calculate speedup
    speedup = [avg_jax / avg_irgesv for avg_jax, avg_irgesv in zip(avg_times_linalg_solve, avg_times_irgesv)]

    # Plotting in subplots
    fig, axs = plt.subplots(1,2, figsize=(8, 5))

    # Plot benchmark times
    axs[0].errorbar(matrix_sizes, avg_times_irgesv, yerr=np.sqrt(var_times_irgesv), label="irgesv", fmt='-o')
    axs[0].errorbar(matrix_sizes, avg_times_linalg_solve, yerr=np.sqrt(var_times_linalg_solve), label="jax.linalg.solve", fmt='-s')
    axs[0].set_xlabel("Matrix size (n)")
    axs[0].set_ylabel("Time (s)")
    axs[0].set_title("Benchmark: irgesv vs jax.linalg.solve")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")

    # Plot speedup
    axs[1].plot(matrix_sizes, speedup, '-o')
    axs[1].set_xlabel("Matrix size (n)")
    axs[1].set_ylabel("Speedup")
    axs[1].set_title("Speedup of irgesv over jax.linalg.solve")
    axs[1].grid(True)
    axs[1].set_xscale("log")
    axs[1].set_ylim([0,15])

    # Adjust layout and show
    plt.suptitle(f"TypeMain {type_main.__name__} - TypeLow {type_low.__name__} - {ntrials} trials")
    plt.savefig(f"benchmark_{gpu_name}_{type_main.__name__}_{type_low.__name__}.pdf")
    plt.show()
