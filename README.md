# Iterative Refinement Linear Solve
The code in this repository is a Jax interface for the `gesv` linear solver in CUSOLVE (see the [paper](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf) Haidar, Azzam and Tomov (2018)).

# Important notes

This code only supports a GPU backend. 

# Installation

To compile this library, you will need to following:
- A Python environment with a Jax installation.
- CUDA (>=12.3) and CUDNN (Compatible with CUDA), findable through CMAKE's `find_package`
- Cuda compilation tools (I used release 12.3, V12.3.107)
- GCC compiler that can compile C++17

The compilation is a little involved at the moment, but that could be improved in the future.
Ideally, the whole thing will be a pip install command, but this will likely be hard given that
we depend on a CUDA installation being present.

As of July 22nd, 2025, the following works on the Flation cluster. Clone the repo
```bash
git clone git@github.com:therooler/irgesv_jax.git
cd irgesv_jax
``` 
Install Python environment:
```bash
module load python/3.11.7
python -m venv .venv
source .venv/bin/activate
pip install "jax[cuda12]==0.6.2" pytest
```
Note that since Jax is shipped with CUDA these days. However, to compile we still need a local
version of all the CUDA runtime libraries to compile against.  On the cluster, we use
```bash
module load gcc/12.2.0 cuda/12.3.2 cudnn/8.9.7.29-12
```
Next, we make a build directory.
```bash
mkdir build && cd build
```
Then, we configure the dependencies from the CMakeLists.txt file:
```bash
cmake ..
```
This command will do the following:
- Git clone Abseil, a C++ package by google that is required to link against. 
- Find CudaToolkit, CUDNN
- Git clone jax, because we need some of the jaxlib libraries.
- Find the XLA directory of your jax installation in `.venv`
- Create symlinks to your CUDA and CUDNN directories

We then build the package with
```bash
cmake --build . --target irgesv_cuda
```

Going up a directory and running
```bash
cd ..
pytest tests.py
```
should then work. I also provide a benchmark to compare the standard linalgsolve against. 
