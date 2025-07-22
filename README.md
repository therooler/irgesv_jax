# Iterative Refinement Linear Solve
The code in this repository is a Jax interface for the `gesv` linear solver in CUSOLVE (see the [paper](https://www.netlib.org/utk/people/JackDongarra/PAPERS/haidar_fp16_sc18.pdf) Haidar, Azzam and Tomov (2018)).

# Installation

To compile this library, you will need to following:
- A Python environment with a Jax installation.
- CUDA (>=12.3) and CUDNN (Compatible with CUDA), findable through CMAKE's `find_package`
- Cuda compilation tools (I used release 12.3, V12.3.107)
- GCC compiler that can compile C++17

As of July 22nd, 2025, the following works on the Flation cluster. Install Python environment:
```bash
module load python/3.11.7
python -m venv .venv
source .venv/bin/activate
pip install "jax[cuda12]==0.6.2" pytest
```

Insta
```bash
git clone 
cd 
``` gcc/12.2.0 cuda/12.3.2 cudnn/8.9.7.29-12

Make sure you have a python installation with an up-to-date version of jax
```bash
cd ~/
mkdir jax_cuda && cd jax_cuda
```
Activate module for GCC so we have access to c++17 and activate CUDA module
```bash
```
Make directory for source
```bash
mkdir src && cd src
```
Install abseil, a Google C++ library,
```bash
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/jax_cuda/src/install -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --target install
cd ../..
```
We also need a copy of jaxlib,

```bash
git clone https://github.com/jax-ml/jax.git
```

Install own library
```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/jax_cuda/src/install
cmake --build . --target hello_world_cuda
```

Test the setup by running 

```bash
./hello_world_cuda
```

The tricky thing is the location of the CUDA libraries. Jaxlib is looking for them in relative paths like
`#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"`, however the typical setup is one where
there is some global cuda installation in an HPC setting that we need to refer to. We therefore make
symlinks between `third_party/gpus/cuda` and the user installed `$CUDA_ROOT` and `$CUDNN_ROOT`. 

cd ..
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=~/jax_cuda/src/install
cmake --build . --target hello_world_cuda
./hello_world_cuda

cd ..
rm -rf build
mkdir build && cd build
cmake .. 
cmake --build . --target hello_world_cuda
./hello_world_cuda