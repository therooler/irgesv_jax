/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "solver_interface.h"

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverDn.h"
#endif

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace solver {

// LU decomposition: getrf

#define JAX_GPU_DEFINE_GESV(TypeMain, TypeLow, Name)                           \
  template <>                                                                  \
  absl::StatusOr<size_t> GesvBufferSize<TypeMain, TypeLow>(                    \
    gpusolverDnHandle_t handle, int n, int nrhs) {                             \
    size_t lwork;                                                              \
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(                                         \
      Name##_bufferSize(handle, n, nrhs,/*A=*/nullptr, n, /*dipiv*/nullptr,    \
        /*B=*/nullptr, n, /*X=*/nullptr, n, /*dwork*/nullptr, &lwork)));       \
    return lwork;                                                              \
  }                                                                            \
  template <>                                                                  \
  absl::Status Gesv<TypeMain, TypeLow>(                                        \
    gpusolverDnHandle_t handle, int n, int nrhs,                               \
    TypeMain *a, TypeMain *b, TypeMain *x, TypeMain *workspace,                \
    size_t lwork, int *niter, int *dipiv, int *info) {                         \
    return JAX_AS_STATUS(                                                      \
      Name(handle, n, nrhs, a, n, dipiv,                                       \
        b, n, x, n, workspace, lwork, niter, info));                           \
  }

JAX_GPU_DEFINE_GESV(double, double, cusolverDnDDgesv);
JAX_GPU_DEFINE_GESV(double, float, cusolverDnDXgesv);
JAX_GPU_DEFINE_GESV(double, half, cusolverDnDHgesv);
JAX_GPU_DEFINE_GESV(double, nv_bfloat16, cusolverDnDBgesv);
JAX_GPU_DEFINE_GESV(float, float, cusolverDnSXgesv);
JAX_GPU_DEFINE_GESV(float, half, cusolverDnSHgesv);
JAX_GPU_DEFINE_GESV(float, nv_bfloat16, cusolverDnSBgesv);

// JAX_GPU_DEFINE_GESV(double, double, cusolverDnDDgesv);
// JAX_GPU_DEFINE_GESV(double, float, cusolverDnDSgesv);
// JAX_GPU_DEFINE_GESV(float, float, cusolverDnSSgesv);
// JAX_GPU_DEFINE_Gesv(double, half, cusolverDnDSgesv);
// JAX_GPU_DEFINE_Gesv(double, bfloat, cusolverDnDSgesv);
#undef JAX_GPU_DEFINE_GESV

}  // namespace solver
}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
