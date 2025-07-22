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

// This file defines a standard interface to the GPU linear algebra libraries.

#ifndef JAXLIB_GPU_SOLVER_INTERFACE_H_
#define JAXLIB_GPU_SOLVER_INTERFACE_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/vendor.h"

#ifdef JAX_GPU_CUDA
#include "third_party/gpus/cuda/include/cusolverDn.h"
#endif

namespace jax
{
  namespace JAX_GPU_NAMESPACE
  {
    namespace solver
    {

#define JAX_GPU_SOLVER_EXPAND_DEFINITION(ReturnType, FunctionName) \
  template <typename T1, typename T2>                             \
  ReturnType FunctionName(                                        \
      JAX_GPU_SOLVER_##FunctionName##_ARGS(T1, T2)) {             \
    return absl::UnimplementedError(absl::StrFormat(                \
      #FunctionName " not implemented for types %s, %s",           \
      typeid(T1).name(), typeid(T2).name()));                      \
    }                                                             \
  template <>                                                      \
  ReturnType FunctionName<double, double>(                         \
      JAX_GPU_SOLVER_##FunctionName##_ARGS(double, double));       \
  template <>                                                      \
  ReturnType FunctionName<double, float>(                          \
      JAX_GPU_SOLVER_##FunctionName##_ARGS(double, float));        \
  template <>                                                      \
  ReturnType FunctionName<float, float>(                           \
      JAX_GPU_SOLVER_##FunctionName##_ARGS(float, float));       

// Linear solve: Gesv
#define JAX_GPU_SOLVER_GesvBufferSize_ARGS(TypeMain, TypeLow, ...) \
  gpusolverDnHandle_t handle, int n, int nrhs
  JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::StatusOr<size_t>, GesvBufferSize);
#undef JAX_GPU_SOLVER_GesvBufferSize_ARGS

#define JAX_GPU_SOLVER_Gesv_ARGS(TypeMain, TypeLow, ...)           \
  gpusolverDnHandle_t handle, int n, int nrhs, TypeMain *a, TypeMain *b,TypeMain *x, TypeMain *workspace, size_t lwork, int *niter, int *dipiv, int *info
  JAX_GPU_SOLVER_EXPAND_DEFINITION(absl::Status, Gesv);
#undef JAX_GPU_SOLVER_Gesv_ARGS

#undef JAX_GPU_SOLVER_EXPAND_DEFINITION
    } // namespace solver
  } // namespace JAX_GPU_NAMESPACE
} // namespace jax

#endif // JAXLIB_GPU_SOLVER_INTERFACE_H_
