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

#include "jaxlib/gpu/solver_kernels_ffi.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#if JAX_GPU_HAVE_64_BIT
#include <cstddef>
#endif

#ifdef JAX_GPU_CUDA
#include <limits>
#endif

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/blas_handle_pool.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/make_batch_pointers.h"
#include "jaxlib/gpu/solver_handle_pool.h"
#include "solver_interface.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace jax
{
  template <typename T>
  inline absl::StatusOr<T*> AllocateWorkspaceBytes(
      ::xla::ffi::ScratchAllocator& scratch, int64_t n_bytes,
      std::string_view name) {
    auto maybe_workspace = scratch.Allocate(n_bytes);
    if (!maybe_workspace.has_value()) {
      return absl::Status(
          absl::StatusCode::kResourceExhausted,
          absl::StrFormat("Unable to allocate workspace for %s", name));
    }
    return static_cast<T*>(maybe_workspace.value());
  }
  namespace JAX_GPU_NAMESPACE
  {

    namespace ffi = ::xla::ffi;

#define SOLVER_DISPATCH_IMPL(impl, ...)       \
  switch (dataTypeMain)                       \
  {                                           \
  case ffi::F32:                              \
    switch (dataTypeLow)                      \
    {                                         \
    case ffi::F32:                            \
      return impl<float, float>(__VA_ARGS__); \
    case ffi::F16:                            \
      return impl<float, half>(__VA_ARGS__);  \
    case ffi::BF16:                            \
      return impl<float, nv_bfloat16>(__VA_ARGS__);  \
    default:                                  \
      break;                                  \
    }                                         \
    break;                                    \
  case ffi::F64:                              \
    switch (dataTypeLow)                      \
    {                                         \
    case ffi::F64:                            \
      return impl<double, double>(__VA_ARGS__); \
    case ffi::F32:                            \
      return impl<double, float>(__VA_ARGS__); \
      case ffi::F16:                            \
      return impl<double, half>(__VA_ARGS__); \
    case ffi::BF16:                            \
      return impl<double, nv_bfloat16>(__VA_ARGS__); \
    default:                                  \
      break;                                  \
    }                                         \
  default:                                    \
    break;                                    \
  }
    /*
    cusolverDn gesv Function Variants and Data Type Support

    +---------------------+---------+--------------+
    | Function            | Compute | Input/Output |
    +=====================+=========+==============+
    | cusolverDnDDgesv    | double  | double       |
    +---------------------+---------+--------------+
    | cusolverDnDSgesv*   | double  | single       |
    +---------------------+---------+--------------+
    | cusolverDnDHgesv    | double  | half         |
    +---------------------+---------+--------------+
    | cusolverDnDBgesv    | double  | bfloat       |
    +---------------------+---------+--------------+
    | cusolverDnDXgesv    | double  | tensorfloat  |
    +---------------------+---------+--------------+
    | cusolverDnSSgesv    | float   | single       |
    +---------------------+---------+--------------+
    | cusolverDnSHgesv    | float   | half         |
    +---------------------+---------+--------------+
    | cusolverDnSBgesv    | float   | bfloat       |
    +---------------------+---------+--------------+
    | cusolverDnSXgesv    | float   | tensorfloat  |
    +---------------------+---------+--------------+
    */
    template <typename TMain, typename TLow>
    ffi::Error GesvImpl(int64_t batches, int64_t rows, int64_t cols,
                        gpuStream_t stream, ffi::ScratchAllocator &scratch,
                        ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::Result<ffi::AnyBuffer> out,
                        ffi::Result<ffi::Buffer<ffi::S32>> niter, ffi::Result<ffi::Buffer<ffi::S32>> info)
    {
      // std::cout << absl::StrFormat("Called GesvImpl for types %s, %s", 
      // typeid(TMain).name(), typeid(TLow).name()) << std::endl;

      FFI_ASSIGN_OR_RETURN(auto handle, SolverHandlePool::Borrow(stream));
      FFI_ASSIGN_OR_RETURN(auto batch, MaybeCastNoOverflow<int>(batches));
      FFI_ASSIGN_OR_RETURN(auto n, MaybeCastNoOverflow<int>(rows));
      FFI_ASSIGN_OR_RETURN(auto nrhs, MaybeCastNoOverflow<int>(cols));
      FFI_ASSIGN_OR_RETURN(size_t lwork, (solver::GesvBufferSize<TMain, TLow>(handle.get(), n, nrhs)));
      // size_t workspace_size = ceil(lwork / sizeof(TMain));
      FFI_ASSIGN_OR_RETURN(auto workspace,
                      AllocateWorkspaceBytes<TMain>(scratch, lwork, "irgesv_workspace"));

      auto a_data = static_cast<TMain*>(a.untyped_data());
      auto b_data = static_cast<TMain*>(b.untyped_data());
      auto out_data = static_cast<TMain*>(out->untyped_data());

      // Assign output buffers
      auto niter_data = niter->typed_data();
      int niter_data_host = 0;
      FFI_ASSIGN_OR_RETURN(int* dipiv_data,
                AllocateWorkspaceBytes<int>(scratch, batch * n, "dipiv_workspace"));
      auto info_data = info->typed_data();
      
      if (b_data != out_data) {
        FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(gpuMemcpyAsync(
            out_data, b_data, b.size_bytes(), gpuMemcpyDeviceToDevice, stream)));
      }    

      int a_step = n * n;
      int b_step = n * nrhs;

      for (auto i = 0; i < batch; ++i) {
        FFI_RETURN_IF_ERROR_STATUS(solver::Gesv<TMain, TLow>(
          handle.get(), n, nrhs, a_data, b_data, out_data, workspace, lwork, &niter_data_host, dipiv_data, info_data));
          // // Move pointers
          a_data += a_step;
          b_data += b_step;
          out_data += b_step;
          dipiv_data += nrhs;
          ++niter_data_host;
          ++info_data;
      }
      // Move niters to GPU for return
      JAX_FFI_RETURN_IF_GPU_ERROR(gpuMemcpy(
        niter_data, &niter_data_host, sizeof(int), gpuMemcpyHostToDevice));

      return ffi::Error::Success();
    }

    ffi::Error GesvDispatch(gpuStream_t stream, ffi::ScratchAllocator scratch,
                            ffi::AnyBuffer a, ffi::AnyBuffer b, ffi::AnyBuffer c,
                            ffi::Result<ffi::AnyBuffer> out,
                            ffi::Result<ffi::Buffer<ffi::S32>>niter, ffi::Result<ffi::Buffer<ffi::S32>> info)
    {
      // std::cout << "GesvDispatch called" << std::endl;
      // The high precision datatype
      auto dataTypeMain = a.element_type();
      auto dataTypeLow = c.element_type();
      // std::cout<<"Data type Main: " << dataTypeMain<<std::endl;
      // std::cout<<"Data type Low: " << dataTypeLow<<std::endl;
      if (dataTypeMain != out->element_type())
      {
        return ffi::Error::InvalidArgument(
            "The input and output to getrf must have the same element type");
      }
      FFI_ASSIGN_OR_RETURN((auto [batch, rows, cols]),
                           SplitBatch2D(a.dimensions()));
      FFI_ASSIGN_OR_RETURN((auto [b_batch, b_rows, b_cols]),
                           SplitBatch2D(b.dimensions()));
      if (batch != b_batch)
      {
        return ffi::Error::InvalidArgument(
            "Batch size of A and b must match");
      }
      FFI_RETURN_IF_ERROR(CheckShape(info->dimensions(), batch, "info", "irgesv"));
      FFI_RETURN_IF_ERROR(CheckShape(niter->dimensions(), batch, "niter", "irgesv"));
      FFI_RETURN_IF_ERROR(
          CheckShape(out->dimensions(), {b_batch, b_rows, b_cols}, "out", "irgesv"));
      SOLVER_DISPATCH_IMPL(GesvImpl, batch, b_rows, b_cols, stream, scratch, a, b, out, niter, info);

      return ffi::Error::InvalidArgument(absl::StrFormat(
          "Unsupported combination of data type main %s and data type low %s in irgesv", 
          absl::FormatStreamed(dataTypeMain), absl::FormatStreamed(dataTypeLow)));
    }
    // jax.ffi.ffi_call("getrf", (out_type, ipiv_type, info_out_type))(A.copy().T)
    XLA_FFI_DEFINE_HANDLER_SYMBOL(GesvFfi, GesvDispatch,
                                  ffi::Ffi::Bind()
                                      .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                      .Ctx<ffi::ScratchAllocator>()
                                      .Arg<ffi::AnyBuffer>()        // a
                                      .Arg<ffi::AnyBuffer>()        // b
                                      .Arg<ffi::AnyBuffer>()        // internal dtype
                                      .Ret<ffi::AnyBuffer>()        // X
                                      .Ret<ffi::Buffer<ffi::S32>>() // niter
                                      .Ret<ffi::Buffer<ffi::S32>>() // info
    );

#undef SOLVER_DISPATCH_IMPL

  } // namespace JAX_GPU_NAMESPACE
} // namespace jax