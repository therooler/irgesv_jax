cmake_minimum_required(VERSION 3.16)
project(JaxSolve)

# Get CPM
set(CPM_DOWNLOAD_VERSION 0.42.0) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()
include(${CPM_DOWNLOAD_LOCATION})

# Abseil requires C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

CPMAddPackage(
  NAME absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp
  GIT_TAG 20250512.1
)
# Import CUDAToolkit (Requires CUDA_HOME to be set)
find_package(CUDAToolkit REQUIRED)
# Import CUDAToolkit (Requires CUDA_HOME to be set)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
find_package(CUDNN REQUIRED)
message(STATUS "CUDA include directory: $ENV{CUDA_ROOT}")
message(STATUS "CUDNN include directory: $ENV{CUDNN_ROOT}")
# Enable CUDA 
enable_language(CUDA)

# Get Jaxlib
CPMAddPackage(
  NAME jax
  GIT_REPOSITORY https://github.com/jax-ml/jax.git
  GIT_TAG jax-v0.6.2
  DOWNLOAD_ONLY YES
)
message(STATUS "Jax source directory: ${jax_SOURCE_DIR}")
# Use installed version of Jax for XLA library
execute_process(
  COMMAND "python"
          "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Since jaxlib is asking for #include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
# we need to create a symlink to to CUDA_HOME and pretend it can be accessed from third_party
# Create the target directory
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/gpus)
execute_process(
    COMMAND ln -sfn $ENV{CUDA_ROOT} ${CMAKE_SOURCE_DIR}/third_party/gpus/cuda
    RESULT_VARIABLE ln_result
)
if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for CUDA: $ENV{CUDA_ROOT}")
endif()
execute_process(
    COMMAND ln -sfn $ENV{CUDNN_ROOT}/include ${CMAKE_SOURCE_DIR}/third_party/gpus/cudnn
    RESULT_VARIABLE ln_result
)
if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for CUDNN: $ENV{CUDNN_ROOT}/include")
endif()
execute_process(
    COMMAND ln -sfn ${XLA_DIR}/xla ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE ln_result
)
if(NOT ln_result EQUAL 0)
    message(FATAL_ERROR "Failed to create symlink for XLA_DIR: ${XLA_DIR}")
endif()

# Set JAX_GPU_CUDA to 1 (this is required for jaxlib to compile correctly)
add_definitions(-DJAX_GPU_CUDA=1)
# Add libraries
add_library(irgesv_cuda SHARED 
src/irgesv.cu
src/solver_interface.cc
${jax_SOURCE_DIR}/jaxlib/gpu/solver_handle_pool.cc
# ${jax_SOURCE_DIR}/jaxlib/gpu/solver_interface.cc
${jax_SOURCE_DIR}/jaxlib/gpu/gpu_kernel_helpers.cc
)
# Make sure we can see "thirdparty/gpus" and jax
# target_include_directories()
target_include_directories(irgesv_cuda PUBLIC ${CMAKE_SOURCE_DIR} ${jax_SOURCE_DIR})
# target_include_directories(irgesv_cuda PUBLIC ${XLA_DIR})
# Set the CUDA architecture
set_target_properties(irgesv_cuda PROPERTIES 
CUDA_ARCHITECTURES "all"
BUILD_RPATH $ENV{CUDA_ROOT}/lib64
)
# Link against CUDA libraries and absl
target_link_libraries(irgesv_cuda 
absl::strings
absl::status
absl::statusor
CUDA::cusolver
CUDA::cupti
cuda
)