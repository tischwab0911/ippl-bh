# -----------------------------------------------------------------------------
# unit_tests/NBody/CMakeLists.txt
#
# Sanity tests for the vendored cstone + ryoanji codebases.
# Each source file is a verbatim copy of an upstream SPH-EXA unit test.
# These tests verify that the vendored sources compile, link, and behave
# the same way they do in the upstream tree.
#
# Only built when IPPL_ENABLE_NBODY=ON. Backend follows IPPL_PLATFORMS: the
# header tests and the backend-agnostic .cpp tests build everywhere; the .cu
# tests are gated behind a device compiler (CUDA/HIP).
#
# Note: add_ippl_test() accepts a LINK_LIBS argument but currently only honors
# it on the INTEGRATION branch. We attach the cstone/ryoanji targets via an
# explicit target_link_libraries() call after each add_ippl_test() instead.
# -----------------------------------------------------------------------------

file(RELATIVE_PATH _relPath "${PROJECT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Adding NBody unit tests found in ${_relPath}")

# CPU-only cstone tests: header-only, no GPU code, no MPI.
add_ippl_test(CstoneArray
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(CstoneArray PRIVATE ippl::cstone_headers)

add_ippl_test(CstoneBox
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(CstoneBox PRIVATE ippl::cstone_headers)

# CPU ryoanji test: exercises multipole kernel headers + cstone headers.
add_ippl_test(RyoanjiKernelCpu
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody)
target_link_libraries(RyoanjiKernelCpu PRIVATE
  ippl::ryoanji_headers
  ippl::cstone_headers)

# Backend-agnostic tests (.cpp): build on every backend. They exercise the
# build's NBodyAcc (execution::Gpu on CUDA/HIP, execution::Cpu on CPU) through
# the same library API, using the portable up/download helper in
# NBodyTestUtil.hpp. Each provides
# its own main() (ippl::initialize wraps MPI + Kokkos; cstone's Domain calls MPI
# collectives unconditionally), so no USE_GTEST_MAIN. RDC stays off: these TUs
# include <Kokkos_Core.hpp> via NBodyParticleContainer.hpp, which trips desul's
# atomics check under relocatable device code.
foreach(_t NBodyContainerSync NBodyContainerFieldList NBodyContainerAttribute
           NBodyContainerInjectDestroy LeapfrogStepperBasics LeapfrogDriftWrap)
  add_ippl_test(${_t}
    SOURCES ${_t}.cpp
    NO_MPI
    LABELS nbody)
  target_link_libraries(${_t} PRIVATE ippl::nbody)
  set_target_properties(${_t} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endforeach()

# GPU-only tests: their .cu sources (ryoanji::directSum) require a device
# compiler. Gated so a CPU build still configures. BH-vs-direct correctness on
# CPU is covered by RyoanjiKernelCpu + the Landau/DIH physics runs.
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)

if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
  set_source_files_properties(RyoanjiDirect.cu NBodySolverDirect.cu PROPERTIES LANGUAGE HIP)
endif()

# GPU ryoanji test: exercises the direct N² kernel from ryoanji_kernels.
add_ippl_test(RyoanjiDirect
  SOURCES RyoanjiDirect.cu
  NO_MPI
  USE_GTEST_MAIN
  LABELS nbody gpu)
target_link_libraries(RyoanjiDirect PRIVATE
  ippl::ryoanji
  ippl::cstone_gpu)
set_target_properties(RyoanjiDirect PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON)

# BH solver sanity test: BH ≈ direct N² on a small open-BC system. ippl::ryoanji
# provides the GPU directSum() reference (instantiations in multipole_holder.cu).
add_ippl_test(NBodySolverDirect
  SOURCES NBodySolverDirect.cu
  NO_MPI
  LABELS nbody gpu)
target_link_libraries(NBodySolverDirect PRIVATE
  ippl::nbody
  ippl::ryoanji)
set_target_properties(NBodySolverDirect PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

endif()  # CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER
