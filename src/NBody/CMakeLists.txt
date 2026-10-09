# -----------------------------------------------------------------------------
# src/NBody/CMakeLists.txt
#
# Producer code for the gridless Barnes-Hut module. Built only when
# IPPL_ENABLE_NBODY=ON (which already requires CUDA in IPPL_PLATFORMS — see
# top-level CMakeLists.txt).
#
# This library holds only *shared* infrastructure: container, BH solver,
# generic time-step primitives (leapfrog kick / drift+wrap), shared diagnostics
# (mean velocity, temperature), and the NBodyManager base class.
#
# Simulation-specific code lives next to its driver in test/BH/.
#
# Portability (mirrors sphexa's per-backend source selection):
#   - GPU build (CUDA or HIP): the device kernels live in .cu translation units,
#     retargeted to HIP when hipcc is the device compiler, linked against the
#     cstone/ryoanji GPU libraries.
#   - CPU build: the kernel modules carry their inline-OpenMP body in the header
#     (if constexpr(kHaveGpu) dispatch), so there is no separate .cpp TU; the
#     bodies instantiate at the call sites, linked against the header-only
#     cstone/ryoanji INTERFACE targets.
# Compute TUs are Kokkos-free; Kokkos appears only in the public container
# header as the type of the consumer-facing unmanaged-View accessor.
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Kokkos-free sync layer (ippl_nbody_sync).
#
# Holds the conserved/dependent FieldList threading (NBodySync.cpp — the only TU
# that instantiates cstone::get<FieldList<...>>) plus, on GPU builds, the container
# device kernels it calls (NBodyParticleContainer.cu, source of
# detail::setHFromLeavesGpu et al.) so those references resolve intra-archive. On
# the CPU build the container kernels are header-inline OpenMP (if constexpr
# dispatch in NBodyParticleContainer.hpp), so no separate TU is needed.
#
# Deliberately does NOT link Kokkos: on the CUDA backend that keeps
# -DKOKKOS_DEPENDENCE off NBodySync.cpp's command line, so kokkos_launch_compiler
# leaves it on plain g++ — where the C++20 NTTP-pack CTAD inside
# cstone::get<FieldList> compiles (nvcc rejects it). The .cu kernel still goes to
# nvcc (CUDA language) and gets its arch from the global CMAKE_CUDA_ARCHITECTURES,
# exactly like the Kokkos-free cstone_gpu target.
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
  set(_NBODY_SYNC_SOURCES core/NBodySync.cpp NBodyParticleContainer.cu)
  if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(NBodyParticleContainer.cu PROPERTIES LANGUAGE HIP)
  endif()
  set(_NBODY_SYNC_BACKEND ippl::cstone_gpu)
else()
  set(_NBODY_SYNC_SOURCES core/NBodySync.cpp)
  set(_NBODY_SYNC_BACKEND ippl::cstone_headers OpenMP::OpenMP_CXX)
endif()

add_library(ippl_nbody_sync STATIC ${_NBODY_SYNC_SOURCES})
target_include_directories(ippl_nbody_sync PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ippl>)
target_compile_features(ippl_nbody_sync PUBLIC cxx_std_20)
target_link_libraries(ippl_nbody_sync PUBLIC ${_NBODY_SYNC_BACKEND})
set_target_properties(ippl_nbody_sync PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(ippl::nbody_sync ALIAS ippl_nbody_sync)

# -----------------------------------------------------------------------------
# Kokkos consumer layer (ippl_nbody): BH solver + time-step / diagnostics kernels.

# Pure host orchestration (no device kernels): always compiled by the CXX
# compiler. On GPU builds USE_CUDA is propagated PUBLIC from cstone_gpu, selecting
# NBodyAcc=execution::Gpu and the GPU MultipoleHolder.
set(_NBODY_COMMON_SOURCES
  NBodySolver.cpp)

# Device-kernel module: all NBody Kokkos-consumer GPU kernels live in the single
# helpers/primitives_gpu.cu TU (diagnostics reductions + the element-wise axpy3
# kick / drift+wrap maps), retargeted to HIP under hipcc, compiled only on GPU
# builds. The CPU bodies are header-only inline-OpenMP (see the if
# constexpr(kHaveGpu) dispatch in LeapfrogStepper/BeamDiagnostics.hpp),
# instantiated at the call sites; no .cpp TU on CPU builds.
if(CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
  set(_NBODY_KERNEL_SOURCES
    helpers/primitives_gpu.cu)
  if(CMAKE_HIP_COMPILER AND NOT CMAKE_CUDA_COMPILER)
    set_source_files_properties(${_NBODY_KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)
  endif()
  set(_NBODY_BACKEND_LIBS ippl::cstone_gpu ippl::ryoanji)
else()
  set(_NBODY_KERNEL_SOURCES)
  set(_NBODY_BACKEND_LIBS ippl::cstone_headers ippl::ryoanji_headers OpenMP::OpenMP_CXX)
endif()

add_library(ippl_nbody STATIC ${_NBODY_COMMON_SOURCES} ${_NBODY_KERNEL_SOURCES})

target_include_directories(ippl_nbody PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ippl>)

target_compile_features(ippl_nbody PUBLIC cxx_std_20)

# Links ippl: this layer calls IpplTimings and builds on IPPL's Manager/Inform,
# and inherits the PUBLIC IPPL_ENABLE_TIMER_FENCES definition (set in
# PlatformOptions.cmake) so Timer.h is not compiled with its no-fences #warning.
target_link_libraries(ippl_nbody
  PUBLIC ippl
         ippl::nbody_sync
         ${_NBODY_BACKEND_LIBS}
         Kokkos::kokkos)

# Note: CUDA_SEPARABLE_COMPILATION is intentionally NOT enabled on ippl_nbody.
# Its consumer TUs (NBodySolver.cpp via IpplTimings, the diagnostics kernels)
# pull in <Kokkos_Core.hpp>, and Kokkos' desul atomics emits a hard `#error
# Relocatable device code mode incompatible with desul atomics configuration`
# unless Kokkos was built with Kokkos_ENABLE_RELOCATABLE_DEVICE_CODE=ON.
# ippl_nbody has no device-side cross-TU calls that would need RDC.
set_target_properties(ippl_nbody PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(ippl::nbody ALIAS ippl_nbody)

install(FILES NBodyParticleContainer.hpp
              NBodySolver.hpp
              NBodyManager.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody)

install(FILES core/Accelerator.hpp
              core/BHPrecision.hpp
              core/BHFieldLists.hpp
              core/ParticleAttribBase.hpp
              core/ParticleAttrib.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody/core)

install(FILES physics/LeapfrogStepper.hpp
              physics/BeamDiagnostics.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody/physics)

install(FILES wrappers/GravityWrapper.hpp
              wrappers/NBodyKokkosView.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody/wrappers)

install(FILES helpers/primitives_gpu.h
              helpers/GpuTimer.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ippl/NBody/helpers)
