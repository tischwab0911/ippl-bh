/*
 * IPPL Barnes-Hut
 * 
 * Copyright (c) 2026 CSCS, ETH Zurich
 *               2026 PSI, Villigen
 * 
 * Please refer to the LICENSE file in the root directory
 * SPDX-License-Identifier: GPL-3.0
 */

/*! @file
 * @brief Realize the configure-time GPU/CPU choice
 * 
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <type_traits>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"
#include "cstone/cuda/device_vector.h"
#include "cstone/execution.hpp"
#include "cstone/primitives/primitives_acc.hpp"

namespace ippl::nbody {

// Build-global execution policy, chosen at configure time exactly like sphexa's
// USE_CUDA-driven AccType. The vendored cstone_gpu target defines USE_CUDA
// PUBLIC, so any GPU build (CUDA or HIP) sees it; the CPU build links only
// cstone_headers and never defines it. kExec is the policy object cstone's
// primitives take as first argument: the default GPU stream, or the CPU tag.
#if defined(USE_CUDA)
using NBodyAcc = cstone::execution::Gpu;
inline constexpr NBodyAcc kExec = cstone::execution::gpuDefaultStream;
#else
using NBodyAcc = cstone::execution::Cpu;
inline constexpr NBodyAcc kExec = cstone::execution::cpu;
#endif

// This inline expression is used to select between CPU and GPU implementations
inline constexpr bool kHaveGpu = bool(cstone::execution::HaveGpu<NBodyAcc>{});

// Block until all work queued on kExec has finished; a no-op on the CPU build,
// whose cstone/ryoanji kernels are synchronous. (cstone::syncGpu only accepts a
// GPU stream, so it cannot sit in a discarded if-constexpr branch of a CPU build.)
inline void syncExec() {
#if defined(USE_CUDA)
    cstone::syncGpu(kExec);
#endif
}

// Per-component field storage: GPU-resident DeviceVector on device builds, plain
// std::vector on the CPU build. Mirrors sphexa ParticlesData::FieldVector.
template <class T>
using FieldVector =
    std::conditional_t<kHaveGpu, cstone::DeviceVector<T>, std::vector<T>>;

}  // namespace ippl::nbody
