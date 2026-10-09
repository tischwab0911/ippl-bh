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
 * @brief GPU kernels implementing functionality of NBodyParticleContainer
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#include "NBody/NBodyParticleContainer.hpp"

#include <cstdint>

#include "cstone/sfc/common.hpp"
#include "cstone/tree/csarray.hpp"

namespace ippl::nbody {
namespace detail {

namespace {

template <class KeyType, class Th>
__global__ void setHFromLeafKernel(const KeyType*   keys,
                                   const KeyType*   leaves,
                                   int              nLeafKeys,
                                   Th               cbrtVol,
                                   cstone::AxesBits reductions,
                                   Th*              h,
                                   unsigned         n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    KeyType               k         = keys[i];
    cstone::TreeNodeIndex leafIdx   = cstone::findNodeBelow(leaves, nLeafKeys, k);
    KeyType               codeRange = leaves[leafIdx + 1] - leaves[leafIdx];
    unsigned              level     = cstone::treeLevel(codeRange);
    h[i] = cellEdge(cbrtVol, level, reductions);
}

}  // namespace

template <class Th, class KeyType>
void setHFromLeavesGpu(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                       Th cbrtVol, cstone::AxesBits reductions, Th* h, unsigned n) {
    if (n == 0) return;
    const unsigned blockSize = 256u;
    const unsigned numBlocks = (n + blockSize - 1u) / blockSize;
    setHFromLeafKernel<KeyType, Th>
        <<<numBlocks, blockSize>>>(keys, leaves, nLeafKeys, cbrtVol, reductions, h, n);
}

template void setHFromLeavesGpu<double, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, double, cstone::AxesBits, double*, unsigned);
template void setHFromLeavesGpu<float, std::uint64_t>(
    const std::uint64_t*, const std::uint64_t*, int, float, cstone::AxesBits, float*, unsigned);

namespace {

constexpr unsigned kBlockSize = 256u;

inline unsigned gridFor(unsigned n) { return (n + kBlockSize - 1u) / kBlockSize; }

template <class KeyType>
__global__ void markRemovedKernel(KeyType* keys, const bool* flags, unsigned n, KeyType sentinel) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && flags[i]) { keys[i] = sentinel; }
}

template <class IdType>
__global__ void assignIdsKernel(IdType* id, unsigned first, unsigned n, IdType nextId, IdType stride) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) { id[first + k] = nextId + stride * static_cast<IdType>(k); }
}

}  // namespace

template <class KeyType>
void markRemovedGpu(KeyType* keys, const bool* flags, unsigned n, KeyType sentinel) {
    if (n == 0u) { return; }
    markRemovedKernel<KeyType><<<gridFor(n), kBlockSize>>>(keys, flags, n, sentinel);
}

template <class IdType>
void assignIdsGpu(IdType* id, unsigned first, unsigned n, IdType nextId, IdType stride) {
    if (n == 0u) { return; }
    assignIdsKernel<IdType><<<gridFor(n), kBlockSize>>>(id, first, n, nextId, stride);
}

template void markRemovedGpu<std::uint64_t>(std::uint64_t*, const bool*, unsigned, std::uint64_t);
template void assignIdsGpu<std::uint32_t>(std::uint32_t*, unsigned, unsigned, std::uint32_t, std::uint32_t);

}  // namespace detail
}  // namespace ippl::nbody
