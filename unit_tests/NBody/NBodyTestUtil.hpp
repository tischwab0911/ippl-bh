#ifndef IPPL_NBODY_TEST_UTIL_HPP
#define IPPL_NBODY_TEST_UTIL_HPP

// Backend-agnostic host<->storage transfer helpers for the NBody unit tests.
// On GPU builds they use cstone's memcpy wrappers (which dispatch to CUDA/HIP)
// on the default stream and wait; on the CPU build they are plain std::copy over
// std::vector storage. This lets the container/leapfrog/wrap tests run unchanged
// on either backend.

#include <algorithm>
#include <cstddef>
#include <vector>

#include "cstone/cuda/cuda_utils.hpp"  // memcpyH2DAsync / memcpyD2HAsync

#include "NBody/core/Accelerator.hpp"  // kExec, syncExec

namespace ippl::nbody::test {

template <class T>
void uploadHost(const std::vector<T>& host, T* dst) {
    if (host.empty()) { return; }
#if defined(USE_CUDA)
    cstone::memcpyH2DAsync(kExec, host.data(), host.size(), dst);
    syncExec();
#else
    std::copy(host.begin(), host.end(), dst);
#endif
}

template <class T>
void downloadDevice(const T* src, std::size_t n, std::vector<T>& host) {
    host.resize(n);
    if (n == 0) { return; }
#if defined(USE_CUDA)
    cstone::memcpyD2HAsync(kExec, src, n, host.data());
    syncExec();
#else
    std::copy(src, src + n, host.begin());
#endif
}

}  // namespace ippl::nbody::test

#endif  // IPPL_NBODY_TEST_UTIL_HPP
