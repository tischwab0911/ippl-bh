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
 * @brief cstone-style GPU reduction and element-wise primitives for the NBody kernels
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#include "NBody/helpers/primitives_gpu.h"

#include <cstdint>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "cstone/sfc/box.hpp"

namespace ippl::nbody::detail {

namespace {

// Element-wise sum of two per-axis tuples; the reduce operator for both primitives.
template <class T>
struct TuplePlus3 {
    __host__ __device__ thrust::tuple<T, T, T>
    operator()(const thrust::tuple<T, T, T>& a, const thrust::tuple<T, T, T>& b) const {
        return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                  thrust::get<1>(a) + thrust::get<1>(b),
                                  thrust::get<2>(a) + thrust::get<2>(b));
    }
};

// Squared deviation of a zipped (vx,vy,vz) from the mean; the transform for the
// variance primitive.
template <class T>
struct VelMinusMeanSqFunctor {
    T meanVx, meanVy, meanVz;
    __host__ __device__ thrust::tuple<T, T, T>
    operator()(const thrust::tuple<T, T, T>& v) const {
        T dx = thrust::get<0>(v) - meanVx;
        T dy = thrust::get<1>(v) - meanVy;
        T dz = thrust::get<2>(v) - meanVz;
        return thrust::make_tuple(dx * dx, dy * dy, dz * dz);
    }
};

}  // namespace

template <class T>
Triple<T> sumTripleGpu(const T* x, const T* y, const T* z, std::size_t n) {
    auto first = thrust::make_zip_iterator(x, y, z);
    auto last  = thrust::make_zip_iterator(x + n, y + n, z + n);
    auto s = thrust::reduce(thrust::device, first, last,
                            thrust::make_tuple(T(0), T(0), T(0)), TuplePlus3<T>{});
    return Triple<T>{thrust::get<0>(s), thrust::get<1>(s), thrust::get<2>(s)};
}

template <class T>
Triple<T> varianceTripleGpu(const T* x, const T* y, const T* z, std::size_t n,
                            T mx, T my, T mz) {
    auto first = thrust::make_zip_iterator(x, y, z);
    auto last  = thrust::make_zip_iterator(x + n, y + n, z + n);
    auto s = thrust::transform_reduce(thrust::device, first, last,
                                      VelMinusMeanSqFunctor<T>{mx, my, mz},
                                      thrust::make_tuple(T(0), T(0), T(0)), TuplePlus3<T>{});
    return Triple<T>{thrust::get<0>(s), thrust::get<1>(s), thrust::get<2>(s)};
}

template Triple<double> sumTripleGpu(const double*, const double*, const double*, std::size_t);
template Triple<float>  sumTripleGpu(const float*,  const float*,  const float*,  std::size_t);
template Triple<double> varianceTripleGpu(const double*, const double*, const double*, std::size_t,
                                          double, double, double);
template Triple<float>  varianceTripleGpu(const float*, const float*, const float*, std::size_t,
                                          float, float, float);

// --- element-wise maps ------------------------------------------------------

namespace {

constexpr unsigned kBlockSize = 256u;

inline unsigned gridFor(unsigned n) { return (n + kBlockSize - 1u) / kBlockSize; }

template <class Tout, class Tin>
__global__ void axpy3Kernel(unsigned start, unsigned n,
                            Tout* __restrict__ ax, Tout* __restrict__ ay, Tout* __restrict__ az,
                            const Tin* __restrict__ xx, const Tin* __restrict__ xy,
                            const Tin* __restrict__ xz, Tout s) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    ax[i] += s * static_cast<Tout>(xx[i]);
    ay[i] += s * static_cast<Tout>(xy[i]);
    az[i] += s * static_cast<Tout>(xz[i]);
}

template <class T>
__global__ void driftWrapKernel(unsigned start, unsigned n,
                                T* __restrict__ Rx, T* __restrict__ Ry, T* __restrict__ Rz,
                                const T* __restrict__ Px, const T* __restrict__ Py,
                                const T* __restrict__ Pz, T dt, cstone::Box<T> box) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = start + tid;
    cstone::Vec3<T> X{Rx[i] + dt * Px[i], Ry[i] + dt * Py[i], Rz[i] + dt * Pz[i]};
    X = cstone::putInBox(X, box);
    Rx[i] = X[0];
    Ry[i] = X[1];
    Rz[i] = X[2];
}

template <class T>
__global__ void checkPositionsKernel(const T* __restrict__ x, const T* __restrict__ y,
                                     const T* __restrict__ z, unsigned first, unsigned n,
                                     cstone::Box<T> box, unsigned long long* counts) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) { return; }
    unsigned i = first + tid;
    const T xi = x[i], yi = y[i], zi = z[i];
    if (!isfinite(xi) || !isfinite(yi) || !isfinite(zi)) {
        atomicAdd(&counts[0], 1ull);
    } else if (xi < box.xmin() || xi > box.xmax() || yi < box.ymin() || yi > box.ymax()
               || zi < box.zmin() || zi > box.zmax()) {
        atomicAdd(&counts[1], 1ull);
    }
}

template <class T>
__global__ void groupExtentsKernel(const T* __restrict__ x, const T* __restrict__ y,
                                   const T* __restrict__ z, const unsigned* __restrict__ groupStart,
                                   const unsigned* __restrict__ groupEnd, unsigned numGroups,
                                   float* __restrict__ extents) {
    unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= numGroups) { return; }
    const unsigned first = groupStart[g], last = groupEnd[g];
    if (last <= first) {
        extents[g] = 0.0f;
        return;
    }
    T lo[3] = {x[first], y[first], z[first]}, hi[3] = {lo[0], lo[1], lo[2]};
    for (unsigned i = first + 1; i < last; ++i) {
        const T p[3] = {x[i], y[i], z[i]};
        for (int d = 0; d < 3; ++d) {
            lo[d] = p[d] < lo[d] ? p[d] : lo[d];
            hi[d] = p[d] > hi[d] ? p[d] : hi[d];
        }
    }
    T edge = hi[0] - lo[0];
    edge   = hi[1] - lo[1] > edge ? hi[1] - lo[1] : edge;
    edge   = hi[2] - lo[2] > edge ? hi[2] - lo[2] : edge;
    extents[g] = static_cast<float>(edge);  // NaN coordinates propagate as NaN
}

}  // namespace

template <class T>
PositionCheck checkPositionsGpu(const T* x, const T* y, const T* z, unsigned first, unsigned last,
                                cstone::Box<T> box) {
    PositionCheck result;
    if (last <= first) { return result; }
    thrust::device_vector<unsigned long long> counts(2, 0ull);
    checkPositionsKernel<T><<<gridFor(last - first), kBlockSize>>>(
        x, y, z, first, last - first, box, thrust::raw_pointer_cast(counts.data()));
    result.nonFinite = counts[0];
    result.outside   = counts[1];
    return result;
}

template <class T>
void groupExtentsGpu(const T* x, const T* y, const T* z, const unsigned* groupStart,
                     const unsigned* groupEnd, unsigned numGroups, float* extents) {
    if (numGroups == 0u) { return; }
    groupExtentsKernel<T><<<gridFor(numGroups), kBlockSize>>>(x, y, z, groupStart, groupEnd,
                                                             numGroups, extents);
}

template <class Tout, class Tin>
void axpy3Gpu(unsigned start, unsigned n,
              Tout* ax, Tout* ay, Tout* az,
              const Tin* xx, const Tin* xy, const Tin* xz, Tout s) {
    if (n == 0u) { return; }
    axpy3Kernel<Tout, Tin><<<gridFor(n), kBlockSize>>>(start, n, ax, ay, az, xx, xy, xz, s);
}

template <class T>
void driftWrapGpu(unsigned start, unsigned n, T* Rx, T* Ry, T* Rz,
                  const T* Px, const T* Py, const T* Pz, T dt, cstone::Box<T> box) {
    if (n == 0u) { return; }
    driftWrapKernel<T><<<gridFor(n), kBlockSize>>>(start, n, Rx, Ry, Rz, Px, Py, Pz, dt, box);
}

// Out-of-place right shift via a device temporary: copy [first,last) to tmp, then
// tmp to [first+by, …). Avoids the overlapping in-place device memcpy that a direct
// shift would incur. Container-agnostic — any device array.
template <class T>
void shiftRightGpu(T* data, unsigned first, unsigned last, unsigned by) {
    if (last <= first || by == 0u) { return; }
    const unsigned n = last - first;
    thrust::device_vector<T> tmp(n);
    thrust::copy(thrust::device, thrust::device_pointer_cast(data + first),
                 thrust::device_pointer_cast(data + last), tmp.begin());
    thrust::copy(thrust::device, tmp.begin(), tmp.end(),
                 thrust::device_pointer_cast(data + first + by));
}

template void axpy3Gpu<double, double>(unsigned, unsigned, double*, double*, double*,
                                       const double*, const double*, const double*, double);
template void axpy3Gpu<double, float> (unsigned, unsigned, double*, double*, double*,
                                       const float*,  const float*,  const float*,  double);
template void axpy3Gpu<float,  float> (unsigned, unsigned, float*,  float*,  float*,
                                       const float*,  const float*,  const float*,  float);
template void driftWrapGpu<double>(unsigned, unsigned, double*, double*, double*,
                                   const double*, const double*, const double*, double,
                                   cstone::Box<double>);
template void driftWrapGpu<float> (unsigned, unsigned, float*,  float*,  float*,
                                   const float*,  const float*,  const float*,  float,
                                   cstone::Box<float>);
template PositionCheck checkPositionsGpu<double>(const double*, const double*, const double*,
                                                 unsigned, unsigned, cstone::Box<double>);
template PositionCheck checkPositionsGpu<float>(const float*, const float*, const float*, unsigned,
                                                unsigned, cstone::Box<float>);
template void groupExtentsGpu<double>(const double*, const double*, const double*,
                                      const unsigned*, const unsigned*, unsigned, float*);
template void groupExtentsGpu<float>(const float*, const float*, const float*, const unsigned*,
                                     const unsigned*, unsigned, float*);
template void shiftRightGpu<double>(double*, unsigned, unsigned, unsigned);
template void shiftRightGpu<float> (float*,  unsigned, unsigned, unsigned);
template void shiftRightGpu<std::uint32_t>(std::uint32_t*, unsigned, unsigned, unsigned);
template void shiftRightGpu<std::uint64_t>(std::uint64_t*, unsigned, unsigned, unsigned);

}  // namespace ippl::nbody::detail
