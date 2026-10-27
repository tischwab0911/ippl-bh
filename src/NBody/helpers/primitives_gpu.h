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
 * @brief Per-axis triple type, cstone-style GPU reduction and element-wise primitives
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <mpi.h>

#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/sfc/box.hpp"

#include "NBody/core/Accelerator.hpp"

namespace ippl::nbody {

// Per-axis triple of the same type. Used as the result type of mean-velocity
// and per-axis variance reductions, and exposed publicly so simulation-specific
// diagnostics can build on it (e.g. DIH's reduceMeanAbsAccel).
template <class T>
struct Triple {
    T x;
    T y;
    T z;
};

namespace detail {

// Per-axis sum + local count -> global mean. Shared by both backends.
template <class T>
inline Triple<T> finalizeMeanTriple(Triple<T> localSum, unsigned long localN, MPI_Comm comm) {
    T inout[4] = {localSum.x, localSum.y, localSum.z, static_cast<T>(localN)};
    MPI_Allreduce(MPI_IN_PLACE, inout, 4, ::MpiType<T>{}, MPI_SUM, comm);
    const T invN = (inout[3] > T(0)) ? T(1) / inout[3] : T(0);
    return Triple<T>{inout[0] * invN, inout[1] * invN, inout[2] * invN};
}

// cstone-style GPU primitives: raw pointers in, local per-axis triple out, no MPI.
// Defined + instantiated in primitives_gpu.cu.
template <class T>
Triple<T> sumTripleGpu(const T* x, const T* y, const T* z, std::size_t n);

template <class T>
Triple<T> varianceTripleGpu(const T* x, const T* y, const T* z, std::size_t n,
                            T mx, T my, T mz);

// Element-wise GPU maps over the index range [start, start+n). Defined +
// instantiated in primitives_gpu.cu; CPU counterparts + dispatch live in the
// consumer header (LeapfrogStepper.hpp).

// axpy over three arrays: a[i] += s * static_cast<Tout>(x[i]). Backs the leapfrog
// kick (Tout=Tc, Tin=Ta, s=-half/full dt).
template <class Tout, class Tin>
void axpy3Gpu(unsigned start, unsigned n,
              Tout* ax, Tout* ay, Tout* az,
              const Tin* xx, const Tin* xy, const Tin* xz, Tout s);

// Leapfrog drift + periodic wrap: (Rx,Ry,Rz)[i] = cstone::putInBox(R[i] + dt*P[i], box).
// Open-BC axes are passed through unchanged by putInBox.
template <class T>
void driftWrapGpu(unsigned start, unsigned n, T* Rx, T* Ry, T* Rz,
                  const T* Px, const T* Py, const T* Pz, T dt, cstone::Box<T> box);

// Overlap-safe right shift: move [first, last) to [first+by, last+by). The GPU
// path copies via a device temporary (no overlapping in-place device memcpy); the
// CPU path uses std::copy_backward. shiftRightGpu is defined + instantiated
// (double, float, uint32_t, uint64_t) in primitives_gpu.cu.
template <class T>
void shiftRightGpu(T* data, unsigned first, unsigned last, unsigned by);

// Consistency checks of the Barnes-Hut sources (NBodySolver: info level 3, and always when the
// traversal fails). Defined + instantiated (double, float) in primitives_gpu.cu.
struct PositionCheck {
    unsigned long long nonFinite = 0;  // particles with a non-finite coordinate
    unsigned long long outside   = 0;  // finite particles outside the domain box
};

// Particles [first, last) against the box.
template <class T>
PositionCheck checkPositionsGpu(const T* x, const T* y, const T* z, unsigned first, unsigned last,
                                cstone::Box<T> box);

// Largest edge of the bounding box of every target group [groupStart[g], groupEnd[g]), into
// extents[g] (device memory).
template <class T>
void groupExtentsGpu(const T* x, const T* y, const T* z, const unsigned* groupStart,
                     const unsigned* groupEnd, unsigned numGroups, float* extents);

template <class T>
void shiftRight(T* data, unsigned first, unsigned last, unsigned by) {
    if (last <= first || by == 0u) { return; }
    if constexpr (kHaveGpu) { shiftRightGpu(data, first, last, by); }
    else { std::copy_backward(data + first, data + last, data + last + by); }
}

}  // namespace detail
}  // namespace ippl::nbody
