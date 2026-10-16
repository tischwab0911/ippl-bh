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
 * @brief BH solver for coulomb interactions in NBody simulations
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#include "NBody/NBodySolver.hpp"

#include "NBody/core/Accelerator.hpp"
#include "NBody/helpers/GpuTimer.hpp"
#include "NBody/wrappers/GravityWrapper.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <mpi.h>

#include "Utility/IpplTimings.h"

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"

namespace ippl::nbody {

// Mirrors sphexa main/src/propagator/nbody.hpp::computeForces:
//   zero accel -> computeSpatialGroups -> upsweep -> traverse (BH + Ewald).
template <class P, unsigned Dim>
class NBodySolver<P, Dim>::Impl {
public:
    using Container = typename NBodySolver<P, Dim>::Container;
    using Tmm       = typename P::Tmm;
    using DomainT   = typename Container::DomainT;
    using HolderQ   = MultipoleHolder<ryoanji::CartesianQuadrupole<Tmm>, DomainT, Container, NBodyAcc>;
    using HolderMDQ = MultipoleHolder<ryoanji::CartesianMDQpole<Tmm>, DomainT, Container, NBodyAcc>;

    using Holders   = std::variant<HolderQ, HolderMDQ>;

    Impl(Container& pc, Params params)
        : pc_(pc), params_(params), mHolder_(makeHolder(params.multipoles)) {}

    // Constructs the selected holder in place (guaranteed copy elision, no move needed).
    static Holders makeHolder(MultipoleOrder order) {
        if (order == MultipoleOrder::DipoleQuadrupole) {
            return Holders(std::in_place_type<HolderMDQ>);
        }
        return Holders(std::in_place_type<HolderQ>);
    }

    Container& pc_;
    Params     params_;
    Holders    mHolder_;
};

template <class P, unsigned Dim>
NBodySolver<P, Dim>::NBodySolver(Container& pc, Params params) {
    if (params.multipoles == MultipoleOrder::DipoleQuadrupole
        && pc.domain().box().boundaryX() == cstone::BoundaryType::periodic) {
        throw std::invalid_argument(
            "NBodySolver: dipole multipoles do not support periodic boxes (ryoanji's Ewald "
            "summation exists for quadrupole multipoles only).");
    }
    impl_ = std::make_unique<Impl>(pc, params);
}

template <class P, unsigned Dim>
NBodySolver<P, Dim>::~NBodySolver() = default;

template <class P, unsigned Dim>
NBodySolver<P, Dim>::NBodySolver(NBodySolver&&) noexcept = default;

template <class P, unsigned Dim>
NBodySolver<P, Dim>&
NBodySolver<P, Dim>::operator=(NBodySolver&&) noexcept = default;

template <class P, unsigned Dim>
void NBodySolver<P, Dim>::runSolver(bool warmup) {
    auto& s      = *impl_;
    auto& pc     = s.pc_;
    auto& params = s.params_;
    using Ta     = typename P::Ta;

    static IpplTimings::TimerRef tUpswp  = IpplTimings::getTimer("bh.upsweep");
    static IpplTimings::TimerRef tGroups = IpplTimings::getTimer("bh.groups");
    static IpplTimings::TimerRef tZeroE  = IpplTimings::getTimer("bh.zeroE");
    static IpplTimings::TimerRef tBH     = IpplTimings::getTimer("bh.compute");

    const bool     collect = !warmup;
    auto&          domain  = pc.domain();
    const unsigned start   = pc.startIndex();
    const unsigned end     = pc.endIndex();
    if (end == start) { return; }

    // Gravitational prefactor read by the holder (d.g). The BH near-field
    // lattice extent and the Ewald shell count are the same number; bind both
    // to params.numShells so behavior matches the pre-portability solver
    // regardless of how a driver fills ewaldSettings.
    pc.g                            = params.G;
    auto ewald                      = params.ewaldSettings;
    ewald.numReplicaShells          = params.numShells;

    {
        GpuTimer t(tZeroE, collect);
        cstone::fill(kExec, pc.ax.data() + start, pc.ax.data() + end, Ta(0));
        cstone::fill(kExec, pc.ay.data() + start, pc.ay.data() + end, Ta(0));
        cstone::fill(kExec, pc.az.data() + start, pc.az.data() + end, Ta(0));
    }

    // Same sequence for either multipole type (params.multipoles picked the holder).
    // numGroups: traversal target groups (0 on the CPU)
    unsigned long long numGroups = 0;
    auto stats = std::visit(
        [&](auto& holder) {
            holder.setEwaldSettings(ewald);
            cstone::GroupView grp;
            { GpuTimer t(tGroups, collect); grp = holder.computeSpatialGroups(pc, domain); }
            numGroups = grp.numGroups;
            { GpuTimer t(tUpswp,  collect); holder.upsweep(pc, domain); }
            { GpuTimer t(tBH,     collect); holder.traverse(grp, pc, domain); }
            return holder.readStats();
        },
        s.mHolder_);

    // sphexa gravity_wrapper guard: ryoanji signals traversal-stack exhaustion by
    // setting maxP2P = 0xFFFFFFFF. Without this check the kernel returns with
    // corrupt index data and the next syncGrav deadlocks on bad SFC keys. The CPU
    // holder returns zeroed stats, so this is a no-op there.
    if (stats[1] == 0xFFFFFFFFull) {
        int mpiRank = -1;
        MPI_Comm_rank(pc.comm(), &mpiRank);
        throw std::runtime_error(
            "Barnes-Hut traversal stack exhausted on rank " + std::to_string(mpiRank) +
            ". Raise theta, reduce numShells, or shrink particle count per rank.");
    }

    // Per-step BH stats — global aggregates across ranks for direct comparison
    // with sphexa's timer.logStatistics output. Skipped during the pre_run warmup
    // solve so the printed counts reflect only the timed steps.
    if (collect) {
        // Owned focus-leaf occupancy (host copy of the layout): count, max and sum(c^2)/sum(c), the
        // leaf size around an average particle. Far above the bucket size: tree updates lag.
        unsigned long long numLeaves = 0, maxLeaf = 0, sumSqLeaf = 0;
        {
            const auto     layout    = domain.layout();
            const auto     firstCell = static_cast<std::size_t>(domain.startCell());
            const auto     lastCell  = static_cast<std::size_t>(domain.endCell());
            std::vector<cstone::LocalIndex> hostLayout(lastCell - firstCell + 1);
#if defined(USE_CUDA)
            cstone::memcpyD2HAsync(kExec, layout.data() + firstCell, hostLayout.size(),
                                   hostLayout.data());
            syncExec();
#else
            std::copy_n(layout.data() + firstCell, hostLayout.size(), hostLayout.begin());
#endif
            numLeaves = lastCell - firstCell;
            for (std::size_t i = 0; i + 1 < hostLayout.size(); ++i) {
                const unsigned long long c = hostLayout[i + 1] - hostLayout[i];
                maxLeaf                    = std::max(maxLeaf, c);
                sumSqLeaf += c * c;
            }
        }

        unsigned long long local[7] = {
            static_cast<unsigned long long>(stats[0]),   // sumP2P
            static_cast<unsigned long long>(stats[2]),   // sumM2P
            numGroups,                                   // target groups
            static_cast<unsigned long long>(end - start),  // targets
            static_cast<unsigned long long>(stats[1]),   // maxP2P
            static_cast<unsigned long long>(stats[3]),   // maxM2P
            static_cast<unsigned long long>(stats[4])};  // maxStack
        unsigned long long sum[4]   = {0, 0, 0, 0};
        unsigned long long mx[3]    = {0, 0, 0};
        unsigned long long leafS[2] = {numLeaves, sumSqLeaf};
        unsigned long long leafSum[2] = {0, 0}, leafMax = 0;
        MPI_Allreduce(&local[0], &sum[0], 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, pc.comm());
        MPI_Allreduce(&local[4], &mx[0], 3, MPI_UNSIGNED_LONG_LONG, MPI_MAX, pc.comm());
        MPI_Allreduce(leafS, leafSum, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, pc.comm());
        MPI_Allreduce(&maxLeaf, &leafMax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, pc.comm());
        int mpiRank = -1;
        MPI_Comm_rank(pc.comm(), &mpiRank);
        if (mpiRank == 0) {
            const double weightedLeaf =
                sum[3] > 0 ? static_cast<double>(leafSum[1]) / static_cast<double>(sum[3]) : 0.0;
            std::printf("[bh.stats] sumP2P=%llu sumM2P=%llu maxP2P=%llu maxM2P=%llu maxStack=%llu "
                        "groups=%llu targets=%llu leaves=%llu maxLeaf=%llu wLeaf=%.1f\n",
                        sum[0], sum[1], mx[0], mx[1], mx[2], sum[2], sum[3], leafSum[0], leafMax,
                        weightedLeaf);
            std::fflush(stdout);
        }
    }
}

// Explicit instantiations — one per precision policy. Each maps to a
// pre-compiled ryoanji TRAVERSE_MPOLE + COMPUTE_GRAVITY_EWALD_GPU row (GPU) or
// the header-only ryoanji CPU path.
template class NBodySolver<DoublePrecision, 3>;
template class NBodySolver<MixedPrecision,  3>;
template class NBodySolver<FloatPrecision,  3>;

}  // namespace ippl::nbody
