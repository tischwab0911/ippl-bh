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
#include "NBody/helpers/primitives_gpu.h"
#include "NBody/wrappers/GravityWrapper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include <mpi.h>

#include "Utility/IpplTimings.h"

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/traversal/groups.hpp"
#include "ryoanji/nbody/cartesian_qpole.hpp"

namespace ippl::nbody {

namespace {

// Device array -> host vector (a copy on the CPU build).
template <class T>
std::vector<T> toHost(const T* src, std::size_t n) {
    std::vector<T> host(n);
    if (n == 0) { return host; }
#if defined(USE_CUDA)
    cstone::memcpyD2HAsync(kExec, src, n, host.data());
    syncExec();
#else
    std::copy_n(src, n, host.begin());
#endif
    return host;
}

// Owned focus-leaf occupancy (host copy of the layout): count, max and sum(c^2).
struct LeafOccupancy {
    unsigned long long numLeaves = 0, maxLeaf = 0, sumSq = 0;
};

template <class Domain>
LeafOccupancy leafOccupancy(const Domain& domain) {
    const auto firstCell = static_cast<std::size_t>(domain.startCell());
    const auto lastCell  = static_cast<std::size_t>(domain.endCell());
    const std::vector<cstone::LocalIndex> layout =
        toHost(domain.layout().data() + firstCell, lastCell - firstCell + 1);
    LeafOccupancy o;
    o.numLeaves = lastCell - firstCell;
    for (std::size_t i = 0; i + 1 < layout.size(); ++i) {
        const unsigned long long c = layout[i + 1] - layout[i];
        o.maxLeaf                  = std::max(o.maxLeaf, c);
        o.sumSq += c * c;
    }
    return o;
}

// Consistency of the sources of one solve: owned and halo positions against the domain box, and
// the spatial extent of the target groups. A group far larger than its peers makes almost every
// cell fail the opening criterion and overflows the traversal stack. Rank-local, no MPI.
struct SourceCheck {
    detail::PositionCheck owned, halos;
    unsigned numGroups  = 0;
    float medianExtent  = 0.0f, maxExtent = 0.0f;  // largest bounding-box edge of a group
    unsigned worstGroup = 0, worstFirst = 0, worstLast = 0;
};

template <class Container, class Box>
detail::PositionCheck checkPositions(const Container& pc, unsigned first, unsigned last,
                                     const Box& box) {
    if constexpr (kHaveGpu) {
        return detail::checkPositionsGpu(pc.x.data(), pc.y.data(), pc.z.data(), first, last, box);
    } else {
        detail::PositionCheck c;
        for (unsigned i = first; i < last; ++i) {
            const auto x = pc.x[i], y = pc.y[i], z = pc.z[i];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                ++c.nonFinite;
            } else if (x < box.xmin() || x > box.xmax() || y < box.ymin() || y > box.ymax()
                       || z < box.zmin() || z > box.zmax()) {
                ++c.outside;
            }
        }
        return c;
    }
}

template <class Container, class Domain>
SourceCheck checkSources(const Container& pc, const Domain& domain, const cstone::GroupView& grp) {
    SourceCheck c;
    const auto box = domain.box();
    c.owned        = checkPositions(pc, pc.startIndex(), pc.endIndex(), box);
    c.halos        = checkPositions(pc, 0, pc.startIndex(), box);
    const auto h2  = checkPositions(pc, pc.endIndex(), pc.nWithHalos(), box);
    c.halos.nonFinite += h2.nonFinite;
    c.halos.outside += h2.outside;

    c.numGroups = grp.numGroups;
    if constexpr (kHaveGpu) {
        if (grp.numGroups == 0) { return c; }
        FieldVector<float> extents(grp.numGroups);
        detail::groupExtentsGpu(pc.x.data(), pc.y.data(), pc.z.data(), grp.groupStart,
                                grp.groupEnd, grp.numGroups, extents.data());
        std::vector<float> e = toHost(extents.data(), extents.size());
        for (unsigned g = 0; g < e.size(); ++g) {
            const float v = std::isnan(e[g]) ? INFINITY : e[g];
            if (v > c.maxExtent || g == 0) {
                c.maxExtent  = v;
                c.worstGroup = g;
            }
        }
        std::nth_element(e.begin(), e.begin() + e.size() / 2, e.end());
        c.medianExtent = e[e.size() / 2];
        c.worstFirst   = toHost(grp.groupStart + c.worstGroup, 1)[0];
        c.worstLast    = toHost(grp.groupEnd + c.worstGroup, 1)[0];
    }
    return c;
}

// Rank-local report of a failed traversal (stderr, no MPI: the other ranks did not fail).
template <class Container, class Domain>
void reportTraversalFailure(const Container& pc, const Domain& domain, const cstone::GroupView& grp,
                            int rank, unsigned maxStack) {
    const SourceCheck c     = checkSources(pc, domain, grp);
    const LeafOccupancy occ = leafOccupancy(domain);
    const auto box          = domain.box();
    std::ostringstream out;
    out << "*** Barnes-Hut traversal stack exhausted on rank " << rank << " (stack depth " << maxStack
        << " entries)\n"
        << "    sources: owned " << pc.getLocalNum() << " [" << pc.startIndex() << ", "
        << pc.endIndex() << "), with halos " << pc.nWithHalos() << "; box x [" << box.xmin() << ", "
        << box.xmax() << "] y [" << box.ymin() << ", " << box.ymax() << "] z [" << box.zmin()
        << ", " << box.zmax() << "]\n"
        << "    positions: owned non-finite " << c.owned.nonFinite << ", outside the box "
        << c.owned.outside << "; halos non-finite " << c.halos.nonFinite << ", outside "
        << c.halos.outside << "\n"
        << "    focus tree: " << occ.numLeaves << " owned leaves, max leaf " << occ.maxLeaf << "\n"
        << "    target groups: " << c.numGroups << ", extent median " << c.medianExtent
        << ", max " << c.maxExtent << " (group " << c.worstGroup << ", particles ["
        << c.worstFirst << ", " << c.worstLast << "))\n";
    if (c.worstLast > c.worstFirst) {
        const unsigned n = std::min(c.worstLast - c.worstFirst, 16u);
        const auto x     = toHost(pc.x.data() + c.worstFirst, n);
        const auto y     = toHost(pc.y.data() + c.worstFirst, n);
        const auto z     = toHost(pc.z.data() + c.worstFirst, n);
        out << "    first " << n << " particles of that group (x y z):\n";
        for (unsigned i = 0; i < n; ++i) {
            out << "      " << x[i] << " " << y[i] << " " << z[i] << "\n";
        }
    }
    std::fprintf(stderr, "%s", out.str().c_str());
    std::fflush(stderr);
}

}  // namespace

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
    using HolderH   = HeavisideHolder<DomainT, Container, NBodyAcc>;

    using Holders   = std::variant<HolderQ, HolderMDQ, HolderH>;

    Impl(Container& pc, Params params)
        : pc_(pc), params_(params), mHolder_(makeHolder(params.multipoles)) {}

    // Constructs the selected holder in place (guaranteed copy elision, no move needed).
    static Holders makeHolder(MultipoleOrder order) {
        if (order == MultipoleOrder::DipoleQuadrupole) {
            return Holders(std::in_place_type<HolderMDQ>);
        }
        if (order == MultipoleOrder::Heaviside) {
            return Holders(std::in_place_type<HolderH>);
        }
        return Holders(std::in_place_type<HolderQ>);
    }

    Container& pc_;
    Params     params_;
    Holders    mHolder_;
    unsigned long long remoteEvents_ = 0;  // this rank, since the last takeRemoteEvents
};

template <class P, unsigned Dim>
NBodySolver<P, Dim>::NBodySolver(Container& pc, Params params) {
    if (params.multipoles == MultipoleOrder::DipoleQuadrupole
        && pc.domain().box().boundaryX() == cstone::BoundaryType::periodic) {
        throw std::invalid_argument(
            "NBodySolver: dipole multipoles do not support periodic boxes (ryoanji's Ewald "
            "summation exists for quadrupole multipoles only).");
    }
    if (params.multipoles == MultipoleOrder::Heaviside
        && pc.domain().box().boundaryX() == cstone::BoundaryType::periodic) {
        throw std::invalid_argument(
            "NBodySolver: per-particle velocities (Heaviside multipoles) support open boxes only.");
    }
    impl_ = std::make_unique<Impl>(pc, params);
}

template <class P, unsigned Dim>
void NBodySolver<P, Dim>::setHeavisideParams(const ryoanji::HeavisideParams<typename P::Tf>& params) {
    impl_->params_.heaviside = params;
    // The next syncGravBH (Heaviside field lists) gives cstone the same criterion, so the locally
    // essential tree, the halos and the traversal use one MAC radius per cell.
    using Tc = typename Container::DomainT::RealType;
    const auto vm = ryoanji::velocityMac(params.frame, params.thetaV, params.velocityMoments);
    cstone::VelocityMac<Tc> p;
    p.gammaRef          = Tc(vm.gammaRef);
    p.betaRef           = Tc(vm.betaRef);
    p.wRef              = Tc(vm.wRef);
    p.thetaV            = Tc(vm.thetaV);
    p.rapidityCriterion = vm.rapidityCriterion;
    p.active            = true;
    impl_->pc_.setVelocityMac(p);
}

template <class P, unsigned Dim>
unsigned long long NBodySolver<P, Dim>::takeRemoteEvents() {
    unsigned long long global = 0;
    MPI_Allreduce(&impl_->remoteEvents_, &global, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, impl_->pc_.comm());
    impl_->remoteEvents_ = 0;
    return global;
}

template <class P, unsigned Dim>
typename NBodySolver<P, Dim>::LetStatistics NBodySolver<P, Dim>::letStatistics() const {
    const auto& pc     = impl_->pc_;
    const auto  cells  = pc.domain().velocityMacStats();
    unsigned long long local[3] = {
        static_cast<unsigned long long>(pc.nWithHalos() - pc.getLocalNum()), cells[0], cells[1]};
    unsigned long long global[3] = {0, 0, 0};
    MPI_Allreduce(local, global, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, pc.comm());
    return {global[0], global[1], global[2]};
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
        if (params.multipoles == MultipoleOrder::Heaviside) {
            cstone::fill(kExec, pc.cx.data() + start, pc.cx.data() + end, Ta(0));
            cstone::fill(kExec, pc.cy.data() + start, pc.cy.data() + end, Ta(0));
        }
    }
    if (params.multipoles == MultipoleOrder::Heaviside) {
        // Halo sources need their rapidity for P2P; syncGrav moves w only with owned particles.
        pc.exchangeHalos(std::tie(pc.w), pc.haloSendBuf(), pc.haloRecvBuf());
    }

    // Same sequence for either multipole type (params.multipoles picked the holder).
    // numGroups: traversal target groups (0 on the CPU)
    unsigned long long numGroups = 0, retried = 0;
    cstone::GroupView  groups{};
    auto stats = std::visit(
        [&](auto& holder) {
            holder.setEwaldSettings(ewald);
            if constexpr (std::is_same_v<std::decay_t<decltype(holder)>, typename Impl::HolderH>) {
                holder.setHeavisideParams(params.heaviside);
            }
            cstone::GroupView grp;
            { GpuTimer t(tGroups, collect); grp = holder.computeSpatialGroups(pc, domain); }
            numGroups = grp.numGroups;
            groups    = grp;
            { GpuTimer t(tUpswp,  collect); holder.upsweep(pc, domain); }
            { GpuTimer t(tBH,     collect); holder.traverse(grp, pc, domain); }
            retried = holder.retriedGroups();
            return holder.readStats();
        },
        s.mHolder_);

    s.remoteEvents_ += std::visit([](auto& holder) { return holder.remoteEvents(); }, s.mHolder_);

    // sphexa gravity_wrapper guard: ryoanji signals traversal-stack exhaustion by
    // setting maxP2P = 0xFFFFFFFF. Without this check the kernel returns with
    // corrupt index data and the next syncGrav deadlocks on bad SFC keys. The CPU
    // holder returns zeroed stats, so this is a no-op there.
    if (stats[1] == 0xFFFFFFFFull) {
        int mpiRank = -1;
        MPI_Comm_rank(pc.comm(), &mpiRank);
        reportTraversalFailure(pc, domain, groups, mpiRank, static_cast<unsigned>(stats[4]));
        throw std::runtime_error(
            "Barnes-Hut traversal stack exhausted on rank " + std::to_string(mpiRank) +
            ". Raise theta, reduce numShells, or shrink particle count per rank.");
    }

    // Consistency checks (Params::checks: info level 3). Collective; a rank with bad positions
    // also prints its own report.
    if (params.checks) {
        int mpiRank = -1;
        MPI_Comm_rank(pc.comm(), &mpiRank);
        const SourceCheck c = checkSources(pc, domain, groups);
        if (c.owned.nonFinite + c.owned.outside + c.halos.nonFinite + c.halos.outside > 0) {
            std::fprintf(stderr,
                         "[bh.check] rank %d: owned non-finite %llu, outside the box %llu; halos "
                         "non-finite %llu, outside %llu\n",
                         mpiRank, c.owned.nonFinite, c.owned.outside, c.halos.nonFinite,
                         c.halos.outside);
        }
        unsigned long long bad[4] = {c.owned.nonFinite, c.owned.outside, c.halos.nonFinite,
                                     c.halos.outside};
        unsigned long long badSum[4];
        MPI_Allreduce(bad, badSum, 4, MPI_UNSIGNED_LONG_LONG, MPI_SUM, pc.comm());
        struct {
            double value;
            int rank;
        } ratio{c.medianExtent > 0.0f ? double(c.maxExtent) / double(c.medianExtent) : 0.0,
                mpiRank},
            worst{};
        MPI_Allreduce(&ratio, &worst, 1, MPI_DOUBLE_INT, MPI_MAXLOC, pc.comm());
        double extents[2] = {c.maxExtent, c.medianExtent}, maxExtents[2];
        MPI_Allreduce(extents, maxExtents, 2, MPI_DOUBLE, MPI_MAX, pc.comm());
        if (mpiRank == 0) {
            std::printf("[bh.check] owned non-finite=%llu outside=%llu halos non-finite=%llu "
                        "outside=%llu groupExtent max=%.4g median(max over ranks)=%.4g "
                        "worst max/median=%.1f (rank %d)\n",
                        badSum[0], badSum[1], badSum[2], badSum[3], maxExtents[0], maxExtents[1],
                        worst.value, worst.rank);
            std::fflush(stdout);
        }
    }

    // Per-step BH stats — global aggregates across ranks for direct comparison
    // with sphexa's timer.logStatistics output. Skipped during the pre_run warmup
    // solve so the printed counts reflect only the timed steps.
    if (collect) {
        // Owned focus-leaf occupancy (host copy of the layout): count, max and sum(c^2)/sum(c), the
        // leaf size around an average particle. Far above the bucket size: tree updates lag.
        const LeafOccupancy occ = leafOccupancy(domain);
        unsigned long long numLeaves = occ.numLeaves, maxLeaf = occ.maxLeaf, sumSqLeaf = occ.sumSq;

        unsigned long long local[7] = {
            static_cast<unsigned long long>(stats[0]),   // sumP2P
            static_cast<unsigned long long>(stats[2]),   // sumM2P
            numGroups,                                   // target groups
            static_cast<unsigned long long>(end - start),  // targets
            static_cast<unsigned long long>(stats[1]),   // maxP2P
            static_cast<unsigned long long>(stats[3]),   // maxM2P
            static_cast<unsigned long long>(stats[4])};  // maxStack
        unsigned long long retriedSum = 0;
        MPI_Allreduce(&retried, &retriedSum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, pc.comm());
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
                        "groups=%llu targets=%llu leaves=%llu maxLeaf=%llu wLeaf=%.1f retried=%llu\n",
                        sum[0], sum[1], mx[0], mx[1], mx[2], sum[2], sum[3], leafSum[0], leafMax,
                        weightedLeaf, retriedSum);
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
