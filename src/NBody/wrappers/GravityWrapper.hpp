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
 * @brief Gravity wrapper vendored from sphexa
 * 
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

// Local adaptations for the IPPL fork:
//   - namespace ippl::nbody (was sphexa)
//   - scalar types derived via std::declval (our container/DataType is not
//     default-constructible — it owns a cstone::Domain)
//   - setEwaldSettings() so the BH solver can pass numShells / Ewald params
//   - the per-step MPI stats printout + stack-exhaustion guard live in the
//     solver (NBodySolver), not here; the holders stay pure-compute
//
// MultipoleHolderCpu / MultipoleHolderGpu present an identical
// computeSpatialGroups / upsweep / traverse / readStats interface. The BH solver
// selects between them with std::conditional_t<HaveGpu<Acc>, Gpu, Cpu>, exactly
// like NbodyProp does upstream.

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "cstone/domain/domain.hpp"
#include "ryoanji/interface/ewald.cuh"
#include "ryoanji/interface/global_multipole.hpp"
#include "ryoanji/interface/multipole_holder.cuh"
#include "ryoanji/nbody/ewald.hpp"
#include "ryoanji/nbody/traversal_cpu.hpp"

namespace ippl::nbody {

// ryoanji's Ewald summation exists for CartesianQuadrupole only: the GPU kernels are not
// instantiated for other types (link error) and the header-only CPU version would silently
// ignore a dipole. Both multipole types are util::array<T, N> aliases, so is_same works.
template <class MType>
inline constexpr bool kHasEwald =
    std::is_same_v<MType, ryoanji::CartesianQuadrupole<typename MType::value_type>>;

[[noreturn]] inline void throwNoEwald() {
    throw std::runtime_error(
        "Barnes-Hut: periodic boxes need Ewald summation, which exists only for quadrupole "
        "multipoles (MultipoleOrder::Quadrupole).");
}

template <class MType, class DomainType, class DataType>
class MultipoleHolderCpu {
    using Ta = typename std::decay_t<decltype(std::declval<DataType>().ax)>::value_type;
    using Tu = typename std::decay_t<decltype(std::declval<DataType>().x)>::value_type;

public:
    MultipoleHolderCpu() = default;

    void setEwaldSettings(const ryoanji::EwaldSettings& s) { ewaldSettings_ = s; }

    cstone::GroupView computeSpatialGroups(const DataType& /*d*/, const DomainType& domain) {
        return {.firstBody  = domain.startIndex(),
                .lastBody   = domain.endIndex(),
                .numGroups  = 0,
                .groupStart = nullptr,
                .groupEnd   = nullptr};
    }

    void upsweep(const DataType& d, const DomainType& domain) {
        const auto& focusTree = domain.focusTree();
        reallocate(multipoles_, focusTree.octreeViewAcc().numNodes, 1.05);
        ryoanji::computeMultipoles(d.x.data(), d.y.data(), d.z.data(), d.m.data(),
                                   domain.globalTree(), domain.focusTree(),
                                   domain.layout().data(), multipoles_.data());
    }

    void traverse(cstone::GroupView /*grp*/, DataType& d, const DomainType& domain) {
        const auto& focusTree = domain.focusTree();
        const auto  octree    = focusTree.octreeViewAcc();
        const auto& box       = domain.box();
        bool        usePbc    = box.boundaryX() == cstone::BoundaryType::periodic;
        int         numShells = usePbc ? ewaldSettings_.numReplicaShells : 0;

        d.egrav = 0;
        ryoanji::computeGravity(octree.childOffsets, octree.parents, octree.internalToLeaf,
                                focusTree.expansionCentersAcc().data(), multipoles_.data(),
                                domain.layout().data(), domain.startCell(), domain.endCell(),
                                d.x.data(), d.y.data(), d.z.data(), d.h.data(), d.m.data(),
                                domain.box(), d.g, d.ugrav.data(), d.ax.data(), d.ay.data(),
                                d.az.data(), &d.egrav, numShells);

        if constexpr (kHasEwald<MType>) {
            if (usePbc) {
                ryoanji::computeGravityEwald(makeVec3(focusTree.expansionCentersAcc()[0]),
                                             multipoles_.front(), domain.startIndex(),
                                             domain.endIndex(), d.x.data(), d.y.data(),
                                             d.z.data(), d.m.data(), box, d.g, d.ugrav.data(),
                                             d.ax.data(), d.ay.data(), d.az.data(), &d.egrav,
                                             ewaldSettings_);
            }
        } else if (usePbc) {
            throwNoEwald();
        }
    }

    util::array<uint64_t, 5> readStats() const { return {0, 0, 0, 0, 0}; }

    const MType* multipoles() const { return multipoles_.data(); }

private:
    std::vector<MType>     multipoles_;
    ryoanji::EwaldSettings ewaldSettings_;
};

template <class MType, class DomainType, class DataType>
class MultipoleHolderGpu {
    using KeyType = typename DataType::KeyType;
    using Tc      = typename std::decay_t<decltype(std::declval<DataType>().x)>::value_type;
    using Th      = typename std::decay_t<decltype(std::declval<DataType>().h)>::value_type;
    using Tm      = typename std::decay_t<decltype(std::declval<DataType>().m)>::value_type;
    using Ta      = typename std::decay_t<decltype(std::declval<DataType>().ax)>::value_type;
    using Tf      = typename DomainType::RealType;

public:
    MultipoleHolderGpu() = default;

    void setEwaldSettings(const ryoanji::EwaldSettings& s) { ewaldSettings_ = s; }

    cstone::GroupView computeSpatialGroups(const DataType& d, const DomainType& domain) {
        return mHolder_.computeSpatialGroups(domain.startIndex(), domain.endIndex(), rawPtr(d.x),
                                             rawPtr(d.y), rawPtr(d.z), rawPtr(d.h),
                                             domain.focusTree(), domain.layout().data(),
                                             domain.box());
    }

    void upsweep(const DataType& d, const DomainType& domain) {
        const auto& focusTree = domain.focusTree();
        mHolder_.upsweep(rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.m), domain.globalTree(),
                         focusTree, domain.layout().data());
    }

    void traverse(cstone::GroupView grp, DataType& d, const DomainType& domain) {
        const auto& box       = domain.box();
        bool        usePbc    = box.boundaryX() == cstone::BoundaryType::periodic;
        int         numShells = usePbc ? ewaldSettings_.numReplicaShells : 0;

        d.egrav = mHolder_.compute(grp, rawPtr(d.x), rawPtr(d.y), rawPtr(d.z), rawPtr(d.m),
                                   rawPtr(d.h), d.g, numShells, domain.box(), rawPtr(d.ugrav),
                                   rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az));

        if constexpr (kHasEwald<MType>) {
            if (usePbc) {
                ryoanji::Vec4<Tf> rootCenter;
                cstone::memcpyD2HAsync(domain.exec(), domain.focusTree().expansionCentersAcc().data(), 1,
                                       &rootCenter);
                MType rootM;
                cstone::memcpyD2HAsync(domain.exec(), mHolder_.deviceMultipoles(), 1, &rootM);
                cstone::syncGpu(domain.exec());

                computeGravityEwaldGpu(makeVec3(rootCenter), rootM, grp, rawPtr(d.x), rawPtr(d.y),
                                       rawPtr(d.z), rawPtr(d.m), box, d.g, rawPtr(d.ugrav),
                                       rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az), &d.egrav,
                                       ewaldSettings_);
            }
        } else if (usePbc) {
            throwNoEwald();
        }
    }

    //! @brief return numP2P, maxP2P, numM2P, maxM2P, maxStack stats
    util::array<uint64_t, 5> readStats() const { return mHolder_.readStats(); }

    const MType* deviceMultipoles() const { return mHolder_.deviceMultipoles(); }

private:
    ryoanji::MultipoleHolder<Tc, Th, Tm, Ta, Tf, KeyType, MType> mHolder_;
    ryoanji::EwaldSettings                                       ewaldSettings_;
};

// Selects the holder matching the build's accelerator, mirroring NbodyProp's
// MHolder_t. MType is the multipole expansion type (CartesianQuadrupole<Tmm>).
template <class MType, class DomainType, class DataType, class Acc>
using MultipoleHolder = std::conditional_t<cstone::execution::HaveGpu<Acc>{},
                                           MultipoleHolderGpu<MType, DomainType, DataType>,
                                           MultipoleHolderCpu<MType, DomainType, DataType>>;

}  // namespace ippl::nbody
