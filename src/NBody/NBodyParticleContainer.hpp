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
 * @brief Particle container for NBody simulations
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "cstone/cuda/device_vector.h"
#include "cstone/domain/domain.hpp"
#include "cstone/fields/field_get.hpp"
#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "cstone/sfc/common.hpp"
#include "cstone/sfc/sfc_gpu.h"
#include "cstone/tree/csarray.hpp"
#include "cstone/tree/definitions.h"
#include "cstone/util/reallocate.hpp"

#include "NBody/core/Accelerator.hpp"
#include "NBody/core/BHFieldLists.hpp"
#include "NBody/core/BHPrecision.hpp"
#include "NBody/core/ParticleAttribBase.hpp"
#include "NBody/helpers/primitives_gpu.h"

namespace ippl::nbody {

namespace detail {

// Each container kernel is a backend dispatcher: the GPU body is a kernel launch
// defined in NBodyParticleContainer.cu (declared here, explicitly instantiated
// there), the CPU body is the inline OpenMP loop in the if constexpr(!kHaveGpu)
// branch. Mirrors the LeapfrogStepper split, so the container needs only a .cu —
// no per-backend .cpp.

// Geometric-mean edge length of a level-`level` octree cell, cbrtVol = cbrt(box volume).
// reductions[d] = maxTreeLevel - (SFC key bits of axis d) under cstone's mixed-dimension
// keys: axis d is not split above level reductions[d], so its cell edge is
// l_d * 2^(min(reductions[d], level) - level). For a cubic box all reductions are 0 and the
// edge is cbrtVol / 2^level.
template <class Th>
HOST_DEVICE_FUN inline Th cellEdge(Th cbrtVol, unsigned level, cstone::AxesBits reductions) {
    unsigned unsplit = 0;
    for (int d = 0; d < 3; ++d) {
        unsplit += reductions[d] < level ? reductions[d] : level;
    }
    return cbrtVol * Th(exp2(double(unsplit) / 3.0 - double(level)));
}

// Per-particle leaf-derived smoothing length: h[i] = cellEdge(level(leaf(i))).
// Both backends call the same host/device cstone helpers (findNodeBelow,
// treeLevel). Instantiated for (Th, KeyType).
template <class Th, class KeyType>
void setHFromLeavesGpu(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                       Th cbrtVol, cstone::AxesBits reductions, Th* h, unsigned n);

template <class Th, class KeyType>
void setHFromLeaves(const KeyType* keys, const KeyType* leaves, int nLeafKeys,
                    Th cbrtVol, cstone::AxesBits reductions, Th* h, unsigned n) {
    if constexpr (kHaveGpu) {
        setHFromLeavesGpu(keys, leaves, nLeafKeys, cbrtVol, reductions, h, n);
    } else {
#pragma omp parallel for schedule(static)
        for (unsigned i = 0; i < n; ++i) {
            KeyType               k         = keys[i];
            cstone::TreeNodeIndex leafIdx   = cstone::findNodeBelow(leaves, nLeafKeys, k);
            KeyType               codeRange = leaves[leafIdx + 1] - leaves[leafIdx];
            unsigned              level     = cstone::treeLevel(codeRange);
            h[i] = cellEdge(cbrtVol, level, reductions);
        }
    }
}

// keys[i] = sentinel where flags[i] != 0, for i in [0, n). Used by destroy() to
// mark owned particles with removeKey. Instantiated for KeyType (uint64_t).
template <class KeyType>
void markRemovedGpu(KeyType* keys, const bool* flags, unsigned n, KeyType sentinel);

template <class KeyType>
void markRemoved(KeyType* keys, const bool* flags, unsigned n, KeyType sentinel) {
    if constexpr (kHaveGpu) { markRemovedGpu(keys, flags, n, sentinel); }
    else {
#pragma omp parallel for
        for (unsigned i = 0; i < n; ++i) {
            if (flags[i]) { keys[i] = sentinel; }
        }
    }
}

// id[first + k] = nextId + stride*k, for k in [0, n) — globally unique strided
// IDs, like ParticleBase. Used by create(). Instantiated for IdType (uint32_t).
template <class IdType>
void assignIdsGpu(IdType* id, unsigned first, unsigned n, IdType nextId, IdType stride);

template <class IdType>
void assignIds(IdType* id, unsigned first, unsigned n, IdType nextId, IdType stride) {
    if constexpr (kHaveGpu) { assignIdsGpu(id, first, n, nextId, stride); }
    else {
#pragma omp parallel for
        for (unsigned k = 0; k < n; ++k) {
            id[first + k] = nextId + stride * static_cast<IdType>(k);
        }
    }
}

}  // namespace detail

template <class P, unsigned Dim>
class NBodyParticleContainer;

// sphexa-style conserved/dependent sync entry points:
//
//   syncGravBH<P, ConservedFields, DependentFields>(pc)
//       cstone Domain::syncGrav: SFC-sort + redistribute + halo discovery,
//       threading ConservedFields in lockstep with positions, resizing
//       DependentFields to nParticlesWithHalos, then refilling h for ryoanji P2P
//       softening. The multipole-mass slot is idxOf<"charge">.
//
//   updateBH<P, ConservedFields>(pc)
//       non-gravity Domain::sync (used by tests).
//
// ConservedFields / DependentFields are cstone util::FieldList<...> selections
// over fieldNames (see fields::StdConserved / fields::StdDependent). Each must
// list >= 2 fields — cstone's get<FieldList> collapses a single-field list to a
// bare reference, which the tuple-based sync API does not accept.
//
// Defined in NBodySync.cpp — a Kokkos-free translation unit compiled by plain
// g++, the only place cstone::get<FieldList<...>> (C++20 NTTP-pack CTAD that nvcc
// rejects) is instantiated. Caller TUs (compiled via nvcc_wrapper) see only these
// declarations and link the g++ instantiations, so the forbidden CTAD never
// reaches nvcc; every (P, ConservedFields, DependentFields) combo a caller uses
// must be explicitly instantiated in NBodySync.cpp. Declared here as friends so
// they can drive the private sort scratch and the post-sync h refill.
template <class P, class ConservedFields, class DependentFields>
void syncGravBH(NBodyParticleContainer<P, 3>& pc);
template <class P, class ConservedFields>
void updateBH(NBodyParticleContainer<P, 3>& pc);

// Gridless particle container wrapping cstone::Domain. Doubles as the sphexa
// "DataType": its public field members (x, y, z, h, m, ax, ay, az, ugrav) and
// the AcceleratorType / KeyType / RealType typedefs are exactly the surface the
// vendored gravity_wrapper.hpp MultipoleHolder{Cpu,Gpu} read, while fieldNames +
// dataTuple() are the surface cstone::get<FieldList<...>> reads. The friendly
// getRaw<"Name">/idxOf/getDV accessors map the legacy names onto these members
// (Rx->x, charge->m, Ex->ax, ...), so existing call sites are unchanged.
//
// Kokkos-free by design: consumer-side unmanaged-View accessors live in
// NBody/NBodyKokkosView.hpp, included only by Kokkos/.cu consumers. This header
// stays raw-pointer / cstone only, so the g++ sync TU that includes it links
// against cstone_gpu without dragging in Kokkos.
//
// Field storage is FieldVector<T>: cstone::DeviceVector<T> on GPU builds, plain
// std::vector<T> on the CPU build. The accelerator (NBodyAcc) is build-global,
// selected by USE_CUDA — one AccType per build, like sphexa.
template <class P, unsigned Dim>
class NBodyParticleContainer {
    static_assert(Dim == 3, "NBodyParticleContainer requires Dim == 3");

public:
    using Precision       = P;
    using Tc              = typename P::Tc;
    using Th              = typename P::Th;
    using Tm              = typename P::Tm;
    using Ta              = typename P::Ta;
    using KeyType         = std::uint64_t;
    using LocalIndex      = unsigned;
    using AcceleratorType = NBodyAcc;
    using RealType        = Tc;

    // ID is stored as uint32_t (4 B) so it fits the same-type scratch-buffer
    // constraint imposed by cstone's GlobalAssignment::distribute (which
    // requires the first two scratch buffers to share the value_type of T).
    // ~4e9 IDs; promote to uint64_t when needed.
    using IdType = std::uint32_t;

    using DomainT = cstone::Domain<KeyType, Tc, NBodyAcc>;

    // Field registry. The order here defines the idxOf<"Name"> indices and the
    // dataTuple() slots; the static_assert below keeps the three aligned.
    //   Core      : Rx Ry Rz h charge  (structural args of domain.sync(Grav))
    //   Conserved : Px Py Pz ID         (SFC-permuted; see fields::StdConserved)
    //   Dependent : Ex Ey Ez ugrav      (BH force output; see fields::StdDependent)
    //   Payload   : aux0 aux64          (application-defined per-particle data that a
    //                                    driver carries through redistribution by listing
    //                                    them as conserved; see fields::AuxConserved)
    static constexpr std::array<std::string_view, 15> fieldNames{
        "Rx", "Ry", "Rz", "h", "charge",
        "Px", "Py", "Pz", "ID",
        "Ex", "Ey", "Ez", "ugrav",
        "aux0", "aux64"};

    using DataTupleT = std::tuple<
        FieldVector<Tc>&, FieldVector<Tc>&, FieldVector<Tc>&,
        FieldVector<Th>&, FieldVector<Tm>&,
        FieldVector<Tc>&, FieldVector<Tc>&, FieldVector<Tc>&,
        FieldVector<IdType>&,
        FieldVector<Ta>&, FieldVector<Ta>&, FieldVector<Ta>&, FieldVector<Ta>&,
        FieldVector<double>&, FieldVector<std::uint64_t>&>;
    using DataTupleConstT = std::tuple<
        const FieldVector<Tc>&, const FieldVector<Tc>&, const FieldVector<Tc>&,
        const FieldVector<Th>&, const FieldVector<Tm>&,
        const FieldVector<Tc>&, const FieldVector<Tc>&, const FieldVector<Tc>&,
        const FieldVector<IdType>&,
        const FieldVector<Ta>&, const FieldVector<Ta>&, const FieldVector<Ta>&,
        const FieldVector<Ta>&,
        const FieldVector<double>&, const FieldVector<std::uint64_t>&>;

    DataTupleT dataTuple() {
        return std::tie(x, y, z, h, m, px, py, pz, id, ax, ay, az, ugrav, aux0, aux64);
    }
    DataTupleConstT dataTuple() const {
        return std::tie(x, y, z, h, m, px, py, pz, id, ax, ay, az, ugrav, aux0, aux64);
    }

    static_assert(std::tuple_size_v<DataTupleT> == fieldNames.size(),
                  "dataTuple() and fieldNames must list the same fields in order");

    NBodyParticleContainer(int rank, int nRanks,
                            unsigned bucketSize, unsigned bucketSizeFocus,
                            float theta,
                            std::array<Tc, 6> boxLoHi,
                            std::array<cstone::BoundaryType, 3> boundaries,
                            MPI_Comm comm = MPI_COMM_WORLD)
        : domain_(kExec, rank, nRanks, bucketSize, bucketSizeFocus, theta, comm,
                  cstone::Box<Tc>(boxLoHi[0], boxLoHi[1], boxLoHi[2], boxLoHi[3],
                                  boxLoHi[4], boxLoHi[5],
                                  boundaries[0], boundaries[1], boundaries[2]))
        , comm_(comm)
        , nextId_m(static_cast<IdType>(rank))
        , nRanks_m(static_cast<IdType>(nRanks)) {}
    ~NBodyParticleContainer() = default;

    NBodyParticleContainer(const NBodyParticleContainer&)            = delete;
    NBodyParticleContainer& operator=(const NBodyParticleContainer&) = delete;
    NBodyParticleContainer(NBodyParticleContainer&&) noexcept            = default;
    NBodyParticleContainer& operator=(NBodyParticleContainer&&) noexcept = default;

    // Inject nLocal new locally-owned particles (mirrors ParticleBase::create —
    // incremental delta + auto unique IDs). Grows every registry field + keys +
    // the dummy-h buffer to nParticlesWithHalos()+nLocal, preserving existing data
    // (reallocate reserves only when capacity is short), opens the gap
    // [end, end+nLocal) by shifting the suffix halos right, fills valid keys and
    // unique strided IDs for the new slots, and extends the assigned range. The
    // next syncGravBH redistributes the new particles to their owner ranks and
    // refills halos. Returns the offset; the caller writes positions / velocity /
    // charge / h at [offset, offset+nLocal) via getRaw before that sync.
    //
    // The init call (empty domain) reproduces a plain nLocal allocation.
    LocalIndex create(LocalIndex nLocal) {
        const LocalIndex end     = domain_.endIndex();
        const LocalIndex size    = domain_.nParticlesWithHalos();
        const LocalIndex newSize = size + nLocal;

        auto t = dataTuple();
        util::for_each_tuple([this, newSize](auto& a) { reallocate(a, newSize, allocGrowthRate_); }, t);
        reallocate(keys,  newSize, allocGrowthRate_);
        reallocate(hZero, newSize, allocGrowthRate_);
        cstone::fill(kExec, hZero.data(), hZero.data() + newSize, Th(0));
        for (auto* a : userAttribs) a->resize(newSize);

        // Open the gap for the new particles: relocate the suffix-halo payload
        // [end, size) -> [end+nLocal, newSize). Overlap-safe (see detail::shiftRight).
        // Keys are not relocated — cstone recomputes them from positions on the next
        // sync, reading only the removeKey sentinel that the fill below clears.
        util::for_each_tuple([end, size, nLocal](auto& a) { detail::shiftRight(a.data(), end, size, nLocal); }, t);

        // New particles occupy [end, end+nLocal): valid (non-removeKey) keys so the
        // next sync includes them; unique IDs strided by rank (like ParticleBase).
        // Zero the whole suffix, not only the new slots: the shifted halos' key slots
        // hold stale values and the grown tail is uninitialized memory. A stale removeKey
        // there would be inherited by a particle that the next sync receives into it.
        cstone::fill(kExec, keys.data() + end, keys.data() + newSize, KeyType(0));
        detail::assignIds(id.data(), end, nLocal, nextId_m, nRanks_m);
        nextId_m += nRanks_m * static_cast<IdType>(nLocal);

        // Advertise the grown buffer (assigned range end + the shifted suffix halos)
        // so the next sync sizes its exchange/scratch to newSize. setEndIndex alone
        // leaves nParticlesWithHalos() stale, which only self-corrects on the domain's
        // first sync (initBounds); a later create() would otherwise under-size the
        // exchange buffers and corrupt the heap during the gather.
        domain_.setEndIndex(end + nLocal);
        domain_.setNParticlesWithHalos(newSize);
        return end;
    }

    // Flag locally-owned particles for removal at the next sync (mirrors
    // ParticleBase::destroy). invalid has length getLocalNum(); invalid[i] sets the
    // key of owned particle startIndex()+i to removeKey, so the next syncGravBH
    // drops it and gathers the survivors. Lazy: getLocalNum() is unchanged until
    // that sync. destroyNum is accepted for ParticleBase parity (unused).
    void destroy(const bool* invalid, std::size_t /*destroyNum*/) {
        detail::markRemoved(keys.data() + domain_.startIndex(), invalid,
                            domain_.nParticles(), cstone::removeKey<KeyType>::value);
    }

    // Caller owns the attribute and must keep it alive while registered.
    // Owned-only: not permuted by sync and not halo-exchanged.
    void addAttribute(ParticleAttribBase& attr) {
        userAttribs.push_back(&attr);
        if (x.size() > 0) attr.resize(x.size());
    }

    // Compile-time index of field Name in fieldNames (single-NTTP, nvcc-safe).
    template <util::StructuralString Name>
    static constexpr std::size_t idxOf = cstone::getFieldIndex(Name.value, fieldNames);

    // Pass-through to domain.exchangeHalos. The caller supplies the field tuple
    // (e.g. std::tie(pc.getDV<"charge">())) and matching-type scratch buffers.
    template <class Arrays, class SendBuf, class RecvBuf>
    void exchangeHalos(Arrays&& arrays, SendBuf& send, RecvBuf& recv) {
        domain_.exchangeHalos(std::forward<Arrays>(arrays), send, recv);
    }

    // Reference to the underlying FieldVector of field Name. Use to build tuples
    // for exchangeHalos / driver-side composed calls.
    template <util::StructuralString Name>
    auto& getDV()             { return std::get<idxOf<Name>>(dataTuple()); }
    template <util::StructuralString Name>
    auto const& getDV() const { return std::get<idxOf<Name>>(dataTuple()); }

    // Tm-typed scratch buffers for halo exchange of the m-slot field. Drivers
    // halo-exchanging other types should declare their own scratch DVs.
    FieldVector<Tm>&       haloSendBuf()       { return haloSendBuf_; }
    FieldVector<Tm>&       haloRecvBuf()       { return haloRecvBuf_; }
    const FieldVector<Tm>& haloSendBuf() const { return haloSendBuf_; }
    const FieldVector<Tm>& haloRecvBuf() const { return haloRecvBuf_; }

    // Raw pointer to the storage of field Name. The field must be one of fieldNames.
    template <util::StructuralString Name>
    auto* getRaw()             { return std::get<idxOf<Name>>(dataTuple()).data(); }
    template <util::StructuralString Name>
    auto const* getRaw() const { return std::get<idxOf<Name>>(dataTuple()).data(); }

    LocalIndex getLocalNum() const { return domain_.nParticles(); }
    LocalIndex startIndex()  const { return domain_.startIndex(); }
    LocalIndex endIndex()    const { return domain_.endIndex(); }
    LocalIndex nWithHalos()  const { return domain_.nParticlesWithHalos(); }

    // Record the uniform smoothing length used by ryoanji P2P softening.
    // syncGravBH passes a dummy h=0 buffer to cstone (so halo bounding boxes are
    // not inflated by 2*h*factor), then refills h to nWithHalos with this value
    // for ryoanji. Caller must call this once after initializing particles,
    // before the first syncGravBH. Assumes h is uniform; non-uniform unsupported.
    void setUniformH(Th val) { uniformH = val; }

    // Switch the post-syncGrav h refill from uniform (fill with uniformH) to
    // leaf-derived (per-particle h = local octree leaf edge length). Off by default.
    void setLeafBasedH(bool on) { leafBasedH = on; }

    cstone::Box<Tc> box() const { return domain_.box(); }

    // MPI communicator the domain and all module-level collectives run on.
    // Defaults to MPI_COMM_WORLD; pass a sub-communicator to the ctor to confine
    // the module to a subset of ranks.
    MPI_Comm comm() const { return comm_; }

    // Underlying cstone::Domain reference. Used by NBodySolver to call
    // exchangeHalos / focusTree / globalTree / layout.
    DomainT&       domain()       { return domain_; }
    const DomainT& domain() const { return domain_; }

    // --- sphexa DataType surface (read by gravity_wrapper.hpp) ----------------
    FieldVector<Tc>     x, y, z;       // positions   (Rx, Ry, Rz)
    FieldVector<Th>     h;             // smoothing length
    FieldVector<Tm>     m;             // mass/source (charge)
    FieldVector<Tc>     px, py, pz;    // velocity    (Px, Py, Pz) — conserved
    FieldVector<IdType> id;            // particle id (ID)         — conserved
    FieldVector<Ta>     ax, ay, az;    // accelerations (Ex, Ey, Ez)
    FieldVector<Ta>     ugrav;         // gravitational potential (unused output)
    // Application payload, opaque to the BH stack. Only redistributed when a
    // driver lists them in its ConservedFields (e.g. fields::AuxConserved);
    // otherwise they are merely resized by create(). 8-byte types satisfy
    // cstone's scratch-buffer element-size requirement (<= sizeof(Tc)).
    FieldVector<double>        aux0;   // aux0  (e.g. per-particle time step)
    FieldVector<std::uint64_t> aux64;  // aux64 (e.g. 64-bit application particle ID)
    Tc                  g{Tc(1)};      // gravitational/Coulomb prefactor
    Tc                  egrav{Tc(0)};  // total potential energy (unused output)

private:
    template <class P_, class CF, class DF>
    friend void syncGravBH(NBodyParticleContainer<P_, 3>& pc);
    template <class P_, class CF>
    friend void updateBH(NBodyParticleContainer<P_, 3>& pc);

    // After a sync cstone holds valid keys only for the assigned range; the halo slots
    // keep their previous content, including removeKey markers set by destroy() (sorted
    // there by the exchange). cstone's computeSfcKeys never overwrites a marker, so a
    // particle received into such a slot by a later sync would inherit it: it is dropped
    // from the local set, the assignment counts no longer match and the halo layout
    // becomes inconsistent ("LET refine"). Reset the halo keys and recompute them from
    // the halo positions (leaf-based h reads them).
    void resetHaloKeys() {
        const LocalIndex start = domain_.startIndex();
        const LocalIndex end   = domain_.endIndex();
        const LocalIndex n     = domain_.nParticlesWithHalos();
        const auto&      bx    = domain_.box();
        cstone::fill(kExec, keys.data(), keys.data() + start, KeyType(0));
        cstone::fill(kExec, keys.data() + end, keys.data() + n, KeyType(0));
        cstone::computeSfcKeys(kExec, x.data(), y.data(), z.data(),
                               cstone::sfcKindPointer(keys.data()), start, bx);
        cstone::computeSfcKeys(kExec, x.data() + end, y.data() + end, z.data() + end,
                               cstone::sfcKindPointer(keys.data()) + end, n - end, bx);
    }

    // Restore h after syncGravBH (which threads a dummy h=0 through cstone) for
    // ryoanji P2P softening, over [0, nWithHalos). Uniform fill of newly-grown
    // entries, or per-particle leaf-edge length when leafBasedH is set.
    void refillHaloH() {
        const LocalIndex n = domain_.nParticlesWithHalos();
        if (leafBasedH) {
            // Per-particle h = local octree leaf edge length. Recomputed every
            // step over [0, nWithHalos) because focus-tree refinement migrates
            // particles between leaves between calls.
            h.resize(n);
            auto           leaves   = domain_.focusTree().treeLeavesAcc();
            const auto&    bx       = domain_.box();
            const Tc       vol      = bx.lx() * bx.ly() * bx.lz();
            const Th       cbrtVol  = static_cast<Th>(std::cbrt(static_cast<double>(vol)));
            const unsigned maxLevel = cstone::maxTreeLevel<KeyType>{};
            const auto     reductions =
                cstone::AxesBits{maxLevel, maxLevel, maxLevel} - bx.getBoxDimBits(maxLevel);
            detail::setHFromLeaves<Th, KeyType>(
                keys.data(), leaves.data(), static_cast<int>(leaves.size()),
                cbrtVol, reductions, h.data(), n);
        } else {
            // Fill only newly-grown entries — assumes uniform h (the existing
            // entries already hold the correct value from create()/IC or the
            // previous grow).
            const LocalIndex oldH = h.size();
            h.resize(n);
            if (n > oldH) {
                cstone::fill(kExec, h.data() + oldH, h.data() + n, uniformH);
            }
        }
    }

    DomainT                       domain_;
    MPI_Comm                      comm_;
    FieldVector<KeyType>          keys;
    FieldVector<Tc>               scratch0, scratch1;
    FieldVector<Th>               scratchTh;
    FieldVector<cstone::LocalIndex> scratchSfc;
    FieldVector<Tm>               haloSendBuf_, haloRecvBuf_;
    FieldVector<Th>               hZero;
    Th                            uniformH{Th(0)};
    bool                          leafBasedH{false};
    std::vector<ParticleAttribBase*> userAttribs;

    // Buffer growth factor for create()'s reallocation fallback — the container's
    // own factor (cstone's Domain has a separate one for its sync reallocations).
    float                         allocGrowthRate_{1.05f};

    // Next unique particle ID this rank will assign, strided by nRanks_m so IDs
    // are globally unique across ranks (id = nextId_m + nRanks_m*k), like
    // ParticleBase. Advanced by create().
    IdType                        nextId_m;
    IdType                        nRanks_m;
};

// Free-function shortcuts so callers in templated contexts don't need
// `.template` to invoke the member-template accessor.
template <util::StructuralString Name, class P, unsigned Dim>
auto* getRaw(NBodyParticleContainer<P, Dim>& pc) {
    return pc.template getRaw<Name>();
}
template <util::StructuralString Name, class P, unsigned Dim>
auto const* getRaw(const NBodyParticleContainer<P, Dim>& pc) {
    return pc.template getRaw<Name>();
}

} // namespace ippl::nbody
