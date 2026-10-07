#include "NBody/NBodyParticleContainer.hpp"

#include <tuple>

#include "cstone/fields/field_get.hpp"
#include "cstone/util/tuple_util.hpp"

// Compiled by plain g++ (the ippl_nbody_sync target links cstone_gpu + MPI but
// NOT Kokkos, so kokkos_launch_compiler does not reroute it through nvcc_wrapper).
// This is the only TU where cstone::get<FieldList<...>> is instantiated.

namespace ippl::nbody {

template <class P, class ConservedFields, class DependentFields>
void syncGravBH(NBodyParticleContainer<P, 3>& pc) {
    using C                     = NBodyParticleContainer<P, 3>;
    constexpr std::size_t mSlot = C::template idxOf<"charge">;

    auto t = pc.dataTuple();
    pc.domain_.syncGrav(pc.keys, pc.x, pc.y, pc.z, pc.hZero, std::get<mSlot>(t),
                        cstone::get<ConservedFields>(pc),
                        std::tie(pc.scratch0, pc.scratch1, pc.scratchTh, pc.scratchSfc));

    // Dependent fields hold the BH force output; resize to the post-sync extent
    // so consumer kernels index over [0, nWithHalos). Halo-region content is
    // undefined until the solver writes it.
    const auto n = pc.domain_.nParticlesWithHalos();
    util::for_each_tuple([n](auto& a) { a.resize(n); }, cstone::get<DependentFields>(pc));

    pc.refillHaloH();
}

template <class P, class ConservedFields>
void updateBH(NBodyParticleContainer<P, 3>& pc) {
    pc.domain_.sync(pc.keys, pc.x, pc.y, pc.z, pc.h,
                    cstone::get<ConservedFields>(pc),
                    std::tie(pc.scratch0, pc.scratch1, pc.scratchTh, pc.scratchSfc));
}

// Explicit instantiations. Every (Precision, ConservedFields, DependentFields)
// combo used by a driver or test must appear here — caller TUs only see the
// declarations and resolve these symbols at link.
#define IPPL_NBODY_INSTANTIATE_SYNC(POLICY)                                          \
    template void syncGravBH<POLICY, fields::StdConserved, fields::StdDependent>(     \
        NBodyParticleContainer<POLICY, 3>&);                                         \
    template void updateBH<POLICY, fields::StdConserved>(NBodyParticleContainer<POLICY, 3>&);

IPPL_NBODY_INSTANTIATE_SYNC(DoublePrecision)
IPPL_NBODY_INSTANTIATE_SYNC(MixedPrecision)
IPPL_NBODY_INSTANTIATE_SYNC(FloatPrecision)

#undef IPPL_NBODY_INSTANTIATE_SYNC

// Payload-carrying sync (fields::AuxConserved). The 8-byte aux slots need an
// 8-byte scratch buffer, so only the policies with Tc = double qualify.
template void syncGravBH<DoublePrecision, fields::AuxConserved, fields::StdDependent>(
    NBodyParticleContainer<DoublePrecision, 3>&);
template void syncGravBH<MixedPrecision, fields::AuxConserved, fields::StdDependent>(
    NBodyParticleContainer<MixedPrecision, 3>&);

// Test-only combo (per-driver field-list selection): conserve only Px and ID, so
// Py/Pz are NOT permuted. unit_tests/NBody names the same literal FieldList type.
template void updateBH<DoublePrecision, util::FieldList<"Px", "ID">>(
    NBodyParticleContainer<DoublePrecision, 3>&);

}  // namespace ippl::nbody
