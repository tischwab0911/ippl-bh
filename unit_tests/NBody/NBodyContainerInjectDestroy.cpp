// Tests for runtime particle injection (create) and deletion (destroy) on
// NBodyParticleContainer:
//   - destroy(mask): marked owned particles are dropped by the next syncGravBH,
//     survivors keep their IDs.
//   - create(n): new particles are appended with unique strided IDs and survive
//     the next syncGravBH; the global count grows by n.
//   - chunked vs. one-step: building a particle set in consecutive create() calls
//     (with or without a sync between chunks) yields the same post-sync state,
//     value by value, as creating the whole set in a single create().
//   - detail::shiftRight: overlap-safe right shift (exercises the suffix-halo move
//     that single-rank container tests don't otherwise hit — no halos locally).

#include <array>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"
#include "cstone/sfc/box.hpp"

#include "NBody/NBodyParticleContainer.hpp"
#include "NBody/NBodySolver.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::FieldVector;
using ippl::nbody::getRaw;
using ippl::nbody::NBodyParticleContainer;
using ippl::nbody::NBodySolver;
using ippl::nbody::syncGravBH;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;
namespace fields = ippl::nbody::fields;

namespace {

using C      = NBodyParticleContainer<DoublePrecision, 3>;
using IdType = C::IdType;

constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

C makeContainer() {
    using cstone::BoundaryType;
    return C(/*rank=*/0, /*nRanks=*/1,
             kBucketSize, kBucketSizeFoc, kTheta,
             std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
             std::array<BoundaryType, 3>{
                 BoundaryType::open, BoundaryType::open, BoundaryType::open});
}

// Fill positions + h for the owned slots [0, n); IDs are auto-assigned by create().
void seedPositions(C& pc, unsigned n, unsigned long seed) {
    std::vector<double> x(n), y(n), z(n), h(n, 1.0e-2);
    ::srand48(static_cast<long>(seed));
    for (unsigned i = 0; i < n; ++i) {
        x[i] = drand48();
        y[i] = drand48();
        z[i] = drand48();
    }
    uploadHost(x, ippl::nbody::getRaw<"Rx">(pc));
    uploadHost(y, ippl::nbody::getRaw<"Ry">(pc));
    uploadHost(z, ippl::nbody::getRaw<"Rz">(pc));
    uploadHost(h, ippl::nbody::getRaw<"h">(pc));
}

std::vector<IdType> ownedIds(C& pc) {
    const unsigned start = pc.startIndex();
    const unsigned end   = pc.endIndex();
    std::vector<IdType> all;
    downloadDevice(ippl::nbody::getRaw<"ID">(pc), pc.nWithHalos(), all);
    return std::vector<IdType>(all.begin() + start, all.begin() + end);
}

// --- one-step vs. chunked construction --------------------------------------
//
// A per-ID initial condition (a pure function of the global particle ID) lets us
// build the same particle set two ways. On a single rank create() assigns
// ID == global slot index, so seeding slot k from icFor(k) gives both builds the
// identical (ID, attributes) set; cstone's SFC sort is a deterministic function
// of position, so identical sets sort to identical layouts and yield an identical
// BH solve. Single rank exercises create()'s append / grow / strided-ID / seed
// path and (with a sync between chunks) injection into an already-sorted
// container; the suffix-halo shift inside create() is multi-rank only.

constexpr unsigned kNInject = 4096;
constexpr std::array<unsigned, 4> kChunkSizes{1000u, 1u, 1549u, 1546u};
static_assert(kChunkSizes[0] + kChunkSizes[1] + kChunkSizes[2] + kChunkSizes[3] == kNInject,
              "chunk sizes must sum to kNInject");

uint64_t splitmix(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double unitFrac(IdType id, uint64_t salt) {  // deterministic value in [0, 1)
    const uint64_t h = splitmix((static_cast<uint64_t>(id) << 8) ^ salt);
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);
}

struct Ic { double rx, ry, rz, h, q, px, py, pz; };

Ic icFor(IdType id) {
    return Ic{0.01 + 0.98 * unitFrac(id, 0),   // positions strictly inside [0,1]^3
              0.01 + 0.98 * unitFrac(id, 1),
              0.01 + 0.98 * unitFrac(id, 2),
              1.0e-2,
              0.1 + unitFrac(id, 3),
              -1.0 + 2.0 * unitFrac(id, 4),
              -1.0 + 2.0 * unitFrac(id, 5),
              -1.0 + 2.0 * unitFrac(id, 6)};
}

// Seed the freshly-created slots [off, off+n); on a single rank slot k has ID k.
void seedRange(C& pc, unsigned off, unsigned n) {
    std::vector<double> rx(n), ry(n), rz(n), h(n), q(n), px(n), py(n), pz(n);
    for (unsigned j = 0; j < n; ++j) {
        const Ic c = icFor(static_cast<IdType>(off + j));
        rx[j] = c.rx; ry[j] = c.ry; rz[j] = c.rz; h[j] = c.h;
        q[j]  = c.q;  px[j] = c.px; py[j] = c.py; pz[j] = c.pz;
    }
    uploadHost(rx, getRaw<"Rx">(pc) + off);
    uploadHost(ry, getRaw<"Ry">(pc) + off);
    uploadHost(rz, getRaw<"Rz">(pc) + off);
    uploadHost(h,  getRaw<"h">(pc) + off);
    uploadHost(q,  getRaw<"charge">(pc) + off);
    uploadHost(px, getRaw<"Px">(pc) + off);
    uploadHost(py, getRaw<"Py">(pc) + off);
    uploadHost(pz, getRaw<"Pz">(pc) + off);
}

template <class T>
std::vector<T> ownedSlice(const T* dev, const C& pc) {
    std::vector<T> all;
    downloadDevice(dev, pc.nWithHalos(), all);
    return std::vector<T>(all.begin() + pc.startIndex(), all.begin() + pc.endIndex());
}

// Every SFC-permuted field (Core + StdConserved) must match slot-by-slot.
void expectOwnedIdentical(C& a, C& b) {
    ASSERT_EQ(a.getLocalNum(), b.getLocalNum());
    const auto rxA = ownedSlice(getRaw<"Rx">(a), a),     rxB = ownedSlice(getRaw<"Rx">(b), b);
    const auto ryA = ownedSlice(getRaw<"Ry">(a), a),     ryB = ownedSlice(getRaw<"Ry">(b), b);
    const auto rzA = ownedSlice(getRaw<"Rz">(a), a),     rzB = ownedSlice(getRaw<"Rz">(b), b);
    const auto hA  = ownedSlice(getRaw<"h">(a), a),      hB  = ownedSlice(getRaw<"h">(b), b);
    const auto qA  = ownedSlice(getRaw<"charge">(a), a), qB  = ownedSlice(getRaw<"charge">(b), b);
    const auto pxA = ownedSlice(getRaw<"Px">(a), a),     pxB = ownedSlice(getRaw<"Px">(b), b);
    const auto pyA = ownedSlice(getRaw<"Py">(a), a),     pyB = ownedSlice(getRaw<"Py">(b), b);
    const auto pzA = ownedSlice(getRaw<"Pz">(a), a),     pzB = ownedSlice(getRaw<"Pz">(b), b);
    const auto idA = ownedSlice(getRaw<"ID">(a), a),     idB = ownedSlice(getRaw<"ID">(b), b);
    for (unsigned j = 0; j < idA.size(); ++j) {
        EXPECT_EQ(idA[j], idB[j])        << "ID slot "     << j;
        EXPECT_DOUBLE_EQ(rxA[j], rxB[j]) << "Rx slot "     << j;
        EXPECT_DOUBLE_EQ(ryA[j], ryB[j]) << "Ry slot "     << j;
        EXPECT_DOUBLE_EQ(rzA[j], rzB[j]) << "Rz slot "     << j;
        EXPECT_DOUBLE_EQ(hA[j],  hB[j])  << "h slot "      << j;
        EXPECT_DOUBLE_EQ(qA[j],  qB[j])  << "charge slot " << j;
        EXPECT_DOUBLE_EQ(pxA[j], pxB[j]) << "Px slot "     << j;
        EXPECT_DOUBLE_EQ(pyA[j], pyB[j]) << "Py slot "     << j;
        EXPECT_DOUBLE_EQ(pzA[j], pzB[j]) << "Pz slot "     << j;
    }
}

// Run the BH solve on the post-sync container and reduce the field to a scalar
// Σ_owned (Ex² + Ey² + Ez²): an end-to-end observable of the whole pipeline.
double solveFieldSumSq(C& pc) {
    typename NBodySolver<DoublePrecision, 3>::Params params;
    params.G         = 1.0;
    params.theta     = kTheta;
    params.numShells = 0;   // open BC: no image lattice / Ewald
    NBodySolver<DoublePrecision, 3> solver(pc, params);
    solver.runSolver();
    const auto ex = ownedSlice(getRaw<"Ex">(pc), pc);
    const auto ey = ownedSlice(getRaw<"Ey">(pc), pc);
    const auto ez = ownedSlice(getRaw<"Ez">(pc), pc);
    double sumSq = 0.0;
    for (unsigned j = 0; j < ex.size(); ++j) {
        sumSq += ex[j] * ex[j] + ey[j] * ey[j] + ez[j] * ez[j];
    }
    return sumSq;
}

void buildOneStep(C& pc, unsigned n) {
    pc.setUniformH(1.0e-2);
    pc.create(n);
    seedRange(pc, 0, n);
    syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);
}

// Build by consecutive create() calls; syncBetween re-syncs after every chunk.
void buildChunked(C& pc, const std::array<unsigned, 4>& chunks, bool syncBetween) {
    pc.setUniformH(1.0e-2);
    unsigned base = 0;
    for (unsigned c : chunks) {
        const unsigned off = pc.create(c);
        ASSERT_EQ(off, base) << "create() appends at the current owned end";
        seedRange(pc, off, c);
        base += c;
        if (syncBetween) {
            syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);
        }
    }
    if (!syncBetween) {
        syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);
    }
}

} // namespace

TEST(NBodyContainerInjectDestroy, DestroyDropsMarkedSurvivorsKeepIds) {
    constexpr unsigned kN = 4096;
    C pc = makeContainer();
    pc.setUniformH(1.0e-2);

    pc.create(kN);               // auto-IDs 0..kN-1
    seedPositions(pc, kN, /*seed=*/424242);
    syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);
    ASSERT_EQ(pc.getLocalNum(), kN);

    // Mark every other owned particle for removal (mask over [0, nParticles())).
    const unsigned n = pc.getLocalNum();
    Kokkos::View<bool*> mask("destroyMask", n);
    auto maskHost = Kokkos::create_mirror_view(mask);
    unsigned destroyNum = 0;
    for (unsigned i = 0; i < n; ++i) {
        maskHost(i) = (i % 2 == 1);
        destroyNum += maskHost(i);
    }
    Kokkos::deep_copy(mask, maskHost);

    // Record the IDs that are about to be deleted (at the masked owned slots).
    const std::vector<IdType> before = ownedIds(pc);
    std::unordered_set<IdType> deleted;
    for (unsigned i = 0; i < n; ++i) {
        if (maskHost(i)) { deleted.insert(before[i]); }
    }

    pc.destroy(mask.data(), destroyNum);
    syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);

    EXPECT_EQ(pc.getLocalNum(), kN - destroyNum);

    const std::vector<IdType> after = ownedIds(pc);
    std::unordered_set<IdType> survivors(after.begin(), after.end());
    EXPECT_EQ(survivors.size(), kN - destroyNum) << "survivor IDs must be unique";
    for (IdType s : after) {
        EXPECT_EQ(deleted.count(s), 0u) << "deleted ID " << s << " reappeared";
        EXPECT_LT(s, kN);
    }
}

TEST(NBodyContainerInjectDestroy, CreateInjectsParticlesWithUniqueIds) {
    constexpr unsigned kN = 4096;
    constexpr unsigned kInject = 512;
    C pc = makeContainer();
    pc.setUniformH(1.0e-2);

    pc.create(kN);                       // auto-IDs 0..kN-1
    seedPositions(pc, kN, /*seed=*/131);
    syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);
    ASSERT_EQ(pc.getLocalNum(), kN);

    // Inject kInject more particles; create() returns the write offset.
    const unsigned first = pc.create(kInject);   // auto-IDs kN..kN+kInject-1
    {
        std::vector<double> x(kInject), y(kInject), z(kInject), h(kInject, 1.0e-2);
        ::srand48(/*seed=*/999);
        for (unsigned i = 0; i < kInject; ++i) {
            x[i] = drand48();
            y[i] = drand48();
            z[i] = drand48();
        }
        uploadHost(x, ippl::nbody::getRaw<"Rx">(pc) + first);
        uploadHost(y, ippl::nbody::getRaw<"Ry">(pc) + first);
        uploadHost(z, ippl::nbody::getRaw<"Rz">(pc) + first);
        uploadHost(h, ippl::nbody::getRaw<"h">(pc) + first);
    }
    syncGravBH<DoublePrecision, fields::StdConserved, fields::StdDependent>(pc);

    EXPECT_EQ(pc.getLocalNum(), kN + kInject);

    const std::vector<IdType> ids = ownedIds(pc);
    std::unordered_set<IdType> seen(ids.begin(), ids.end());
    EXPECT_EQ(seen.size(), kN + kInject) << "all IDs unique after injection";
    for (IdType v = 0; v < kN + kInject; ++v) {
        EXPECT_EQ(seen.count(v), 1u) << "missing ID " << v;
    }
}

TEST(NBodyContainerInjectDestroy, ShiftRightIsOverlapSafe) {
    constexpr unsigned kLen = 10;
    FieldVector<double> v;
    v.resize(kLen);
    std::vector<double> host(kLen);
    for (unsigned i = 0; i < kLen; ++i) { host[i] = static_cast<double>(i); }
    uploadHost(host, v.data());

    // Move [2, 6) -> [5, 9): destination overlaps the source at index 5.
    ippl::nbody::detail::shiftRight(v.data(), /*first=*/2u, /*last=*/6u, /*by=*/3u);

    std::vector<double> out;
    downloadDevice(v.data(), kLen, out);
    // Shifted window holds the original [2,6) values; a naive forward copy would
    // corrupt out[8] (it reads index 5 after writing it).
    EXPECT_DOUBLE_EQ(out[5], 2.0);
    EXPECT_DOUBLE_EQ(out[6], 3.0);
    EXPECT_DOUBLE_EQ(out[7], 4.0);
    EXPECT_DOUBLE_EQ(out[8], 5.0);
    // Untouched slots.
    EXPECT_DOUBLE_EQ(out[0], 0.0);
    EXPECT_DOUBLE_EQ(out[1], 1.0);
    EXPECT_DOUBLE_EQ(out[9], 9.0);
}

// Chunked back-to-back creates (single final sync) reproduce the one-step build
// exactly, field by field after sync.
TEST(NBodyContainerInjectDestroy, ChunkedCreateMatchesSingleCreate) {
    C a = makeContainer();
    buildOneStep(a, kNInject);
    ASSERT_EQ(a.getLocalNum(), kNInject);

    C b = makeContainer();
    buildChunked(b, kChunkSizes, /*syncBetween=*/false);

    expectOwnedIdentical(a, b);
}

// Same, but re-syncing after every chunk — i.e. injecting into a container that
// is already SFC-sorted and halo-populated. Still bit-identical after sync.
TEST(NBodyContainerInjectDestroy, ChunkedCreateWithSyncBetweenMatchesSingleCreate) {
    C a = makeContainer();
    buildOneStep(a, kNInject);

    C b = makeContainer();
    buildChunked(b, kChunkSizes, /*syncBetween=*/true);

    expectOwnedIdentical(a, b);
}

// End-to-end: run the BH solve on both builds and compare the field energy.
// Uses the back-to-back build (single final sync) so both containers do one sync
// on byte-identical pre-sync buffers — identical SFC layout AND identical focus
// tree ⇒ identical solve ⇒ identical Σ|E|², to the bit. (A sync between chunks
// would re-converge the focus tree along a different path, perturbing the BH
// multipole grouping at the ~theta truncation level, so that build is only
// asserted bit-exact for the SFC-permuted fields, not the solve.)
TEST(NBodyContainerInjectDestroy, ChunkedCreateProducesSameBHFieldEnergy) {
    C a = makeContainer();
    buildOneStep(a, kNInject);

    C b = makeContainer();
    buildChunked(b, kChunkSizes, /*syncBetween=*/false);

    const double ea = solveFieldSumSq(a);
    const double eb = solveFieldSumSq(b);
    ASSERT_GT(ea, 0.0) << "BH solve produced zero field — invalid setup";
    EXPECT_DOUBLE_EQ(ea, eb) << "BH field energy must be insertion-grouping invariant";
}

// The application payload slots (aux0 double, aux64 uint64) must travel with
// their particle through the SFC permutation when listed as conserved
// (fields::AuxConserved), including across a destroy-all + re-create cycle,
// which is how a driver re-injects an externally owned particle set.
namespace {

constexpr std::uint64_t kAuxKeyBase = 0x0123456700000000ULL;  // needs > 32 bits

std::uint64_t auxKeyFor(IdType id) { return kAuxKeyBase + 3ULL * static_cast<std::uint64_t>(id); }
double auxValueFor(std::uint64_t key) { return 1.0e-12 * static_cast<double>(key & 0xFFFFFULL) + 0.5; }

void seedAux(C& pc, unsigned off, unsigned n) {
    std::vector<double> a0(n);
    std::vector<std::uint64_t> a64(n);
    for (unsigned j = 0; j < n; ++j) {
        a64[j] = auxKeyFor(static_cast<IdType>(off + j));
        a0[j]  = auxValueFor(a64[j]);
    }
    uploadHost(a0, getRaw<"aux0">(pc) + off);
    uploadHost(a64, getRaw<"aux64">(pc) + off);
}

// Every owned particle's payload must belong to it: decode the IPPL ID from
// aux64 and compare positions/momenta/aux0 against the per-ID initial condition.
void expectPayloadConsistent(C& pc, unsigned expectedN) {
    ASSERT_EQ(pc.getLocalNum(), expectedN);
    const auto rx  = ownedSlice(getRaw<"Rx">(pc), pc);
    const auto px  = ownedSlice(getRaw<"Px">(pc), pc);
    const auto a0  = ownedSlice(getRaw<"aux0">(pc), pc);
    const auto a64 = ownedSlice(getRaw<"aux64">(pc), pc);
    std::unordered_set<std::uint64_t> seen;
    for (unsigned j = 0; j < a64.size(); ++j) {
        ASSERT_GE(a64[j], kAuxKeyBase);
        ASSERT_EQ((a64[j] - kAuxKeyBase) % 3ULL, 0ULL);
        const auto id = static_cast<IdType>((a64[j] - kAuxKeyBase) / 3ULL);
        const Ic c    = icFor(id);
        EXPECT_EQ(rx[j], c.rx) << "slot " << j;
        EXPECT_EQ(px[j], c.px) << "slot " << j;
        EXPECT_EQ(a0[j], auxValueFor(a64[j])) << "slot " << j;
        seen.insert(a64[j]);
    }
    EXPECT_EQ(seen.size(), expectedN) << "payload keys must stay unique";
}

}  // namespace

TEST(NBodyContainerInjectDestroy, AuxPayloadFollowsParticlesThroughSync) {
    constexpr unsigned kN = 4096;
    C pc = makeContainer();
    pc.setUniformH(1.0e-2);
    pc.create(kN);
    seedRange(pc, 0, kN);
    seedAux(pc, 0, kN);
    syncGravBH<DoublePrecision, fields::AuxConserved, fields::StdDependent>(pc);
    expectPayloadConsistent(pc, kN);

    // Re-inject the same set: destroy every owned particle, create kN fresh
    // slots and seed them from the per-ID IC again, then sync.
    const unsigned n = pc.getLocalNum();
    Kokkos::View<bool*> mask("destroyAll", n);
    Kokkos::deep_copy(mask, true);
    pc.destroy(mask.data(), n);
    const unsigned off = pc.create(kN);
    seedRange(pc, off, kN);
    seedAux(pc, off, kN);
    syncGravBH<DoublePrecision, fields::AuxConserved, fields::StdDependent>(pc);
    expectPayloadConsistent(pc, kN);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    int success = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
