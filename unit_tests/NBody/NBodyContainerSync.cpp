// Sanity test for NBodyParticleContainer<DoublePrecision, 3>.
// Verifies that updateBH<ConservedFields>(...) (which wraps cstone::Domain::sync)
// permutes the container-resident ID array in lockstep with positions.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"
#include "cstone/sfc/box.hpp"

#include "NBody/NBodyParticleContainer.hpp"
#include "NBodyTestUtil.hpp"

using ippl::nbody::DoublePrecision;
using ippl::nbody::NBodyParticleContainer;
using ippl::nbody::updateBH;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;
namespace fields = ippl::nbody::fields;

namespace {

using PC     = NBodyParticleContainer<DoublePrecision, 3>;
using IdType = PC::IdType;

constexpr unsigned kN              = 8192;
constexpr unsigned kBucketSize     = 64;
constexpr unsigned kBucketSizeFoc  = 64;
constexpr float    kTheta          = 0.5f;

void verifyLockstep(PC& pc,
                    const std::vector<double>& xPre,
                    const std::vector<double>& yPre,
                    const std::vector<double>& zPre) {
    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();

    ASSERT_LE(start, end);
    ASSERT_LE(end, nWithHalos);
    EXPECT_EQ(end - start, kN) << "Single-rank: every input particle should be locally owned.";

    std::vector<double> xPost, yPost, zPost;
    std::vector<IdType> idPost;
    downloadDevice(ippl::nbody::getRaw<"Rx">(pc), nWithHalos, xPost);
    downloadDevice(ippl::nbody::getRaw<"Ry">(pc), nWithHalos, yPost);
    downloadDevice(ippl::nbody::getRaw<"Rz">(pc), nWithHalos, zPost);
    downloadDevice(ippl::nbody::getRaw<"ID">(pc), nWithHalos, idPost);

    std::unordered_set<IdType> seen;
    seen.reserve(end - start);
    for (unsigned j = start; j < end; ++j) {
        const IdType ii = idPost[j];
        ASSERT_LT(ii, kN) << "ID out of range at slot " << j;
        ASSERT_TRUE(seen.insert(ii).second) << "Duplicate ID " << ii << " at slot " << j;
        EXPECT_DOUBLE_EQ(xPost[j], xPre[ii]) << "Rx mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(yPost[j], yPre[ii]) << "Ry mismatch at j=" << j << " ID=" << ii;
        EXPECT_DOUBLE_EQ(zPost[j], zPre[ii]) << "Rz mismatch at j=" << j << " ID=" << ii;
    }
    EXPECT_EQ(seen.size(), end - start) << "ID set must be a permutation of [0, N).";
}

} // namespace

TEST(NBodyParticleContainer, SyncPermutesAttributesInLockstep) {
    using cstone::BoundaryType;

    PC pc(/*rank=*/0, /*nRanks=*/1,
          kBucketSize, kBucketSizeFoc, kTheta,
          std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
          std::array<BoundaryType, 3>{
              BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(kN);

    std::vector<double> xPre(kN), yPre(kN), zPre(kN), hPre(kN, 1.0e-2);
    std::vector<IdType> idPre(kN);
    ::srand48(/*seed=*/424242);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
        idPre[i] = static_cast<IdType>(i);
    }

    uploadHost(xPre,  ippl::nbody::getRaw<"Rx">(pc));
    uploadHost(yPre,  ippl::nbody::getRaw<"Ry">(pc));
    uploadHost(zPre,  ippl::nbody::getRaw<"Rz">(pc));
    uploadHost(hPre,  ippl::nbody::getRaw<"h">(pc));
    uploadHost(idPre, ippl::nbody::getRaw<"ID">(pc));

    updateBH<DoublePrecision, fields::StdConserved>(pc);
    verifyLockstep(pc, xPre, yPre, zPre);

    updateBH<DoublePrecision, fields::StdConserved>(pc);
    verifyLockstep(pc, xPre, yPre, zPre);
}

// Leaf-based softening (setLeafBasedH) uses cellEdge(): the geometric-mean edge of a
// level-L octree cell. cstone's mixed-dimension SFC keys give short box axes fewer key
// bits, so cells in elongated boxes are not box/2^L; compare against the physical node
// size cstone itself derives (hilbertIBox + centerAndSize) for cubic and elongated boxes.
TEST(NBodyParticleContainer, CellEdgeMatchesMixedDimensionNodes) {
    using KeyType = std::uint64_t;
    using T       = double;

    const unsigned maxLevel = cstone::maxTreeLevel<KeyType>{};
    const std::vector<cstone::Box<T>> boxes{
        cstone::Box<T>(0, 1),                                // cube: no reductions
        cstone::Box<T>(0, 1, 0, 0.015625, 0, 0.00390625),    // reductions {0, 6, 8}
        cstone::Box<T>(-1e-3, 1e-3, -1e-3, 1e-3, 0.0, 0.5),  // long bunch along z
    };

    for (const auto& box : boxes) {
        const auto axesBits   = box.getBoxDimBits(maxLevel);
        const auto reductions = cstone::AxesBits{maxLevel, maxLevel, maxLevel} - axesBits;
        const T    cbrtVol    = std::cbrt(box.lx() * box.ly() * box.lz());

        for (unsigned level = 0; level <= 12; ++level) {
            const auto nodeBox = cstone::sfcIBox(cstone::sfcKey(KeyType(0)), level, axesBits);
            [[maybe_unused]] const auto [center, halfSize] =
                cstone::centerAndSize<KeyType>(nodeBox, box);
            const T    reference = 2 * std::cbrt(halfSize[0] * halfSize[1] * halfSize[2]);

            EXPECT_NEAR(ippl::nbody::detail::cellEdge(cbrtVol, level, reductions) / reference,
                        1.0, 1e-12)
                << "box " << box.lx() << " x " << box.ly() << " x " << box.lz() << ", level "
                << level;
        }
    }
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
