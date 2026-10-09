// Sanity test: NBodySolver should match direct N² to within the
// theta-determined error bound on a small, well-conditioned system.
//
// We run the BH pipeline first (which SFC-sorts particles via updateGrav), then
// run ryoanji::directSum on the post-sync positions/charges to obtain reference
// accelerations indexed in the same SFC slot. Comparison: per-particle L2 norm
// of the acceleration error relative to the L2 norm of the reference.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <gtest/gtest.h>

#include "Ippl.h"

#include "NBody/NBodyParticleContainer.hpp"
#include "NBody/NBodySolver.hpp"
#include "NBodyTestUtil.hpp"

#include "cstone/sfc/box.hpp"
// cstone/cuda/cuda_utils.cuh must precede direct.cuh — direct.cuh references
// kernelSuccess() but does not include its declaration (upstream omission).
// RyoanjiDirect.cu uses the same workaround. It also provides the portable
// memcpy wrappers (CUDA or HIP).
#include "cstone/cuda/cuda_utils.cuh"
#include "ryoanji/nbody/direct.cuh"
#include "ryoanji/nbody/types.h"

using ippl::nbody::DoublePrecision;
using ippl::nbody::MultipoleOrder;
using ippl::nbody::NBodySolver;
using ippl::nbody::NBodyParticleContainer;
using ippl::nbody::syncExec;
using ippl::nbody::syncGravBH;
using ippl::nbody::test::downloadDevice;
using ippl::nbody::test::uploadHost;
namespace fields = ippl::nbody::fields;

namespace {

using C = NBodyParticleContainer<DoublePrecision, 3>;

constexpr unsigned kN             = 4096;
constexpr unsigned kBucketSize    = 64;
constexpr unsigned kBucketSizeFoc = 64;
constexpr float    kTheta         = 0.5f;

/*! @brief Solve the given particles with BH and return the relative L2 error against
 *         ryoanji::directSum (same P2P kernel) over all particles.
 *
 * @param bounds  initial container box (open BCs)
 */
double relativeErrorVsDirect(const std::vector<double>& xPre, const std::vector<double>& yPre,
                             const std::vector<double>& zPre, const std::vector<double>& qPre,
                             const std::array<double, 6>& bounds, MultipoleOrder order) {
    using T = double;
    using P = DoublePrecision;
    using cstone::BoundaryType;

    const unsigned n = static_cast<unsigned>(xPre.size());

    NBodyParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1,
        kBucketSize, kBucketSizeFoc, kTheta,
        bounds,
        std::array<BoundaryType, 3>{
            BoundaryType::open, BoundaryType::open, BoundaryType::open});

    pc.create(n);

    std::vector<T> hPre(n, 1.0e-2);
    uploadHost(xPre,  getRaw<"Rx">(pc));
    uploadHost(yPre,  getRaw<"Ry">(pc));
    uploadHost(zPre,  getRaw<"Rz">(pc));
    uploadHost(hPre,  getRaw<"h">(pc));
    uploadHost(qPre,  getRaw<"charge">(pc));

    typename NBodySolver<P, 3>::Params params;
    params.G          = T(1);
    params.numShells  = 0;
    params.multipoles = order;

    NBodySolver<P, 3> solver(pc, params);
    pc.setUniformH(0.01);
    // BH consumes positions/charge/h; velocity/ID are conserved but unused here.
    syncGravBH<P, fields::StdConserved, fields::StdDependent>(pc);
    solver.runSolver();

    const unsigned start      = pc.startIndex();
    const unsigned end        = pc.endIndex();
    const unsigned nWithHalos = pc.nWithHalos();
    EXPECT_EQ(end - start, n) << "Single-rank: every particle should be locally owned.";

    // Reference: direct N² on the post-sync (SFC-sorted) positions/charges/h.
    // directKernel does `+=` into these buffers (direct.cuh:78-83), so they must
    // be zero-initialized — the DeviceVector(size, init) ctor does that.
    cstone::DeviceVector<T> refPx(nWithHalos, T(0));
    cstone::DeviceVector<T> refAx(nWithHalos, T(0));
    cstone::DeviceVector<T> refAy(nWithHalos, T(0));
    cstone::DeviceVector<T> refAz(nWithHalos, T(0));

    // Open BCs: numShells=0; box vector unused for the gravity sum but required
    // by the API. Pass the same box dimensions the container was constructed with.
    const auto box = pc.box();
    ryoanji::Vec3<T> boxL{box.lx(), box.ly(), box.lz()};

    ryoanji::directSum(
        /*first=*/start, /*last=*/end, /*numBodies=*/end,
        boxL, /*numShells=*/0,
        getRaw<"Rx">(pc), getRaw<"Ry">(pc), getRaw<"Rz">(pc),
        getRaw<"charge">(pc), getRaw<"h">(pc),
        refPx.data(), refAx.data(), refAy.data(), refAz.data());

    syncExec();

    std::vector<T> bhAx, bhAy, bhAz, dirAx, dirAy, dirAz;
    downloadDevice(getRaw<"Ex">(pc), nWithHalos, bhAx);
    downloadDevice(getRaw<"Ey">(pc), nWithHalos, bhAy);
    downloadDevice(getRaw<"Ez">(pc), nWithHalos, bhAz);
    downloadDevice(refAx.data(), nWithHalos, dirAx);
    downloadDevice(refAy.data(), nWithHalos, dirAy);
    downloadDevice(refAz.data(), nWithHalos, dirAz);

    // Mean-relative L2 error over the locally-owned range.
    long double sqErr = 0.0L;
    long double sqRef = 0.0L;
    for (unsigned j = start; j < end; ++j) {
        const long double ex = static_cast<long double>(bhAx[j]) - dirAx[j];
        const long double ey = static_cast<long double>(bhAy[j]) - dirAy[j];
        const long double ez = static_cast<long double>(bhAz[j]) - dirAz[j];
        sqErr += ex * ex + ey * ey + ez * ez;

        const long double rx = dirAx[j];
        const long double ry = dirAy[j];
        const long double rz = dirAz[j];
        sqRef += rx * rx + ry * ry + rz * rz;
    }
    EXPECT_GT(sqRef, 0.0L) << "Direct sum produced zero-norm reference — invalid input.";
    return static_cast<double>(std::sqrt(sqErr / sqRef));
}

} // namespace

TEST(NBodySolver, MatchesDirectSumOpenBC) {
    std::vector<double> xPre(kN), yPre(kN), zPre(kN), qPre(kN, 1.0);
    ::srand48(/*seed=*/424242);
    for (unsigned i = 0; i < kN; ++i) {
        xPre[i]  = drand48();
        yPre[i]  = drand48();
        zPre[i]  = drand48();
    }

    const std::array<double, 6> bounds{0.0, 1.0, 0.0, 1.0, 0.0, 1.0};
    for (MultipoleOrder order : {MultipoleOrder::Quadrupole, MultipoleOrder::DipoleQuadrupole}) {
        const double relL2 = relativeErrorVsDirect(xPre, yPre, zPre, qPre, bounds, order);
        std::printf("[NBodySolver] same-sign %s relative L2 error = %.3e\n",
                    order == MultipoleOrder::Quadrupole ? "Quadrupole" : "DipoleQuadrupole", relL2);

        // theta=0.5, uniform random: measured 5.18e-4 for both orders (same-sign cells have
        // no dipole about the |q|-weighted center). Limit ~2.5x.
        EXPECT_LT(relL2, 1.3e-3)
            << "BH-vs-direct relative L2 error " << relL2 << " exceeds 1.3e-3 for theta=" << kTheta;
    }
}

// Image-charge geometry: +1 charges uniform in the unit cube and their mirrors (-1) across the
// plane z = 0.2. Reals below the plane put mirrors into [0.2, 0.4], so cells there mix signs.
// Without the dipole term those cells are first-order wrong; with it the error must drop to the
// same-sign level.
TEST(NBodySolver, MixedSignNeedsDipoles) {
    constexpr unsigned kReal   = kN / 2;
    constexpr double   kPlaneZ = 0.2;
    std::vector<double> xPre(kN), yPre(kN), zPre(kN), qPre(kN);
    ::srand48(/*seed=*/171717);
    for (unsigned i = 0; i < kReal; ++i) {
        xPre[i] = xPre[kReal + i] = drand48();
        yPre[i] = yPre[kReal + i] = drand48();
        zPre[i]          = drand48();
        zPre[kReal + i]  = 2.0 * kPlaneZ - zPre[i];
        qPre[i]          = 1.0;
        qPre[kReal + i]  = -1.0;
    }

    const std::array<double, 6> bounds{0.0, 1.0, 0.0, 1.0, 2.0 * kPlaneZ - 1.0, 1.0};
    const double errQ =
        relativeErrorVsDirect(xPre, yPre, zPre, qPre, bounds, MultipoleOrder::Quadrupole);
    const double errDQ =
        relativeErrorVsDirect(xPre, yPre, zPre, qPre, bounds, MultipoleOrder::DipoleQuadrupole);
    std::printf("[NBodySolver] mixed-sign relative L2 error: Quadrupole = %.3e, "
                "DipoleQuadrupole = %.3e\n", errQ, errDQ);

    // Measured (GH200, 1 rank): Quadrupole 4.44e-3, DipoleQuadrupole 4.97e-4, i.e. at the
    // same-sign level (5.18e-4). The box is 1 x 1 x 1.6, so cstone's mixed-dimension SFC keys
    // give x and y one key bit less and the cells straddling the plane depend on that choice
    // (before those keys: Quadrupole 1.27e-2, DipoleQuadrupole 3.14e-4). DipoleQuadrupole must
    // stay at the same-sign level, and the gap to Quadrupole (9x) proves the dipole is used.
    EXPECT_LT(errDQ, 8e-4);
    EXPECT_GT(errQ, 5.0 * errDQ);
}

TEST(NBodySolver, DipolesRejectPeriodicBox) {
    using P = DoublePrecision;
    using cstone::BoundaryType;
    NBodyParticleContainer<P, 3> pc(
        /*rank=*/0, /*nRanks=*/1, kBucketSize, kBucketSizeFoc, kTheta,
        std::array<double, 6>{0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        std::array<BoundaryType, 3>{
            BoundaryType::periodic, BoundaryType::periodic, BoundaryType::periodic});
    typename NBodySolver<P, 3>::Params params;
    params.multipoles = MultipoleOrder::DipoleQuadrupole;
    EXPECT_THROW((NBodySolver<P, 3>(pc, params)), std::invalid_argument);
}

int main(int argc, char* argv[]) {
    // ippl::initialize wraps MPI_Init + Kokkos::initialize; cstone's Domain calls
    // MPI collectives unconditionally even on nRanks=1.
    ippl::initialize(argc, argv);
    int success = 1;
    {
        ::testing::InitGoogleTest(&argc, argv);
        success = RUN_ALL_TESTS();
    }
    ippl::finalize();
    return success;
}
