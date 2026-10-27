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
#pragma once 

#include <memory>

#include "ryoanji/interface/heaviside_holder.cuh"
#include "ryoanji/nbody/ewald.h"

#include "NBody/NBodyParticleContainer.hpp"

namespace ippl::nbody {

// Multipole expansion of the tree cells. Quadrupole (ryoanji::CartesianQuadrupole) has no
// dipole term: exact when all sources share one sign (the dipole about the |q|-weighted
// expansion center vanishes), but first-order wrong for cells that mix signs (e.g. image
// charges). DipoleQuadrupole (ryoanji::CartesianMDQpole) adds the dipole; open boxes only,
// ryoanji's Ewald summation exists for Quadrupole alone.
// Heaviside (ryoanji::HeavisideQpole, GPU only, open boxes): sources move along z with their own
// rapidity (container field w, relative to Params::heaviside.frame); the field is that of
// uniformly moving charges, written to Ex/Ey/Ez (tree-frame E~) and Cx/Cy (second channel). Needs
// the fields::RelConserved / fields::RelDependent sync.
enum class MultipoleOrder { Quadrupole, DipoleQuadrupole, Heaviside };

// Barnes-Hut field solver: build local BH tree, P2M + M2M upsweep, traverse
// to write Ex/Ey/Ez. When the container's box is fully periodic the canonical
// Ewald correction (real-space erfc tail + reciprocal-space sum) is appended
// after the truncated near-field lattice sum; for open BCs Ewald is skipped.
//
// Templated on a precision policy P (see NBody/BHPrecision.hpp). The five
// ryoanji scalar template parameters (Tc, Th, Tm, Ta, Tf) plus the multipole
// value type (Tmm) all come from P, so a single template parameter selects
// one of the three pre-compiled ryoanji traversal instantiations.
template <class P, unsigned Dim>
class NBodySolver {
    static_assert(Dim == 3, "NBodySolver requires Dim == 3");

public:
    using Container = NBodyParticleContainer<P, Dim>;
    using Tc        = typename P::Tc;

    struct Params {
        Tc    G         = Tc(1);  // Coulomb / gravitational prefactor (position units)
        float theta     = 0.5f;   // multipole-acceptance angle
        int   numShells = 1;      // BH near-field image-lattice extent (periodic only)
        ryoanji::EwaldSettings ewaldSettings;
        MultipoleOrder multipoles = MultipoleOrder::Quadrupole;  // fixed per solver instance
        // MultipoleOrder::Heaviside only: frame, rapidity MAC (may change between solves)
        ryoanji::HeavisideParams<typename P::Tf> heaviside;
        // Consistency checks of the sources every solve (positions against the box, target-group
        // extents), printed as [bh.check]. A failed traversal always reports them.
        bool checks = false;
    };

    NBodySolver(Container& pc, Params params);
    ~NBodySolver();

    NBodySolver(const NBodySolver&)            = delete;
    NBodySolver& operator=(const NBodySolver&) = delete;
    NBodySolver(NBodySolver&&) noexcept;
    NBodySolver& operator=(NBodySolver&&) noexcept;

    // warmup=true skips the per-scope bh.* IpplTimings and the per-step stats
    // printout. Used by NBodyManager::pre_run() so the initial t=0 solve does
    // not pollute the per-step BH timing accumulators (the "firstSolve" outer
    // timer in pre_run already captures init cost separately).
    void runSolver(bool warmup = false);

    // MultipoleOrder::Heaviside: frame and rapidity MAC of the next runSolver.
    void setHeavisideParams(const ryoanji::HeavisideParams<typename P::Tf>& params);

    // MultipoleOrder::Heaviside: remote leaves of the last solve that were applied as multipoles
    // although they failed the rapidity-scaled MAC (cstone's LET uses the unscaled MAC; 0 on one
    // rank). Summed over ranks.
    unsigned long long remoteMacFailures() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ippl::nbody