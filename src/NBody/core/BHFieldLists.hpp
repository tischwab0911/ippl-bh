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
 * @brief Declare standard field lists for tests
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include "cstone/util/value_list.hpp"

namespace ippl::nbody::fields {

// Per-driver conserved/dependent field selections, in the spirit of sphexa's
// per-propagator ConservedFields/DependentFields. Field names resolve against
// NBodyParticleContainer::fieldNames. The Core fields (Rx, Ry, Rz, h, charge)
// are the structural arguments of domain.sync(Grav) and are not listed here.
//
//   Conserved: SFC-permuted and redistributed in lockstep with positions.
//   Dependent: recomputed each step (the BH force output); resized to
//              nParticlesWithHalos after sync, not transported.
//
// A driver that needs a different set declares its own util::FieldList<...> and
// adds the (Precision, Conserved, Dependent) combo to the explicit-instantiation
// list in NBodySync.cpp.
using StdConserved = util::FieldList<"Px", "Py", "Pz", "ID">;
using StdDependent = util::FieldList<"Ex", "Ey", "Ez", "ugrav">;

// Momenta plus the application payload slots, used by drivers that hand the
// owned particle set back to the application after a sync (e.g. OPAL-X space
// charge carries its time step in aux0 and its int64 particle ID in aux64).
// ID is omitted: such drivers identify particles through aux64.
using AuxConserved = util::FieldList<"Px", "Py", "Pz", "aux0", "aux64">;

}  // namespace ippl::nbody::fields
