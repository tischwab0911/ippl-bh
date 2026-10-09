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
 * @brief Small wrapper for IPPL-Style scoped timer for NBody simulations
 *
 * @author Timo Schwab, <tischwab@ethz.ch>
 */
#pragma once

#include "Utility/IpplTimings.h"

#include "NBody/core/Accelerator.hpp"

namespace ippl::nbody {

// Times a scope including the trailing device synchronize. On GPU builds the
// dtor blocks on syncExec() so the elapsed time covers async kernel completion;
// on the CPU build syncExec() is a no-op (kernels are synchronous).
// collect=false skips both start/stop, leaving the timer untouched.
class GpuTimer {
public:
    explicit GpuTimer(IpplTimings::TimerRef ref, bool collect = true)
        : ref_(ref), collect_(collect) {
        if (collect_) { IpplTimings::startTimer(ref_); }
    }
    ~GpuTimer() {
        syncExec();
        if (collect_) { IpplTimings::stopTimer(ref_); }
    }
    GpuTimer(const GpuTimer&)            = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

private:
    IpplTimings::TimerRef ref_;
    bool                  collect_;
};

} // namespace ippl::nbody