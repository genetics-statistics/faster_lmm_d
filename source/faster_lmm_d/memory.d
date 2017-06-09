/*
   This code is part of faster_lmm_d and published under the GPLv3
   License (see LICENSE.txt)

   Copyright © 2017 Prasun Anand & Pjotr Prins

   This module checks memory use and does generalised memory
   management for offloaded computations.
*/

module faster_lmm_d.memory;

import resusage.memory;

import std.experimental.logger;
import std.conv;
import std.stdio;

/*
 * Check available RAM
 */

void check_memory(string msg = "") {
  debug {
    SystemMemInfo sysMemInfo = systemMemInfo();
    const gb = 1024.0*1024*1024;
    auto ram_tot = sysMemInfo.totalRAM/gb;

    ProcessMemInfo procMemInfo = processMemInfo();
    auto ram_used = procMemInfo.usedRAM/gb;
    trace(msg, " - RAM used (",ram_used*100.0/ram_tot,"%) ",ram_used,"GB, total ",to!int(procMemInfo.usedVirtMem/gb),"/",to!int(ram_tot),"GB");
  }
}
