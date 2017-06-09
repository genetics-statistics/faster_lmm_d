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
import std.algorithm;
import std.concurrency;
import std.conv;
import std.exception;
import std.parallelism;
import std.stdio;
import std.typecons;
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

/*
 * Memory management. Essentially RAM on the host and RAM on the GPU
 * gets tracked through a list of pointers. When GPU RAM gets
 * exhausted we start deleting stuff by the oldest access (a second
 * index list). This list may be accessed from multiple threads,
 * therefore it is implemented as an actor.
 *
 * One complication is that GPU RAM may be divided for two (or more)
 * devices, so we have a list per device (currently only one).
 */

alias RAM_PTR = void *;
alias DEV_PTR = void *;
// alias GPU_PTRS = Tuple!(GPU_PTR, GPU_PTR, GPU_PTR);
// alias GPU_PTRS_SIZE = Tuple!(size_t, size_t, size_t);

alias CachedPtr = Tuple!(DEV_PTR,size_t);
CachedPtr[RAM_PTR] ptr_list;
enum CacheMsg { CacheInit };

void init_offload_memory(uint device) {
  trace("Initialize cache");
  Tid pid = spawn(&spawnedReceiver, thisTid);
  send(pid, CacheMsg.CacheInit);
  enforce(receiveOnly!(bool));
  trace("Cache initialized");
}

void spawnedReceiver(Tid ownerTid)
{
    // Receive a message from the owner thread.
    receive(
        (CacheMsg i) { writeln("Received the number ", i);}
    );

    // Send a message back to the owner thread
    // indicating success.
    send(ownerTid, true);
}
