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

import faster_lmm_d.cuda;
import faster_lmm_d.dmatrix;

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

alias RAM_PTR = ulong;
alias GPU_PTR = ulong;
alias CachedPtr = Tuple!(GPU_PTR,size_t);

CachedPtr[RAM_PTR] ptr_cache;
enum CacheMsg { Init, Store, Get };

alias CacheUpdateMsg = Tuple!(CacheMsg,"msg",RAM_PTR,"ram_ptr",size_t,"size");

void offload_init(uint device)
  in {
    enforce(device==0);
  }
  body {
    trace("Initialize cache");
    auto pid = spawnLinked(&spawnedReceiver);
    register("MemManageGPU",pid);
    pid.send(thisTid,CacheUpdateMsg(CacheMsg.Init,cast(RAM_PTR)null,0));
    enforce(receiveOnly!(bool));
    pid.send(thisTid,CacheUpdateMsg(CacheMsg.Init,cast(RAM_PTR)null,0));
    enforce(receiveOnly!(bool));
    trace("Cache initialized");
  }


void spawnedReceiver()
{
  for (;;) {
    auto m = receiveOnly!(Tid,CacheUpdateMsg);
    auto pid = locate("MemManageGPU");
    trace("pid",pid,"tid",m[0]);
    trace(m);
    m[0].send(true);
    trace("EXIT");
  }
  /*
           (Tid tid, CacheMsg i) {
              trace("Received ", i, " on ",tid);
              assert(tid == parent);
              return;
            },
            (Tid tid2, CacheUpdateMsg msg) {
              trace("Received update ",msg.msg, " on ",tid2);
              if (!(msg.ram_ptr in ptr_cache)) {
                trace("Add pointer!");
                auto gpu_ptr = gpu_malloc(msg.size);
                ptr_cache[msg.ram_ptr] = tuple(cast(GPU_PTR)gpu_ptr,msg.size);
              }
              assert(tid2 == parent);
              send(tid2, true);
              return;
            }

    );
    enforce(false); // not supposed to get here
  */
}

/*
 * Store a block of data if not already there
 */
void offload_cache(const(void *)ram_ptr, size_t size) {
  auto pid = locate("MemManageGPU");
  trace("pid",pid,"tid",thisTid);
  pid.send(thisTid,CacheUpdateMsg(CacheMsg.Init,cast(RAM_PTR)null,0));
  trace("Waiting");
  enforce(receiveOnly!(bool));
}

void offload_cache(DMatrix m) {
  offload_cache(m.elements.ptr, m.size);
}

/*
 * Fetch a pointer. Returns null if not found.
 */
GPU_PTR offload_get_ptr(RAM_PTR ram_ptr, size_t size) {
  auto pid = locate("MemManageGPU");
  trace("pid",pid,"tid",thisTid);
  pid.send(thisTid,CacheUpdateMsg(CacheMsg.Get,ram_ptr,size));
  trace("Waiting");
  enforce(receiveOnly!(bool));
  return cast(GPU_PTR)null;
}

GPU_PTR offload_get_ptr(DMatrix m) {
  return offload_get_ptr(cast(RAM_PTR)m.elements.ptr, m.size);
}
