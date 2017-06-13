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


alias OFFLOAD_PTR = ulong;

version(CUDA) {

import faster_lmm_d.cuda;

alias RAM_PTR = ulong;
alias CachedPtr = Tuple!(OFFLOAD_PTR,"gpu_ptr",size_t,"size");

OFFLOAD_PTR test_ptr;

CachedPtr[RAM_PTR] ptr_cache;
enum CacheMsg { Init, Destroy, Store, Get };

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
    trace("Cache initialized");
  }

void offload_destroy(uint device) {
  trace("Terminate");
  auto pid = locate("MemManageGPU");
  pid.send(thisTid,CacheUpdateMsg(CacheMsg.Destroy,cast(RAM_PTR)null,0));
  enforce(receiveOnly!(bool));
}

void spawnedReceiver()
{
  for (;;) {
    auto m = receiveOnly!(Tid,CacheUpdateMsg);
    trace(m);
    auto tid = m[0];
    auto message = m[1];
    auto ram_ptr = message.ram_ptr;
    auto size = message.size;
    final switch(message.msg) {
    case CacheMsg.Init:
      tid.send(true);
      break;
    case CacheMsg.Destroy:
      foreach(k, v; ptr_cache) {
        gpu_free(cast(double *)v.gpu_ptr);
      }
      tid.send(true);
      return;
    case CacheMsg.Store:
      if (!(ram_ptr in ptr_cache)) {
        trace("Storing pointer ",ram_ptr);
        auto gpu_ptr = cast(OFFLOAD_PTR)gpu_malloc(size);
        cuda_copy_ram_to_gpu(cast(GPU_PTR)gpu_ptr,cast(double *)ram_ptr,size);
        ptr_cache[ram_ptr] = tuple(gpu_ptr,size);
      }
      tid.send(true);
      break;
    case CacheMsg.Get:
      trace("Fetching pointer ",ram_ptr);
      auto gpu_ptr = ( ram_ptr in ptr_cache ? ptr_cache[ram_ptr].gpu_ptr : cast(OFFLOAD_PTR)null);
      tid.send(gpu_ptr);
      break;
    }
    trace("Wait for next message");
  }
}

/*
 * Store a block of data if not already there
 */
void offload_cache(const(void *)ram_ptr, size_t size) {
  auto pid = locate("MemManageGPU");
  trace("pid",pid,"tid",thisTid);
  pid.send(thisTid,CacheUpdateMsg(CacheMsg.Store,cast(RAM_PTR)ram_ptr,size));
  trace("Waiting");
  enforce(receiveOnly!(bool));
}

void offload_cache(const DMatrix m) {
  offload_cache(m.elements.ptr, m.size);
}

/*
 * Fetch a pointer. Returns null if not found.
 */
OFFLOAD_PTR offload_get_ptr(RAM_PTR ram_ptr, size_t size) {
  auto pid = locate("MemManageGPU");
  trace("pid",pid,"tid",thisTid);
  pid.send(thisTid,CacheUpdateMsg(CacheMsg.Get,ram_ptr,size));
  trace("Waiting");
  return receiveOnly!(OFFLOAD_PTR);
}

OFFLOAD_PTR offload_get_ptr(const DMatrix m) {
  return offload_get_ptr(cast(RAM_PTR)m.elements.ptr, m.size);
}

} else {

  void offload_init(uint device) {};
  // void offload_cache(const(void *)ram_ptr, size_t size) {};
  void offload_cache(const DMatrix m) {};
  OFFLOAD_PTR offload_get_ptr(const DMatrix m) { return cast(OFFLOAD_PTR)0; };

 }
