module faster_lmm_d.memory;

import resusage.memory;

import std.experimental.logger;
import std.conv;
import std.stdio;

void check_memory(string msg = "") {
  SystemMemInfo sysMemInfo = systemMemInfo();
  auto gb = 1024.0*1024*1025;
  auto ram_tot = sysMemInfo.totalRAM/gb;

  ProcessMemInfo procMemInfo = processMemInfo();
  auto ram_used = procMemInfo.usedRAM/gb;
  trace(msg, " - RAM used (",ram_used*100.0/ram_tot,"%) ",ram_used,"GB, total ",to!int(procMemInfo.usedVirtMem/gb),"/",to!int(ram_tot),"GB");
}
