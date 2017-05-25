module faster_lmm_d.memory;

import resusage.memory;
import core.sys.linux.config;
import core.sys.linux.sys.sysinfo;

import std.conv;
import std.stdio;

c_ulong virtual_memory_total(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong totalVirtualMem = memInfo.totalram;
  totalVirtualMem += memInfo.totalswap;
  totalVirtualMem *= memInfo.mem_unit;
  return totalVirtualMem/(8 * 1024 * 1024);
}

c_ulong virtual_memory_used(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong virtualMemUsed = memInfo.totalram - memInfo.freeram;
  virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
  virtualMemUsed *= memInfo.mem_unit;
  writeln(virtualMemUsed/(8 * 1024 * 1024));
  return virtualMemUsed/(8 * 1024 * 1024);
}

c_ulong ram_total(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong totalPhysMem = memInfo.totalram;
  totalPhysMem *= memInfo.mem_unit;
  return totalPhysMem/(8 * 1024 * 1024);
}

c_ulong ram_used(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong physMemUsed = memInfo.totalram - memInfo.freeram;
  physMemUsed *= memInfo.mem_unit;
  return physMemUsed/(8 * 1024 * 1024);
}

void check_memory(string msg = "check_memory") {
  stderr.writeln(msg);
  SystemMemInfo sysMemInfo = systemMemInfo();
  auto gb = 1024.0*1024*1025;
  auto ram_tot = sysMemInfo.totalRAM/gb;

  ProcessMemInfo procMemInfo = processMemInfo();
  auto ram_used = procMemInfo.usedRAM/gb;
  stderr.writeln("RAM used (",ram_used*100.0/ram_tot,"%) ",ram_used,"GB, total ",to!int(ram_tot),"GB");
}
