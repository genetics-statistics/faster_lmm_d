module faster_lmm_d.memory;

import core.sys.linux.config;
import core.sys.linux.sys.sysinfo;

import std.stdio;

c_ulong total_virtual_memory(){
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

c_ulong total_RAM(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong totalPhysMem = memInfo.totalram;
  totalPhysMem *= memInfo.mem_unit;
  return totalPhysMem/(8 * 1024 * 1024);
}

c_ulong total_RAM_used(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong physMemUsed = memInfo.totalram - memInfo.freeram;
  physMemUsed *= memInfo.mem_unit;
  return physMemUsed/(8 * 1024 * 1024);
}
