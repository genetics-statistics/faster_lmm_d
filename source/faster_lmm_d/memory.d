module faster_lmm_d.memory;

import core.sys.linux.config;
import core.sys.linux.sys.sysinfo;

import std.stdio;


void tota_virtual_memory(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong totalVirtualMem = memInfo.totalram;
  totalVirtualMem += memInfo.totalswap;
  totalVirtualMem *= memInfo.mem_unit;
  writeln(totalVirtualMem);
}

void virtual_memory_used(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong virtualMemUsed = memInfo.totalram - memInfo.freeram;
  virtualMemUsed += memInfo.totalswap - memInfo.freeswap;
  virtualMemUsed *= memInfo.mem_unit;
  writeln(virtualMemUsed);
}

void total_RAM(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong totalPhysMem = memInfo.totalram;
  totalPhysMem *= memInfo.mem_unit;
  writeln(totalPhysMem);
}

void total_RAM_used(){
  sysinfo_ memInfo;
  sysinfo (&memInfo);
  c_ulong physMemUsed = memInfo.totalram - memInfo.freeram;
  physMemUsed *= memInfo.mem_unit;
  writeln(physMemUsed);
}
