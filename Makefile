D_COMPILER=ldc2

LDMD=ldmd2

DUB_INCLUDE = -I~/.dub/packages/dstats-1.0.3/dstats/source/ -I~/.dub/packages/gsl-0.1.8/gsl/source/ -I~/.dub/packages/cblas-1.0.0/cblas/source/ -I~/.dub/packages/dyaml-0.5.3/dyaml/source/ -I~/.dub/packages/tinyendian-0.1.2/tinyendian/source/
DUB_LIBS = \
$(HOME)/.dub/packages/dstats-1.0.3/dstats/libdstats.a \
$(HOME)/.dub/packages/dyaml-0.5.3/dyaml/libdyaml.a \
$(HOME)/.dub/packages/gsl-0.1.8/gsl/libgsl.a \
$(HOME)/.dub/packages/lapack-0.1.2/lapack/liblapack.a \
$(HOME)/.dub/packages/tinyendian-0.1.2/tinyendian/libtinyendian.a

DFLAGS = -wi -I./source $(DUB_INCLUDE)
# DLIBS  = $(LDC_LIB_PATH)/libphobos2-ldc.a $(LDC_LIB_PATH)/libdruntime-ldc.a $(DUB_LIBS)
# DLIBS_DEBUG = $(LDC_LIB_PATH)/libphobos2-ldc-debug.a $(LDC_LIB_PATH)/libdruntime-ldc-debug.a $(DUB_LIBS)
RPATH  =
LIBS   = -L-lgsl -L-lopenblas -L-llapacke
SRC    = $(wildcard source/faster_lmm_d/*.d)
OBJ    = $(SRC:.d=.o)
OUT    = build/faster_lmm_d

.PHONY: profile test clean
# The development options are run from ~/.guix-profile and need to inject the RPATH
debug: DFLAGS += -O0 -g -d-debug $(RPATH) -link-debuglib

release: DFLAGS += -O -release $(RPATH)

profile:  DFLAGS += -fprofile-instr-generate=fast_lmm_d-profiler.out $(RPATH)

all: debug

build-setup:
  mkdir -p build/

default debug release profile: $(OUT)

# ---- Compile step
%.o: %.d
  $(D_COMPILER) $(DFLAGS) -c $< -od=$(dir $@)

# ---- Link step
$(OUT): build-setup $(OBJ)
  $(D_COMPILER) -lib $(DFLAGS) -of=build/faster_lmm_d $(OBJ) $(LIBS) $(DUB_LIBS)

test:
  chmod 755 build/faster_lmm_d
  ./run_tests.sh

debug-strip: debug

run-profiler: profile test
  ldc-profdata merge fast_lmm_d-profiler.out -output fast_lmm_d.profdata

install:
  install -m 0755 build/faster_lmm_d $(prefix)/bin

clean:
  rm -rf build/*
  rm -f $(OBJ) $(OUT) trace.{def,log}