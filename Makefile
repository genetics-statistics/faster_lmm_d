# To build faster_lmm_d:
#
#   make [CUDA=1] [VALIDATE=1] [FORCE_DUPLICATE=1] [debug|release|check]
#
# run with
#
#   ./build/faster_lmm_d
#
# where
#
#   CUDA=1            builds with CUDA support
#   FORCE_DUPLICATE=1 prevents overwrites (purity)
#   PARALLEL=1        switch on parallel compute
#   VALIDATE=1        validates matrix multiplication
#
#  current combinations are
#
#   Default:          -
#   Parallel version: PARALLEL=1
#   CUDA version:     CUDA=1

D_COMPILER=ldc2

LDMD=ldmd2

BACKEND_FLAG='-d-version=CPU'

ifeq ($(CUDA),1)
  BACKEND_FLAG=-d-version=CUDA
endif

ifeq ($(ARRAYFIRE),1)
  BACKEND_FLAG=-d-version=ARRAYFIRE
endif

DUB_INCLUDE = \
-I~/.dub/packages/dstats-1.0.3/dstats/source/ \
-I~/.dub/packages/gsl-0.1.8/gsl/source/ \
-I~/.dub/packages/cblas-1.0.0/cblas/source/ \
-I~/.dub/packages/dyaml-0.5.3/dyaml/source/ \
-I~/.dub/packages/tinyendian-0.1.2/tinyendian/source/ \
-I~/.dub/packages/resusage-0.2.4/resusage/source/

DUB_LIBS = \
$(HOME)/.dub/packages/dstats-1.0.3/dstats/libdstats.a \
$(HOME)/.dub/packages/dyaml-0.5.3/dyaml/libdyaml.a \
$(HOME)/.dub/packages/gsl-0.1.8/gsl/libgsl.a \
$(HOME)/.dub/packages/tinyendian-0.1.2/tinyendian/libtinyendian.a \
$(HOME)/.dub/packages/resusage-0.2.4/resusage/lib/libresusage.a

DFLAGS = -wi -I./source $(DUB_INCLUDE)
RPATH  =
LIBS   = -L=-llapacke -L=-llapack -L=-lblas -L=-lgsl -L=-lgslcblas -L=-lm -L=-lopenblas -L=-lm -L=-lgslcblas
SRC    = $(wildcard source/faster_lmm_d/*.d)
IR     = $(wildcard source/faster_lmm_d/*.ll)
BC     = $(wildcard source/faster_lmm_d/*.bc)
OBJ    = $(SRC:.d=.o)
OUT    = build/faster_lmm_d

ifeq ($(CUDA),1)
  DUB_INCLUDE += -I~/.dub/packages/cuda_d-0.1.0/cuda_d/source/
  DUB_LIBS    += $(HOME)/.dub/packages/cuda_d-0.1.0/cuda_d/libcuda_d.a
  LIBS        += -L=-lcuda -L=-lcublas -L=-lcudart
endif

ifeq ($(ARRAYFIRE),1)
  LIBS        += -L=-lafcuda
endif

debug: DFLAGS += -O0 -g -d-debug $(RPATH) -link-debuglib $(BACKEND_FLAG)

release: DFLAGS += -O -release $(RPATH)

profile: DFLAGS += -fprofile-instr-generate=fast_lmm_d-profiler.out

getIR: DFLAGS += -output-ll

getBC: DFLAGS += -output-bc

gperf: LIBS += -L=-lprofiler

gperf: DUB_INCLUDE += -I~/.dub/packages/gperftools_d-0.1.0/gperftools_d/source/

gperf: DUB_LIBS += $(HOME)/.dub/packages/gperftools_d-0.1.0/gperftools_d/libgperftools_d.a

.PHONY: profile test clean cleanIR cleanBC gperf

all: debug

build-setup:
	mkdir -p build/

build-cuda-setup:
	mkdir -p build/cuda/

ifeq ($(FORCE_DUPLICATE),1)
  DFLAGS += -d-version=FORCE_DUPLICATE
endif
ifeq ($(PARALLEL),1)
  DFLAGS += -d-version=PARALLEL
endif
ifeq ($(VALIDATE),1)
  DFLAGS += -d-version=VALIDATE
endif


default debug release profile getIR getBC gperf: $(OUT)

# ---- Compile step
%.o: %.d
	$(D_COMPILER) -lib $(DFLAGS) -c $< -od=$(dir $@) $(BACKEND_FLAG)

# ---- Link step
$(OUT): build-setup $(OBJ)
	$(D_COMPILER) -of=build/faster_lmm_d $(DFLAGS)  $(OBJ) $(LIBS) $(DUB_LIBS) $(BACKEND_FLAG)

test:
	chmod 755 build/faster_lmm_d
	./run_tests.sh

debug-strip: debug

run-profiler: profile test
	ldc-profdata merge fast_lmm_d-profiler.out -output faster_lmm_d.profdata

install:
	install -m 0755 build/faster_lmm_d $(prefix)/bin

run-gperf: gperf
	$ CPUPROFILE=./prof.out ./run_tests.sh
	pprof --gv build/faster_lmm_d ./prof.out

clean:
	rm -rf build/*
	rm -f $(OBJ) $(OUT) trace.{def,log}

cleanIR:
	rm -f $(IR) trace.{def,log}

cleanBC:
	rm -f $(BC) trace.{def,log}
