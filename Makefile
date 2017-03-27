D_COMPILER=ldc2

LDMD=ldmd2

DUB_INCLUDE = -I~/.dub/packages/dstats-1.0.3/dstats/source/ -I~/.dub/packages/gsl-0.1.8/gsl/source/ -I~/.dub/packages/cblas-1.0.0/cblas/source/ -I~/.dub/packages/dyaml-0.5.3/dyaml/source/ -I~/.dub/packages/tinyendian-0.1.2/tinyendian/source/
DUB_LIBS = \
$(HOME)/.dub/packages/dstats-1.0.3/dstats/libdstats.a \
$(HOME)/.dub/packages/dyaml-0.5.3/dyaml/libdyaml.a \
$(HOME)/.dub/packages/gsl-0.1.8/gsl/libgsl.a \
$(HOME)/.dub/packages/tinyendian-0.1.2/tinyendian/libtinyendian.a

DFLAGS = -wi -I./source $(DUB_INCLUDE)
RPATH  =
LIBS   = -L=-llapacke -L=-llapack -L=-lblas -L=-lgsl -L=-lgslcblas -L=-lm -L=-lopenblas -L=-lm -L=-lgslcblas
SRC    = $(wildcard source/faster_lmm_d/*.d)
IR     = $(wildcard source/faster_lmm_d/*.ll)
BC     = $(wildcard source/faster_lmm_d/*.bc)
OBJ    = $(SRC:.d=.o)
OUT    = build/faster_lmm_d

.PHONY: profile test clean cleanIR cleanBC

debug: DFLAGS += -O0 -g -d-debug $(RPATH) -link-debuglib

release: DFLAGS += -O -release $(RPATH)

profile:  DFLAGS += -fprofile-instr-generate=fast_lmm_d-profiler.out

getIR: DFLAGS += -output-ll

getBC: DFLAGS += -output-bc

all: debug

build-setup:
	mkdir -p build/

default debug release profile getIR getBC: $(OUT)

# ---- Compile step
%.o: %.d
	$(D_COMPILER) -lib $(DFLAGS) -c $< -od=$(dir $@)

# ---- Link step
$(OUT): build-setup $(OBJ)
	$(D_COMPILER) -of=build/faster_lmm_d $(DFLAGS)  $(OBJ) $(LIBS) $(DUB_LIBS)

test:
	chmod 755 build/faster_lmm_d
	./run_tests.sh

debug-strip: debug

run-profiler: profile test
	ldc-profdata merge fast_lmm_d-profiler.out -output faster_lmm_d.profdata

install:
	install -m 0755 build/faster_lmm_d $(prefix)/bin

clean:
	rm -rf build/*
	rm -f $(OBJ) $(OUT) trace.{def,log}

cleanIR:
	rm -f $(IR) trace.{def,log}

cleanBC:
	rm -f $(BC) trace.{def,log}
