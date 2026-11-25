CC ?= clang
UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  BLAS_LIB = -framework Accelerate
  NATIVE = -mcpu=native
else
  BLAS_LIB = -lopenblas
  NATIVE = -march=native
endif

all: alexnet_O0 alexnet_O3 alexnet_blas

# Same source, three builds:
alexnet_O0: alexnet.c         # direct loops, compiler optimisation off
	$(CC) -O0 -o $@ $< -lm

alexnet_O3: alexnet.c         # direct loops, -O3 + auto-vectorisation for this CPU
	$(CC) -O3 $(NATIVE) -ffast-math -o $@ $< -lm

alexnet_blas: alexnet.c       # im2col + BLAS sgemm
	$(CC) -O3 $(NATIVE) -DUSE_BLAS -o $@ $< -lm $(BLAS_LIB)

clean:
	rm -f alexnet_O0 alexnet_O3 alexnet_blas

.PHONY: all clean
