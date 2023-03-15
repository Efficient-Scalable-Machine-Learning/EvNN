AR ?= ar
CXX ?= g++
NVCC ?= nvcc -ccbin $(CXX)
PYTHON ?= python

ifeq ($(OS),Windows_NT)
LIBEVNN := libevnn.lib
CUDA_HOME ?= $(CUDA_PATH)
AR := lib
AR_FLAGS := /nologo /out:$(LIBEVNN)
NVCC_FLAGS := -x cu -Xcompiler "/MD"
else
LIBEVNN := libevnn.a
CUDA_HOME ?= $(CUDA_PATH)
AR ?= ar
AR_FLAGS := -crv $(LIBEVNN)
NVCC_FLAGS := -std=c++11 -x cu -Xcompiler -fPIC -lineinfo -Wno-deprecated-gpu-targets
endif

LOCAL_CUDA_CFLAGS := -I$(CUDA_HOME)/include
LOCAL_CUDA_LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcublas
LOCAL_CFLAGS := -Ilib -O3 -g
LOCAL_LDFLAGS := -L. -lcblas
GPU_ARCH_FLAGS := -gencode arch=compute_37,code=compute_37 -gencode arch=compute_60,code=compute_60 -gencode arch=compute_70,code=compute_70

# Small enough project that we can just recompile all the time.
.PHONY: all evnn evnn_pytorch examples clean

all: evnn evnn_pytorch examples

# Dependencies handled by setup.py
evnn_pytorch: 
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py bdist_wheel)
	@cp $(TMP)/dist/*.whl .
	@rm -rf $(TMP)

dist:
	@$(eval TMP := $(shell mktemp -d))
	@cp -r . $(TMP)
	@cp build/MANIFEST.in $(TMP)
	@cat build/common.py build/setup.pytorch.py > $(TMP)/setup.py
	@(cd $(TMP); $(PYTHON) setup.py -q sdist)
	@cp $(TMP)/dist/*.tar.gz .
	@rm -rf $(TMP)

evnn: 
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/egru_forward_gpu.cu.cc -o lib/egru_forward_gpu.o $(NVCC_FLAGS) $(LOCAL_CUDA_CFLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/egru_backward_gpu.cu.cc -o lib/egru_backward_gpu.o $(NVCC_FLAGS) $(LOCAL_CUDA_CFLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/egru_forward_cpu.cc -o lib/egru_forward_cpu.o $(NVCC_FLAGS) $(LOCAL_CUDA_CFLAGS) $(LOCAL_CFLAGS)
	$(NVCC) $(GPU_ARCH_FLAGS) -c lib/egru_backward_cpu.cc -o lib/egru_backward_cpu.o $(NVCC_FLAGS) $(LOCAL_CUDA_CFLAGS) $(LOCAL_CFLAGS)
	$(AR) $(AR_FLAGS) lib/*.o

evnn_cpu:
	$(CXX)  -c lib/egru_forward_cpu.cc $(LOCAL_LDFLAGS) -o lib/egru_forward_cpu.o  -fPIC $(LOCAL_CFLAGS) 
	$(CXX)  -c lib/egru_backward_cpu.cc $(LOCAL_LDFLAGS) -o lib/egru_backward_cpu.o  -fPIC $(LOCAL_CFLAGS)
	$(AR) $(AR_FLAGS) lib/*.o

examples: evnn
	$(CXX) -std=c++11 examples/egru.cc $(LIBEVNN) -Ieigen3 $(LOCAL_CUDA_CFLAGS) $(LOCAL_CFLAGS) $(LOCAL_CUDA_LDFLAGS) $(LOCAL_LDFLAGS) -o evnn_egru -Wno-ignored-attributes 

clean:
	rm -fr benchmark_lstm benchmark_gru evnn_egru evnn_*.whl evnn_*.tar.gz
	find . \( -iname '*.o' -o -iname '*.so' -o -iname '*.a' -o -iname '*.lib' \) -delete
