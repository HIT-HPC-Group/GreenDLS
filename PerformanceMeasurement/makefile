OPT_CUDA_DIR = /opt/cuda
ifeq (${OPT_CUDA_DIR}, ${wildcard ${OPT_CUDA_DIR}})
	CUDA_DIR = ${OPT_CUDA_DIR}
endif

LOCAL_CUDA_DIR = /usr/local/cuda
ifeq (${LOCAL_CUDA_DIR}, ${wildcard ${LOCAL_CUDA_DIR}})
	CUDA_DIR = ${LOCAL_CUDA_DIR}
endif

CUDA_INC = $(CUDA_DIR)/include
NVML_INC = $(CUDA_DIR)/targets/x86_64-linux/include
CUDA_GDK = $(CUDA_DIR)/gds
CUDA_LIB64 = $(CUDA_DIR)/lib64
CUDA_LIB = $(CUDA_DIR)/lib
CUDA_STUBS = $(CUDA_DIR)/lib64/stubs

CC = g++
CFLAGS = -Wall -O3 -Wno-unused-parameter
# SPECIALFLAGS = -lnvidia-ml -lpthread

RPATH = -Wl,-rpath=${CXX_LIB_DIR}

SRC = ./main.cpp ./PowerManager.cpp
TARGET = PerfMeasure

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -I$(NVML_INC) -I$(CUDA_INC) -I$(CUDA_GDK) -L$(CUDA_LIB64)  -L$(CUDA_STUBS) -fPIC -o $@.bin $(SRC) -lcudart -lcuda -lnvidia-ml -lpthread 

clean:
	rm -f *.o *.bin
