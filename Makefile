# Compilers
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++
# Flags
HOST_FLAGS = -O2 -std=c++17 -I./include -I$(HOME)/gtest/include
RV_FLAGS   = -march=rv64gcv -mabi=lp64d -O3 -ftree-vectorize -std=c++17 -I./include
# Directories
SRC_DIR  = src
TEST_DIR = tests
BUILD    = build

# Source files
RV_SRCS  = $(SRC_DIR)/main.cpp \
            $(SRC_DIR)/gaussian.cpp \
            $(SRC_DIR)/sobel.cpp \
            $(SRC_DIR)/magnitude.cpp \
            $(SRC_DIR)/direction.cpp \
            $(SRC_DIR)/nms.cpp \
            $(SRC_DIR)/threshold.cpp \
            $(SRC_DIR)/rvv_magnitude.cpp \
            $(SRC_DIR)/rvv_gaussian.cpp \
            $(SRC_DIR)/image_io.cpp

TEST_SRCS = $(TEST_DIR)/test_gaussian.cpp \
             $(TEST_DIR)/test_sobel.cpp \
             $(TEST_DIR)/test_magnitude.cpp \
             $(TEST_DIR)/test_direction.cpp \
             $(TEST_DIR)/test_nms.cpp \
             $(TEST_DIR)/test_threshold.cpp \
             $(SRC_DIR)/gaussian.cpp \
             $(SRC_DIR)/sobel.cpp \
             $(SRC_DIR)/magnitude.cpp \
             $(SRC_DIR)/direction.cpp \
             $(SRC_DIR)/nms.cpp \
             $(SRC_DIR)/threshold.cpp \
             $(SRC_DIR)/image_io.cpp
# QEMU settings
VLEN     ?= 256
QEMU     = qemu-riscv64
QEMU_CPU = -cpu rv64,v=true,vlen=$(VLEN)

# Targets
.PHONY: all test test_direction test_nms test_threshold \
        canny_rv run run128 run256 run512 clean sweep
all: canny_rv

# Build for RISC-V
canny_rv: $(BUILD)/canny
$(BUILD)/canny:
	mkdir -p $(BUILD)
	$(RV_CXX) $(RV_FLAGS) $(RV_SRCS) -o $(BUILD)/canny

# Run on QEMU (default VLEN=256)
run: $(BUILD)/canny
	$(QEMU) $(QEMU_CPU) $(BUILD)/canny

# Run at specific VLENs
run128: $(BUILD)/canny
	$(QEMU) -cpu rv64,v=true,vlen=128 $(BUILD)/canny

run256: $(BUILD)/canny
	$(QEMU) -cpu rv64,v=true,vlen=256 $(BUILD)/canny

run512: $(BUILD)/canny
	$(QEMU) -cpu rv64,v=true,vlen=512 $(BUILD)/canny

# Build and run host-side tests
test:
	mkdir -p $(BUILD)
	$(HOST_CXX) $(HOST_FLAGS) $(TEST_SRCS) \
	-L$(HOME)/gtest/lib -lgtest -lgtest_main -lpthread \
	-o $(BUILD)/test_runner
	./$(BUILD)/test_runner
test_direction:
	mkdir -p $(BUILD)
	$(HOST_CXX) $(HOST_FLAGS) \
	$(TEST_DIR)/test_direction.cpp \
	$(SRC_DIR)/direction.cpp \
	-L$(HOME)/gtest/lib -lgtest -lgtest_main -lpthread \
	-o $(BUILD)/test_direction
	./$(BUILD)/test_direction

test_nms:
	mkdir -p $(BUILD)
	$(HOST_CXX) $(HOST_FLAGS) \
	$(TEST_DIR)/test_nms.cpp \
	$(SRC_DIR)/nms.cpp \
	$(SRC_DIR)/direction.cpp \
	-o $(BUILD)/test_nms
	./$(BUILD)/test_nms


test_threshold:
	mkdir -p $(BUILD)
	$(HOST_CXX) $(HOST_FLAGS) \
	$(TEST_DIR)/test_threshold.cpp \
	$(SRC_DIR)/threshold.cpp \
	-o $(BUILD)/test_threshold
	./$(BUILD)/test_threshold
# Clean
clean:
	rm -rf $(BUILD)/*
# Optimization sweep builds
$(BUILD)/canny_O0:
	mkdir -p $(BUILD)
	$(RV_CXX) -march=rv64gcv -mabi=lp64d -O0 -std=c++17 -I./include \
	$(RV_SRCS) -o $(BUILD)/canny_O0

$(BUILD)/canny_O2:
	mkdir -p $(BUILD)
	$(RV_CXX) -march=rv64gcv -mabi=lp64d -O2 -std=c++17 -I./include \
	$(RV_SRCS) -o $(BUILD)/canny_O2

$(BUILD)/canny_O3:
	mkdir -p $(BUILD)
	$(RV_CXX) -march=rv64gcv -mabi=lp64d -O3 -std=c++17 -I./include \
	$(RV_SRCS) -o $(BUILD)/canny_O3

$(BUILD)/canny_O3_vec:
	mkdir -p $(BUILD)
	$(RV_CXX) -march=rv64gcv -mabi=lp64d -O3 -ftree-vectorize -fopt-info-vec \
	-std=c++17 -I./include $(RV_SRCS) -o $(BUILD)/canny_O3_vec 2>$(BUILD)/vec_report.txt

sweep: $(BUILD)/canny_O0 $(BUILD)/canny_O2 $(BUILD)/canny_O3 $(BUILD)/canny_O3_vec
