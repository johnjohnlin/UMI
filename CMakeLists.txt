cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(UMI)
set(CMAKE_BUILD_TYPE DEBUG)
set(SRC_DIR ${UMI_SOURCE_DIR}/src)
set(MAIN_DIR ${UMI_SOURCE_DIR}/main)
include_directories(${SRC_DIR})
set(SRC_FILES
	${SRC_DIR}/UMI.cpp
)
set(MAIN_EXES
	bilateral
	filter3x3
	wfilter3x3
	gemm
	full_me
	fprop
	cost_volume
	iimg
)
add_custom_target(test-gpu)
add_custom_target(test-gpu-strong)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
find_package(CUDA)
set(CUDA_NVCC_FLAGS --std=c++11 -arch sm_50)
foreach(MAIN_EXE ${MAIN_EXES})
	cuda_add_executable(
		${MAIN_EXE}_gpu
		${MAIN_DIR}/${MAIN_EXE}.cu
		${SRC_FILES}
	)
	set_property(TARGET ${MAIN_EXE}_gpu PROPERTY CXX_STANDARD 11)
	add_custom_command(TARGET test-gpu COMMAND ./${MAIN_EXE}_gpu)
	add_custom_command(TARGET test-gpu-strong COMMAND cuda-memcheck ./${MAIN_EXE}_gpu)
	add_dependencies(test-gpu ${MAIN_EXE}_gpu)
endforeach()
