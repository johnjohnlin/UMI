# Brief
This project implements the paper "Unrolled Memory Inner-Products: An Abstract GPU Operator for Efficient Vision-Related Computations" (ICCV 2017).

# Compile test files
CMake and Glog library is required for compiling.

    mkdir build
    cmake ..
    make

The directory main/ show some examples you can program with UMI operator.

# FAQ

If you are using new Linux such as Arch, you have to install gcc/g++ version 5.
You also have to set CMAKE\_CXX\_COMPILER, CMAKE\_C\_COMPILER and CUDA\_HOST\_COMPILER accordingly.

