# English
## Brief
This project implements the *UMI Operator* of the paper "Unrolled Memory Inner-Products: An Abstract GPU Operator for Efficient Vision-Related Computations" (ICCV 2017).

## Compile Test Files
CMake and Glog library is required for compiling.

    mkdir build
    cmake ..
    make

The directory main/ show some examples you can program with *UMI operator*.
I will prepare more document in the future.

## FAQ

If you are using new Linux such as Arch, you have to install gcc/g++ version 5.
You also have to set CMAKE\_CXX\_COMPILER, CMAKE\_C\_COMPILER and CUDA\_HOST\_COMPILER accordingly.

# 中文版 README
I made this part since I am a native Mandarin Chinese speaker (zh_TW).

## 概述
本專案實作了 "Unrolled Memory Inner-Products: An Abstract GPU Operator for Efficient Vision-Related Computations" (ICCV 2017) 的 *UMI Operator*。

## 如何編譯
你需要 CMake 以及 Glog。

    mkdir build
    cmake ..
    make

main/ 下面的檔案提供了一些使用 *UMI operator* 的例子，我未來會提供更完整的文件。

## 常見問題
如果你是用新一點的 Linux 像是 Arch，你得想辦法安裝 g++ version 5。
你也要設定一下 CMake 的 CMAKE\_CXX\_COMPILER、CMAKE\_C\_COMPILER、CUDA\_HOST\_COMPILER.
我還沒測過其他平台能不能使用。
