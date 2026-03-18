cmake_minimum_required(VERSION 3.18)

project(aes_bruteforce LANGUAGES CXX CUDA)

# ── Standards ──────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD   17)
set(CMAKE_CUDA_STANDARD  17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── CUDA architectures (Volta, Ampere, Ada, Hopper) ───────────────────────────
set(CMAKE_CUDA_ARCHITECTURES "70;80;86;90" CACHE STRING "CUDA architectures")

# ── Dependencies ───────────────────────────────────────────────────────────────
find_package(MPI  REQUIRED)
find_package(OpenMP REQUIRED)

# ── Sources ────────────────────────────────────────────────────────────────────
set(CXX_SOURCES
    src/main.cpp
    src/logger.cpp
    src/aes_cpu.cpp
    src/worker.cpp
    src/master.cpp
)
set(CUDA_SOURCES
    src/aes_cuda.cu
    src/aes_cuda_bs.cu
)

add_executable(aes_bruteforce ${CXX_SOURCES} ${CUDA_SOURCES})

# ── Include dirs ───────────────────────────────────────────────────────────────
target_include_directories(aes_bruteforce PRIVATE
    include/
    BitSlice_AES_CUDA_REF/include/
    ${MPI_CXX_INCLUDE_DIRS}
)

# ── Link libraries ─────────────────────────────────────────────────────────────
target_link_libraries(aes_bruteforce PRIVATE
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    cudart
)

# ── Compile options ────────────────────────────────────────────────────────────
target_compile_options(aes_bruteforce PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -maes          # enable AES-NI
        -msse4.1       # required for _mm_movemask_epi8 etc.
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -lineinfo
    >
)

# ── Definitions ───────────────────────────────────────────────────────────────
target_compile_definitions(aes_bruteforce PRIVATE
    OMPI_SKIP_MPICXX   # avoid C++ MPI bindings (use C API only)
)

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS aes_bruteforce RUNTIME DESTINATION bin)
