cmake_minimum_required(VERSION 3.18)  # 3.18+ supports CMAKE_CUDA_ARCHITECTURES
project(cuda_aes_bitslice LANGUAGES CXX CUDA)

# -------- Language standards --------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------- GPU arch (T4 = sm_75) --------
# Users can override at configure time: -DCMAKE_CUDA_ARCHITECTURES="75"
# To also embed PTX as a forward-compat fallback, use: "75;75-virtual"
set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA SM architectures (e.g. 75, 86, 89)")

# -------- Common include dirs --------
include_directories(${CMAKE_SOURCE_DIR}/include)

# -------- Helper interface target for CUDA flags --------
add_library(cuda_common INTERFACE)
# CUDA-only compile options
target_compile_options(cuda_common INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v,--warn-on-spills>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  # Optional (uncomment if you want kernel line info in profiles):
  # $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

# If you really want a global PTX fallback, uncomment:
# set_property(TARGET cuda_common PROPERTY INTERFACE_COMPILE_OPTIONS
#   "$<JOIN:$<FILTER:$<TARGET_PROPERTY:INTERFACE_COMPILE_OPTIONS>,INCLUDE,.*>,;>"
# )
# and set CMAKE_CUDA_ARCHITECTURES to "75;75-virtual" above.

# ========== Executables ==========
add_executable(cuda-aes-sbox-only
  src/run_sbox_only.cu
  src/aes_sbox_kernel.cu
)
set_target_properties(cuda-aes-sbox-only PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-aes-sbox-only PRIVATE cuda_common)

add_executable(cuda-aes-mix-only
  src/run_mix_only.cu
  src/aes_mix_kernel.cu
)
set_target_properties(cuda-aes-mix-only PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-aes-mix-only PRIVATE cuda_common)

add_executable(cuda-aes-full
  src/run_full_aes.cu
  src/aes_full_kernel.cu
  src/aes_debug_kernels.cu
)
set_target_properties(cuda-aes-full PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-aes-full PRIVATE cuda_common)

# ── Bitsliced brute-force ────────────────────────────────────────────────────
# Tests 32 candidate keys per thread using the bitslice technique.
# Runtime key schedule (bs_keyschedule.cuh) – no compile-time key bake-in.
add_executable(cuda-aes-bruteforce-bs
  src/run_bruteforce_bs.cu
  src/aes_bruteforce_bs_kernel.cu
)
set_target_properties(cuda-aes-bruteforce-bs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cuda-aes-bruteforce-bs PRIVATE cuda_common)

# --- Build-time key override (generates include/generated_rk.hpp) ---
set(AES_KEY_HEX "" CACHE STRING "AES-128 key as 32 hex chars, e.g. 00112233445566778899aabbccddeeff")

if (DEFINED AES_KEY_HEX AND NOT "${AES_KEY_HEX}" STREQUAL "")
  # Normalize to 32 hex chars
  string(TOLOWER "${AES_KEY_HEX}" _k_raw)
  string(REGEX REPLACE "[^0-9a-f]" "" _k "${_k_raw}")
  string(LENGTH "${_k}" _k_len)
  message(STATUS "AES_KEY_HEX (raw)  = '${AES_KEY_HEX}'")
  message(STATUS "AES_KEY_HEX (clean)= '${_k}' (len=${_k_len})")
  if (NOT _k_len EQUAL 32)
    message(FATAL_ERROR "AES_KEY_HEX must be exactly 32 hex chars after cleanup.")
  endif()

  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")

  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/include/generated_rk.hpp"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_SOURCE_DIR}/tools/gen_roundkeys_header.py"
            "${_k}" "${CMAKE_BINARY_DIR}/include/generated_rk.hpp"
    DEPENDS "${CMAKE_SOURCE_DIR}/tools/gen_roundkeys_header.py"
    VERBATIM
  )
  add_custom_target(generate_rk DEPENDS "${CMAKE_BINARY_DIR}/include/generated_rk.hpp")
  add_dependencies(cuda-aes-full generate_rk)

  target_compile_definitions(cuda-aes-full PRIVATE USE_GENERATED_RK=1)
  target_include_directories(cuda-aes-full PRIVATE "${CMAKE_BINARY_DIR}/include")
endif()
