cmake_minimum_required(VERSION 3.18)
project(madng_tpsa LANGUAGES C)

# This build follows the MAD-NG platform makefiles where appropriate, but keeps
# the flags suitable for Python wheels. In particular, use C11, -O3,
# -ftree-vectorize, and -fno-math-errno as upstream does for C sources. Avoid
# -march=native, -mtune=native, -mfma, and LTO because wheels must not assume the
# build machine's CPU features. OpenMP is also left off for now; MAD-NG has
# serial fallbacks when _OPENMP is not defined.

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux|Darwin")
  message(FATAL_ERROR "madng_tpsa only supports Linux and macOS")
endif()

set(MADNG_SRC "${CMAKE_CURRENT_SOURCE_DIR}/madng/src" CACHE PATH "Path to MAD-NG src directory")
if(NOT EXISTS "${MADNG_SRC}/mad_tpsa.c")
  message(FATAL_ERROR "MADNG_SRC must point to a MAD-NG src directory with mad_tpsa.c")
endif()

set(GTPSA_BUILD_SRC "${CMAKE_CURRENT_BINARY_DIR}/madng_tpsa_src")
file(MAKE_DIRECTORY "${GTPSA_BUILD_SRC}")
file(GLOB MAD_HEADERS CONFIGURE_DEPENDS
  "${MADNG_SRC}/mad_*.h"
  "${MADNG_SRC}/mad_*.hpp"
)
file(GLOB LIBGTPSA_HEADERS CONFIGURE_DEPENDS "${MADNG_SRC}/libgtpsa/*.h")
file(COPY ${MAD_HEADERS} ${LIBGTPSA_HEADERS} DESTINATION "${GTPSA_BUILD_SRC}")
if(EXISTS "${MADNG_SRC}/sse")
  file(COPY "${MADNG_SRC}/sse" DESTINATION "${GTPSA_BUILD_SRC}")
endif()

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  # MAD-NG annotates allocators with GCC's extended malloc(deallocator,idx)
  # attribute. Clang supports the plain malloc attribute, but not this extended
  # spelling, so patch only the copied build header and leave the submodule
  # source untouched.
  file(READ "${GTPSA_BUILD_SRC}/mad_mem.h" MAD_MEM_H)
  string(REPLACE "malloc(mad_free,1)," "" MAD_MEM_H "${MAD_MEM_H}")
  string(REPLACE ",malloc(mad_free,1)" "" MAD_MEM_H "${MAD_MEM_H}")
  file(WRITE "${GTPSA_BUILD_SRC}/mad_mem.h" "${MAD_MEM_H}")
endif()

file(GLOB MADNG_SOURCE_PATHS CONFIGURE_DEPENDS "${MADNG_SRC}/mad_*.c")
list(REMOVE_ITEM MADNG_SOURCE_PATHS
  "${MADNG_SRC}/mad_main.c"
  "${MADNG_SRC}/mad_fft.c"
  "${MADNG_SRC}/mad_nlopt.c"
)

if(EXISTS "${MADNG_SRC}/sse" AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i[3-6]86)$")
  # Streaming SIMD Extensions (SSE) are x86-only and cannot compile on Apple
  # Silicon/arm64. These sources provide optional vectorized mono/bit routines;
  # the generic C implementations in mad_mono.c and mad_bit.c are still built,
  # so omitting SSE on non-x86 platforms preserves functionality.
  file(GLOB MADNG_SSE_SOURCE_PATHS CONFIGURE_DEPENDS "${MADNG_SRC}/sse/*.c")
  list(APPEND MADNG_SOURCE_PATHS ${MADNG_SSE_SOURCE_PATHS})
endif()

set(MADNG_TPSA_SOURCES)
foreach(source_path IN LISTS MADNG_SOURCE_PATHS)
  file(RELATIVE_PATH relative_source_path "${MADNG_SRC}" "${source_path}")
  configure_file(
    "${source_path}"
    "${GTPSA_BUILD_SRC}/${relative_source_path}"
    COPYONLY
  )
  list(APPEND MADNG_TPSA_SOURCES "${GTPSA_BUILD_SRC}/${relative_source_path}")
endforeach()

add_library(madng_tpsa SHARED ${MADNG_TPSA_SOURCES})
target_sources(madng_tpsa PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/native/madng_panic.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/madng_tpsa.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/native/madng_log.c"
)
set_target_properties(madng_tpsa PROPERTIES
  OUTPUT_NAME "madng_tpsa"
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

set_source_files_properties("${GTPSA_BUILD_SRC}/mad_mem.c" PROPERTIES
  COMPILE_OPTIONS "-std=gnu11"
)

target_include_directories(madng_tpsa PRIVATE
  "${GTPSA_BUILD_SRC}"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/madng_tpsa/include"
)
target_compile_options(madng_tpsa PRIVATE
  -include stdlib.h
  -W
  -Wall
  -Wextra
  -O3
  -ftree-vectorize
  -fno-math-errno
  -fexceptions
  -Wno-misleading-indentation
  -Wno-empty-body
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang")
  target_compile_options(madng_tpsa PRIVATE
    -Wno-array-parameter
    -Wno-unknown-attributes
  )
else()
  target_compile_options(madng_tpsa PRIVATE
    -Wno-vla-parameter
  )
endif()

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

if(APPLE)
  target_link_options(madng_tpsa PRIVATE "LINKER:SHELL:-undefined error")
else()
  target_link_options(madng_tpsa PRIVATE "LINKER:--no-undefined")
endif()

target_link_libraries(madng_tpsa PRIVATE LAPACK::LAPACK BLAS::BLAS m)

set(MADNG_PUBLIC_HEADERS
  mad_tpsa.hpp
  mad_num.h
  mad_def.h
  mad_tpsa.h
  mad_ctpsa.h
  mad_mono.h
  mad_desc.h
)
list(TRANSFORM MADNG_PUBLIC_HEADERS PREPEND "${MADNG_SRC}/" OUTPUT_VARIABLE MADNG_PUBLIC_HEADER_PATHS)

set(MADNG_TPSA_SOURCE_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/madng_tpsa/lib")
set(MADNG_TPSA_SOURCE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/madng_tpsa/include")
set(MADNG_TPSA_PUBLIC_HEADERS
  "${MADNG_TPSA_SOURCE_INCLUDE_DIR}/madng_log.h"
  "${MADNG_TPSA_SOURCE_INCLUDE_DIR}/madng_tpsa.h"
)
add_custom_command(TARGET madng_tpsa POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MADNG_TPSA_SOURCE_LIB_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MADNG_TPSA_SOURCE_INCLUDE_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy "$<TARGET_FILE:madng_tpsa>" "${MADNG_TPSA_SOURCE_LIB_DIR}/$<TARGET_FILE_NAME:madng_tpsa>"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${MADNG_PUBLIC_HEADER_PATHS} "${MADNG_TPSA_SOURCE_INCLUDE_DIR}"
  COMMENT "Copying madng_tpsa shared library and MAD-NG headers into src/madng_tpsa"
)

install(TARGETS madng_tpsa
  LIBRARY DESTINATION madng_tpsa/lib
  RUNTIME DESTINATION madng_tpsa/lib
)
install(FILES
  ${MADNG_PUBLIC_HEADER_PATHS}
  ${MADNG_TPSA_PUBLIC_HEADERS}
  DESTINATION madng_tpsa/include
)
