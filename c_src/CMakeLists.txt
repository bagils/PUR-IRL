###################################################################
# Changelog
# 
# * Conditionally compile mkl or cuda lib (Mario Ceresa 18/01/2022)
# * Basic cmake support (Mario Ceresa 17/01/2022)
#
###################################################################

cmake_minimum_required (VERSION 3.10)

project (converge LANGUAGES CUDA C CXX )

include_directories(${CMAKE_SOURCE_DIR})
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})

option(BUILD_CUDA "Build cuda accelerated kernel library" ON)
option(BUILD_MKL "Build mkl accelerated kernel library" ON)

# Java detection 
find_package(Java REQUIRED)
include(UseJava)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

if (BUILD_MKL)  
  find_package(MKL REQUIRED)
  include_directories(${MKL_INCLUDE_DIRS})
  add_library(mklconverge SHARED 
    CRC_Prediction_InferenceAlgoCancer.c
  )  
  target_link_libraries(mklconverge ${MKL_LIBRARIES})  
endif()

# CUDA detection 
if (BUILD_CUDA)  
  include(FindCUDA)
  cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_TMP Auto)
  set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS_TMP} CACHE STRING "CUDA -gencode parameters")
  string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

  find_library(CUDA_DRIVER_LIBRARY
               NAMES cuda_driver cuda
               HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                     ENV CUDA_PATH
               PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
  if (NOT CUDA_DRIVER_LIBRARY)
      # Don't try any stub directories until we have exhausted all other search locations.
      find_library(CUDA_DRIVER_LIBRARY
                   NAMES cuda_driver cuda
                   HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                         ENV CUDA_PATH
                   PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
  endif ()
  
#  find_library(CUBLAS_LIBRARY 
#               NAMES cublas ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
#               )
   find_library(CUBLAS_LIBRARY 
               NAMES cublas 
               HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                     ENV CUDA_PATH
               PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)               
               
  mark_as_advanced(CUDA_DRIVER_LIBRARY)


  message("Checking for cuda samples include file...")

  # Check if cuda samples are in the same directory of the project
  #  or in its parent
  set(CUDA_SAMPLES_COMM "cuda-samples/Common/")
  set(CUDA_SAMPLES_BASE_PATH "${CMAKE_SOURCE_DIR}/../${CUDA_SAMPLES_COMM}"
    CACHE STRING "Path to cuda sample include files")  
  message("CUDA samples base path is set to:" ${CUDA_SAMPLES_BASE_PATH})


  find_file (CUDA_HELPER_INC
    NAMES helper_cuda.h 
    PATHS ${CUDA_SAMPLES_BASE_PATH}
    DOC "Path to cuda sample include files"
  )

  if ( ${CUDA_HELPER_INC} MATCHES "NOTFOUND$")
    message(FATAL_ERROR "Could not find cuda samples includes (i.e. helper_cuda.h) in ${CUDA_SAMPLES_BASE_PATH}, please set it with -DCUDA_SAMPLES_PATH='path/to/Common'")
  endif()

  include_directories(${CUDA_SAMPLES_BASE_PATH})

  # Create compute library depending on the presence of mkl or cuda
  add_library(cudaconverge SHARED 
    CRC_Prediction_InferenceAlgoCancer.cu
  )
  target_include_directories(cudaconverge PRIVATE ${CUDA_INCLUDE_DIRS})  
  target_link_libraries(cudaconverge PRIVATE 
            ${CUDA_LIBRARIES} ${CUDA_DRIVER_LIBRARY} ${CUDA_CUBLAS_LIBRARIES}
            )  
  
  enable_testing()

  add_executable(cudaconverge_test 
      CodeTester.cu
    )

  target_link_libraries(cudaconverge_test 
       cudaconverge ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}
   )
  
endif()
  


