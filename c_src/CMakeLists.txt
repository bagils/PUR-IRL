cmake_minimum_required (VERSION 3.10)

project (cudaconverge LANGUAGES CUDA C CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Java REQUIRED)
include(UseJava)

find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})

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

add_library(cudaconverge SHARED 
  CRC_Prediction_InferenceAlgoCancer.cu
  CRC_Prediction_InferenceAlgoCancer.c
  )
  
target_link_libraries(cudaconverge ${MKL_LIBRARIES})  

enable_testing()

#set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")

#add_jar(HelloWorld HelloWorld.java)

#get_target_property(_jarFile HelloWorld JAR_FILE)
#get_target_property(_classDir HelloWorld CLASSDIR)

#message(STATUS "Jar file ${_jarFile}")
#message(STATUS "Class compiled to ${_classDir}")

#add_test(NAME TestHelloWorld COMMAND ${Java_JAVA_EXECUTABLE} -cp ${_jarFile} HelloWorld)
