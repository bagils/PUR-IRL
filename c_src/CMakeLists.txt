cmake_minimum_required (VERSION 3.10)

project (cudaconverge LANGUAGES CUDA C CXX)

SET(CMAKE_VERBOSE_MAKEFILE ON)


find_package(Java REQUIRED)
include(UseJava)

find_package(JNI REQUIRED)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR})
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIRS})

include_directories(${JNI_INCLUDE_DIRS})
message (STATUS "JNI_LIBRARIES=${JNI_LIBRARIES}")


enable_testing()

include_directories(${CMAKE_SOURCE_DIR}"/cuda-samples/Common/")


add_library(cudaconverge SHARED 
  CRC_Prediction_InferenceAlgoCancer.cu
  CRC_Prediction_InferenceAlgoCancer.c
  )
  
target_link_libraries(cudaconverge ${MKL_LIBRARIES})  

#set(CMAKE_JAVA_COMPILE_FLAGS "-source" "1.6" "-target" "1.6")

#add_jar(HelloWorld HelloWorld.java)

#get_target_property(_jarFile HelloWorld JAR_FILE)
#get_target_property(_classDir HelloWorld CLASSDIR)

#message(STATUS "Jar file ${_jarFile}")
#message(STATUS "Class compiled to ${_classDir}")

#add_test(NAME TestHelloWorld COMMAND ${Java_JAVA_EXECUTABLE} -cp ${_jarFile} HelloWorld)
