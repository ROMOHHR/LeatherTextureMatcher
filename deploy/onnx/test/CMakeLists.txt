#cmake needs this line
cmake_minimum_required(VERSION 3.1)

#Define project name
project(LEATHER)

#Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

#Find OpenCV, you may need to set OpenCV_DIR variable
#to the absolute path to the directory containing OpenCVConfig.cmake file
#via the command line or GUI
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_INCLUDE_DIRS /root/onnxruntime-linux-x64-1.15.1/include)
set(ONNXRUNTIME_LIBS /root/onnxruntime-linux-x64-1.15.1/lib)

include_directories(
    include
    ${ONNXRUNTIME_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}
)
link_directories(
    ${ONNXRUNTIME_LIBS}
)


file(GLOB SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

#Declare the executable target built from your sources
add_executable(leather_demo ${SRC_LIST})

#Declare the library target built from your sources
# set(LIBRARY_OUTPUT_PATH /usr/lib)
# add_library(leather SHARED ${SRC_LIST})


#Link your application with libraries
target_link_libraries(leather_demo PRIVATE onnxruntime ${OpenCV_LIBS})
# target_link_libraries(leather PRIVATE onnxruntime)

