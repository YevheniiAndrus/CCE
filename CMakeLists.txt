cmake_minimum_required(VERSION 2.8)
project( ChromaBasedColorEnhancement )
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( CCE main.cpp )
target_link_libraries( CCE  ${OpenCV_LIBS})