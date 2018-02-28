cmake_minimum_required(VERSION 2.8)
project( tutorial1 )
find_package( OpenCV REQUIRED )
add_executable( gazeTracker gazeTracker.cpp )
target_link_libraries( gazeTracker ${OpenCV_LIBS} )

