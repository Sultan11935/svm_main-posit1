cmake_minimum_required(VERSION 3.10)
project(cppposit-main)

# Compiler-specific flags
if(SYSPROC MATCHES "arm")
  add_definitions(-D__arm__)
endif()

if(MSVC)
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4 /WX" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX")
  endif()
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wno-long-long -pedantic")
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3 integration
set(Eigen3_DIR "C:/Users/User/Desktop/svm_main-posit1/svm_main-posit1/cppposit-main/eigen/include/eigen/cmake")
#find_package(Eigen3)
# Set the path to the Eigen headers
set(EIGEN_INCLUDE_DIR "C:/Users/User/Desktop/svm_main-posit1/svm_main-posit1/cppposit-main/inlcude/eigen/Eigen")

# Add the Eigen include directory to the include paths
#include_directories(${EIGEN_INCLUDE_DIR})

# Include directories
include_directories(include ${EIGEN3_INCLUDE_DIR})
include_directories(extern)

# Add subdirectories
#add_subdirectory(examples)
enable_testing()
add_subdirectory(tests)
