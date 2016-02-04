# CMake version and project metadata
cmake_minimum_required (VERSION 3.3)
project (cppnam)

# Enable C++14 support
set(CMAKE_CXX_STANDARD 14)

# Enable all warnings on MSVC and GCC/Clang/Intel
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
	endif()
elseif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") OR
		("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR
		("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel"))
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

enable_testing()
find_package(Armadillo REQUIRED)
find_package(GTest REQUIRED)

include_directories(
	src/
	${ARMADILLO_INCLUDE_DIRS}
	${GTEST_INCLUDE_DIRS}
)

add_library(cppnam_util
	src/util/comperator
	src/util/data
	src/util/entropy
	src/util/parameters
	src/util/optimisation
	src/util/ncr
)

target_link_libraries(cppnam_util ${ARMADILLO_LIBRARIES})

add_executable(data_generator_cli
	src/cli/data_generator_cli
)

target_link_libraries(data_generator_cli cppnam_util)

add_executable(cppnam_test_util
	test/util/test_ncr
	test/util/test_comperator
)

target_link_libraries(cppnam_test_util
	cppnam_util
	${GTEST_BOTH_LIBRARIES}
)

add_test(cppnam_test_util cppnam_test_util)