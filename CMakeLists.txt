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
find_package(GTest)

include_directories(src/)

add_library(cppnam_util
	src/util/binam
	src/util/binary_matrix
	src/util/data
	src/util/entropy
	src/util/ncr
	src/util/optimisation
	src/util/parameters
	src/util/population_count
	src/util/spike_trains
	src/util/spiking_binam
	src/util/spiking_parameters
)

target_link_libraries(cppnam_util
	/usr/local/lib/libcypress.a
#	/home/andreas/.local/lib/libcypress.a
	-pthread
)

add_executable(data_generator
	src/cli/data_generator
)

target_link_libraries(data_generator
	cppnam_util
)

add_executable(binam
	src/cli/binam
)

target_link_libraries(binam
	cppnam_util
)

add_executable(sp_binam
	src/cli/sp_binam
)

target_link_libraries(sp_binam
	cppnam_util
)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test/util/test.json 
	DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#
# Unit tests -- only enable if gtest is present
#
if(${GTEST_FOUND})
	include_directories(${GTEST_INCLUDE_DIR})

	add_executable(cppnam_test_util
		test/util/test_binam
		test/util/test_binary_matrix
		test/util/test_entropy
		test/util/test_ncr
		test/util/test_parameters
		test/util/test_population_count
		test/util/test_spike_trains
		test/util/test_spiking_binam
		test/util/test_spiking_parameters
	)

	target_link_libraries(cppnam_test_util
		cppnam_util
		${GTEST_BOTH_LIBRARIES}
	)

	add_test(cppnam_test_util cppnam_test_util)
endif()
