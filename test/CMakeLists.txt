#  CppNAM -- C++ Neural Associative Memory Simulator
#  Copyright (C) 2016  Christoph Jenzen, Andreas Stöckel
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

# CMake version and project metadata
cmake_minimum_required (VERSION 3.2)
project(cppnam)

include_directories(PRIVATE ${GTEST_INCLUDE_DIR})

add_executable(cppnam_test_binam
	binam/test_binary_matrix
	binam/test_binam
	binam/test_data_generator
	binam/test_ncr
	binam/test_population_count
)
target_link_libraries(cppnam_test_binam
	${GTEST_LIBRARIES}
	cppnam_binam
)
#set_target_properties(cppnam_test_binam PROPERTIES LINK_SEARCH_START_STATIC 1)
#set_target_properties(cppnam_test_binam PROPERTIES LINK_SEARCH_END_STATIC 1)
add_test(cppnam_test_binam cppnam_test_binam)

#add_executable(cppnam_test_util
#	util/test_population_count
#)

#add_executable(cppnam_test_core
#	core/test_binam
#	core/test_entropy
#	core/test_parameters
#	core/test_spike_trains
#	core/test_spiking_binam
#	core/test_spiking_parameters
#)


#add_dependencies(cppnam_test_core googletest)
#add_dependencies(cppnam_test_util googletest)

#target_link_libraries(cppnam_test_core
#	cppnam_util
#	cppnam_core
#	${GTEST_LIBRARIES}
#)
#target_link_libraries(cppnam_test_util
#	cppnam_util
#	${GTEST_LIBRARIES}
#)

#add_test(cppnam_test_core cppnam_test_core)
#add_test(cppnam_test_util cppnam_test_util)
