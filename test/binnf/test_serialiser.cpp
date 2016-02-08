/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Christoph Jenzen, Andreas Stöckel
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gtest/gtest.h"

#include <sstream>

#include <binnf/serialiser.hpp>

namespace nam {
namespace binnf {

static Serialiser::Block generate_test_block()
{
	std::string name = "test_matrix";
	Serialiser::Header header = {
	    {"col1", "col2", "col3"},
	    {Serialiser::NumberType::INT, Serialiser::NumberType::FLOAT,
	     Serialiser::NumberType::INT}};
	Matrix<Serialiser::Number> matrix(100, 3);
	for (size_t i = 0; i < matrix.rows(); i++) {
		for (size_t j = 0; j < matrix.cols(); j++) {
			switch (header.types[j]) {
				case Serialiser::NumberType::INT:
					matrix(i, j).i = i * (j + 1);
					break;
				case Serialiser::NumberType::FLOAT:
					matrix(i, j).f = i * (j + 1);
					break;
			}
		}
	}
	return Serialiser::Block(name, header, matrix);
}

TEST(serialiser, read_write)
{
	std::stringstream ss;

	// Generate a test data block
	const Serialiser::Block block_in = generate_test_block();

	// Write the test data to the memory stream
	Serialiser::serialise(ss, block_in);

	// Read it back, expect the deserialisation to be successful
	auto res = Serialiser::deserialise(ss);
	ASSERT_TRUE(res.first);

	// Make sure input and output data are equal
	const Serialiser::Block &block_out = res.second;
	EXPECT_EQ(block_in.name, block_out.name);
	EXPECT_EQ(block_in.matrix, block_out.matrix);
	EXPECT_EQ(block_in.header.size(), block_out.header.size());
	for (size_t i = 0; i < block_in.header.size(); i++) {
		EXPECT_EQ(block_in.header.names[i], block_out.header.names[i]);
		EXPECT_EQ(block_in.header.types[i], block_out.header.types[i]);
	}
}
}
}