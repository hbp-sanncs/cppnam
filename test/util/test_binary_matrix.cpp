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

#include <cstdint>
#include <gtest/gtest.h>
#include <util/binary_matrix.hpp>
#include <cypress/util/matrix.hpp>

namespace nam {

TEST(BinaryMatrix, constexpressions)
{
	BinaryMatrix<uint8_t> bin(3, 9);
	EXPECT_EQ(uint32_t(8), bin.intWidth);
	EXPECT_EQ(uint8_t(255), bin.intMax);
	EXPECT_EQ(uint32_t(0), bin.numberOfCells(0));
	EXPECT_EQ(uint32_t(2), bin.numberOfCells(9));
	EXPECT_EQ(uint32_t(2), bin.numberOfCells(15));
	EXPECT_EQ(uint32_t(2), bin.numberOfCells(16));
	EXPECT_EQ(uint32_t(3), bin.numberOfCells(17));
	EXPECT_EQ(uint32_t(0), bin.cellNumber(0));
	EXPECT_EQ(uint32_t(1), bin.cellNumber(8));
	EXPECT_EQ(uint32_t(1), bin.cellNumber(15));
	EXPECT_EQ(uint32_t(2), bin.cellNumber(16));
	EXPECT_EQ(uint32_t(2), bin.cellNumber(17));

#ifndef NDEBUG
	EXPECT_NO_THROW(bin.check_range(2, 8));
	EXPECT_ANY_THROW(bin.check_range(3, 8));
	EXPECT_ANY_THROW(bin.check_range(2, 9));
	EXPECT_NO_THROW(bin.check_range_cells(2, 1));
	EXPECT_ANY_THROW(bin.check_range_cells(3, 1));
	EXPECT_ANY_THROW(bin.check_range_cells(2, 2));
#endif
}

TEST(BinaryMatrix, manipulation)
{
	BinaryMatrix<uint8_t> bin(3, 9);
	bin.set_cell(0, 0, 1);

	EXPECT_EQ(uint8_t(1), bin.get_cell(0, 0));
	EXPECT_EQ(uint8_t(0), bin.get_cell(0, 1));
	EXPECT_EQ(true, bin.get_bit(0, 0));
	EXPECT_EQ(false, bin.get_bit(0, 1));
#ifndef NDEBUG
	EXPECT_ANY_THROW(bin.get_cell(0, 2));
	EXPECT_ANY_THROW(bin.get_bit(0, 9));
#endif
	
	EXPECT_TRUE(bin.row_vec(0).get_bit(0));
	EXPECT_TRUE(bin.row_vec(0).get_cell(0));
	EXPECT_FALSE(bin.row_vec(0).get_bit(1));
	EXPECT_FALSE(bin.row_vec(0).get_cell(1));

	BinaryVector<uint8_t> vec(9);
	vec.set_bit(1);

	EXPECT_EQ(uint8_t(2), vec.get_cell(0));
	EXPECT_EQ(uint8_t(0), vec.get_cell(1));
	EXPECT_EQ(true, vec.get_bit(1));
	EXPECT_EQ(false, vec.get_bit(0));
#ifndef NDEBUG
	EXPECT_ANY_THROW(vec.get_cell(2));
	EXPECT_ANY_THROW(vec.get_bit(9));
#endif

	bin.write_vec(1, vec);

	EXPECT_EQ(uint8_t(2), bin.get_cell(1, 0));
	EXPECT_EQ(uint8_t(0), bin.get_cell(1, 1));
	EXPECT_EQ(true, bin.get_bit(1, 1));
	EXPECT_EQ(false, bin.get_bit(1, 0));

	EXPECT_FALSE(bin.row_vec(1).get_bit(0));
	EXPECT_TRUE(bin.row_vec(1).get_cell(0));
	EXPECT_TRUE(bin.row_vec(1).get_bit(1));
	EXPECT_FALSE(bin.row_vec(1).get_cell(1));

	vec = bin.row_vec(1);
	EXPECT_EQ(uint8_t(2), vec.get_cell(0));
	EXPECT_EQ(uint8_t(0), vec.get_cell(1));
	EXPECT_EQ(true, vec.get_bit(1));
	EXPECT_EQ(false, vec.get_bit(0));

	vec = bin.row_vec(0);
	EXPECT_EQ(uint8_t(1), vec.get_cell(0));
	EXPECT_EQ(uint8_t(0), vec.get_cell(1));
	EXPECT_EQ(true, vec.get_bit(0));
	EXPECT_EQ(false, vec.get_bit(1));

#ifndef NDEBUG
	BinaryVector<uint8_t> vec_big(10);
	BinaryVector<uint8_t> vec_small(7);
	EXPECT_ANY_THROW(bin.write_vec(1, vec_big));
	EXPECT_ANY_THROW(bin.write_vec(1, vec_small));
#endif
	
	BinaryVector<uint8_t> vec2(3), vec3(3);
	vec2.set_bit(1);
	vec3.set_bit(0).set_bit(1);
	EXPECT_TRUE(vec2.VectorMult(vec2).get_bit(1));
	EXPECT_FALSE(vec2.VectorMult(vec2).get_bit(0));
	EXPECT_TRUE(vec2.VectorMult(vec3).get_bit(1));
	EXPECT_FALSE(vec2.VectorMult(vec3).get_bit(0));
	
	
	BinaryMatrix<uint8_t> mat(3,3);
	cypress::Vector<uint8_t> hector({1,0,1});
	EXPECT_EQ(uint8_t(1), hector[0]);
	EXPECT_EQ(uint8_t(0), hector[1]);
	EXPECT_EQ(uint8_t(1), hector[2]);
	mat.write_col_vec(1,hector);
	EXPECT_EQ(uint8_t(0), mat.get_bit(0,0));
	EXPECT_EQ(uint8_t(1), mat.get_bit(0,1));
	EXPECT_EQ(uint8_t(0), mat.get_bit(0,2));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,0));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,1));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,2));
	EXPECT_EQ(uint8_t(0), mat.get_bit(2,0));
	EXPECT_EQ(uint8_t(1), mat.get_bit(2,1));
	EXPECT_EQ(uint8_t(0), mat.get_bit(2,2));
	mat.write_col_vec(2,hector);
	EXPECT_EQ(uint8_t(0), mat.get_bit(0,0));
	EXPECT_EQ(uint8_t(1), mat.get_bit(0,1));
	EXPECT_EQ(uint8_t(1), mat.get_bit(0,2));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,0));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,1));
	EXPECT_EQ(uint8_t(0), mat.get_bit(1,2));
	EXPECT_EQ(uint8_t(0), mat.get_bit(2,0));
	EXPECT_EQ(uint8_t(1), mat.get_bit(2,1));
	EXPECT_EQ(uint8_t(1), mat.get_bit(2,2));
	
}
}
