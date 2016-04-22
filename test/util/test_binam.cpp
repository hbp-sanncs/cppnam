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

#include <cstdint>
#include <limits>
#include <iostream>

#include <util/binam.hpp>
#include <util/binary_matrix.hpp>
#include <util/entropy.hpp>

namespace nam {
TEST(BiNAM, BiNAM)
{
	BiNAM<uint8_t> bin(3, 3);
	BinaryVector<uint8_t> vec_in(3), vec_out(3);
	vec_in.set_bit(1);
	vec_out.set_bit(0).set_bit(2);
	
	EXPECT_EQ(1, bin.digit_sum(vec_in));
	EXPECT_EQ(2, bin.digit_sum(vec_out));
	
	bin.train_vec_check(vec_in,vec_out);
	EXPECT_FALSE(bin.get_bit(0,0));
	EXPECT_TRUE(bin.get_bit(0,1));
	EXPECT_FALSE(bin.get_bit(0,2));
	EXPECT_FALSE(bin.get_bit(1,0));
	EXPECT_FALSE(bin.get_bit(1,2));
	EXPECT_FALSE(bin.get_bit(1,1));
	EXPECT_FALSE(bin.get_bit(2,0));
	EXPECT_TRUE(bin.get_bit(2,1));
	EXPECT_FALSE(bin.get_bit(2,2));
	
	BinaryVector<uint8_t> vec_rec = bin.recall(vec_in,1);
	EXPECT_TRUE(vec_rec.get_bit(0));
	EXPECT_FALSE(vec_rec.get_bit(1));
	EXPECT_TRUE(vec_rec.get_bit(2));
	
	BinaryMatrix<uint8_t> pat_in(2,3),pat_out(2,3);
	pat_in.set_bit(0,0).set_bit(1,1);
	pat_out.set_bit(0,1).set_bit(1,2);
	BiNAM<uint8_t> bin2(3, 3);
	bin2.train_mat(pat_in,pat_out);
	EXPECT_FALSE(bin2.get_bit(0,0));
	EXPECT_FALSE(bin2.get_bit(0,1));
	EXPECT_FALSE(bin2.get_bit(0,2));
	EXPECT_TRUE(bin2.get_bit(1,0));
	EXPECT_FALSE(bin2.get_bit(1,2));
	EXPECT_FALSE(bin2.get_bit(1,1));
	EXPECT_FALSE(bin2.get_bit(2,0));
	EXPECT_TRUE(bin2.get_bit(2,1));
	EXPECT_FALSE(bin2.get_bit(2,2));
	
	BinaryMatrix<uint8_t> res =bin2.recallMat(pat_in,1);
	EXPECT_FALSE(res.get_bit(0,0));
	EXPECT_TRUE(res.get_bit(0,1));
	EXPECT_FALSE(res.get_bit(0,2));
	EXPECT_FALSE(res.get_bit(1,0));
	EXPECT_TRUE(res.get_bit(1,2));
	EXPECT_FALSE(res.get_bit(1,1));
	
	BinaryMatrix<uint8_t> pat_recall(2,3);
	pat_recall.set_bit(1,1).set_bit(1,2);
	std::vector<SampleError> err = bin2.false_bits_mat(pat_out, pat_recall);
	EXPECT_EQ(0, err[0].fp);
	EXPECT_EQ(1, err[0].fn);
	EXPECT_EQ(1, err[1].fp);
	EXPECT_EQ(0, err[1].fn);
	
	BiNAM<uint8_t> bin3(3, 4);
	BinaryMatrix<uint8_t> pat_in2(2,3), pat_out2(2,4);
	EXPECT_NO_THROW(bin3.train_mat(pat_in2, pat_out2));
	EXPECT_ANY_THROW(bin3.train_mat(pat_in2, pat_out));
	EXPECT_ANY_THROW(bin3.train_mat(bin, pat_out2));
}
}