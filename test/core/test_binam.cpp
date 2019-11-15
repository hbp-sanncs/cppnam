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

#include <core/binam.hpp>
#include <core/entropy.hpp>
#include <core/parameters.hpp>
#include <util/binary_matrix.hpp>

namespace nam {
TEST(BiNAM, BiNAM)
{
	BiNAM<uint8_t> bin(3, 3);
	BinaryVector<uint8_t> vec_in(3), vec_out(3);
	vec_in.set_bit(1);
	vec_out.set_bit(0).set_bit(2);

	EXPECT_EQ(1u, bin.digit_sum(vec_in));
	EXPECT_EQ(2u, bin.digit_sum(vec_out));

	bin.train_vec_check(vec_in, vec_out);
	EXPECT_FALSE(bin.get_bit(0, 0));
	EXPECT_TRUE(bin.get_bit(0, 1));
	EXPECT_FALSE(bin.get_bit(0, 2));
	EXPECT_FALSE(bin.get_bit(1, 0));
	EXPECT_FALSE(bin.get_bit(1, 2));
	EXPECT_FALSE(bin.get_bit(1, 1));
	EXPECT_FALSE(bin.get_bit(2, 0));
	EXPECT_TRUE(bin.get_bit(2, 1));
	EXPECT_FALSE(bin.get_bit(2, 2));

	BinaryVector<uint8_t> vec_rec = bin.recall(vec_in);
	EXPECT_TRUE(vec_rec.get_bit(0));
	EXPECT_FALSE(vec_rec.get_bit(1));
	EXPECT_TRUE(vec_rec.get_bit(2));

	BinaryMatrix<uint8_t> pat_in(2, 3), pat_out(2, 3);
	pat_in.set_bit(0, 0).set_bit(1, 1);
	pat_out.set_bit(0, 1).set_bit(1, 2);
	BiNAM<uint8_t> bin2(3, 3);
	bin2.train_mat(pat_in, pat_out);
	EXPECT_FALSE(bin2.get_bit(0, 0));
	EXPECT_FALSE(bin2.get_bit(0, 1));
	EXPECT_FALSE(bin2.get_bit(0, 2));
	EXPECT_TRUE(bin2.get_bit(1, 0));
	EXPECT_FALSE(bin2.get_bit(1, 2));
	EXPECT_FALSE(bin2.get_bit(1, 1));
	EXPECT_FALSE(bin2.get_bit(2, 0));
	EXPECT_TRUE(bin2.get_bit(2, 1));
	EXPECT_FALSE(bin2.get_bit(2, 2));

	BinaryMatrix<uint8_t> res = bin2.recallMat(pat_in);
	EXPECT_FALSE(res.get_bit(0, 0));
	EXPECT_TRUE(res.get_bit(0, 1));
	EXPECT_FALSE(res.get_bit(0, 2));
	EXPECT_FALSE(res.get_bit(1, 0));
	EXPECT_TRUE(res.get_bit(1, 2));
	EXPECT_FALSE(res.get_bit(1, 1));

	BinaryMatrix<uint8_t> pat_recall(2, 3);
	pat_recall.set_bit(1, 1).set_bit(1, 2);
	std::vector<SampleError> err = bin2.false_bits_mat(pat_out, pat_recall);
	EXPECT_EQ(0u, err[0].fp);
	EXPECT_EQ(1u, err[0].fn);
	EXPECT_EQ(1u, err[1].fp);
	EXPECT_EQ(0u, err[1].fn);

#ifndef NDEBUG
	BiNAM<uint8_t> bin3(3, 4);
	BinaryMatrix<uint8_t> pat_in2(5, 4), pat_out2(5, 3);
	EXPECT_NO_THROW(bin3.train_mat(pat_in2, pat_out2));
	EXPECT_ANY_THROW(bin3.train_mat(pat_in2, pat_out));
	EXPECT_ANY_THROW(bin3.train_mat(bin, pat_out2));
#endif

	BinaryMatrix<uint8_t> false_recall(2, 3);
	false_recall.set_bit(0, 1).set_bit(0, 2);
	auto se = bin2.false_bits_mat(pat_out, false_recall);
	EXPECT_EQ(1, se[0].fp);
	EXPECT_EQ(0, se[0].fn);
	EXPECT_EQ(0, se[1].fp);
	EXPECT_EQ(1, se[1].fn);
	BinaryMatrix<uint8_t> out_test(20, 20), out_false(20, 20);
	out_test.set_bit(0, 0)
	    .set_bit(0, 10)
	    .set_bit(1, 9)
	    .set_bit(1, 15)
	    .set_bit(1, 16)
	    .set_bit(1, 18);
	out_false.set_bit(0, 0)
	    .set_bit(0, 7)
	    .set_bit(0, 8)
	    .set_bit(0, 10)
	    .set_bit(0, 13)
	    .set_bit(1, 9)
	    .set_bit(1, 18);
	auto se2 = bin2.false_bits_mat(out_test, out_false);
	EXPECT_EQ(3u, se2[0].fp);
	EXPECT_EQ(0u, se2[0].fn);
	EXPECT_EQ(0u, se2[1].fp);
	EXPECT_EQ(2u, se2[1].fn);
	EXPECT_EQ(3u, bin2.false_bits(out_test.row_vec(0), out_false.row_vec(0)).fp);
}
TEST(BiNAM, BiNAM2)
{
	BiNAM<uint8_t> binam(10, 10);
	binam.set_bit(0, 0)
	    .set_bit(0, 2)
	    .set_bit(0, 3)
	    .set_bit(1, 0)
	    .set_bit(1, 2)
	    .set_bit(1, 3)
	    .set_bit(2, 0)
	    .set_bit(2, 2)
	    .set_bit(2, 3)
	    .set_bit(3, 0)
	    .set_bit(3, 2)
	    .set_bit(3, 3)
	    .set_bit(3, 4)
	    .set_bit(3, 5)
	    .set_bit(3, 6)
	    .set_bit(4, 0)
	    .set_bit(4, 2)
	    .set_bit(4, 4)
	    .set_bit(5, 3)
	    .set_bit(5, 5)
	    .set_bit(5, 6)
	    .set_bit(6, 0)
	    .set_bit(6, 2)
	    .set_bit(6, 4)
	    .set_bit(7, 3)
	    .set_bit(7, 5)
	    .set_bit(7, 6);

	BinaryMatrix<uint8_t> input(6, 10);
	input.set_bit(0, 0)
	    .set_bit(0, 2)
	    .set_bit(0, 3)
	    .set_bit(1, 0)
	    .set_bit(1, 2)
	    .set_bit(1, 4)
	    .set_bit(2, 3)
	    .set_bit(2, 5)
	    .set_bit(2, 6)
	    .set_bit(3, 0)
	    .set_bit(3, 2)
	    .set_bit(3, 6)
	    .set_bit(4, 3)
	    .set_bit(4, 5)
	    .set_bit(4, 4)
	    .set_bit(5, 5)
	    .set_bit(5, 4)
	    .set_bit(5, 6);

	BinaryMatrix<uint8_t> compare(6, 10);
	compare.set_bit(0, 0)
	    .set_bit(0, 1)
	    .set_bit(0, 2)
	    .set_bit(1, 3)
	    .set_bit(1, 4)
	    .set_bit(1, 6)
	    .set_bit(2, 3)
	    .set_bit(2, 5)
	    .set_bit(2, 7)
	    .set_bit(3, 0)
	    .set_bit(3, 1)
	    .set_bit(3, 3)
	    .set_bit(4, 0)
	    .set_bit(4, 1)
	    .set_bit(4, 3)
	    .set_bit(5, 0)
	    .set_bit(5, 1)
	    .set_bit(5, 2);

	auto output = binam.recallMat(input);

	EXPECT_EQ(1, output.get_bit(0, 0));
	EXPECT_EQ(1, output.get_bit(0, 1));
	EXPECT_EQ(1, output.get_bit(0, 2));
	EXPECT_EQ(1, output.get_bit(0, 3));
	EXPECT_EQ(0, output.get_bit(0, 4));
	EXPECT_EQ(0, output.get_bit(0, 5));
	EXPECT_EQ(0, output.get_bit(0, 6));
	EXPECT_EQ(0, output.get_bit(0, 7));
	EXPECT_EQ(0, output.get_bit(0, 8));
	EXPECT_EQ(0, output.get_bit(0, 9));

	EXPECT_EQ(0, output.get_bit(1, 0));
	EXPECT_EQ(0, output.get_bit(1, 1));
	EXPECT_EQ(0, output.get_bit(1, 2));
	EXPECT_EQ(1, output.get_bit(1, 3));
	EXPECT_EQ(1, output.get_bit(1, 4));
	EXPECT_EQ(0, output.get_bit(1, 5));
	EXPECT_EQ(1, output.get_bit(1, 6));
	EXPECT_EQ(0, output.get_bit(1, 7));
	EXPECT_EQ(0, output.get_bit(1, 8));
	EXPECT_EQ(0, output.get_bit(1, 9));

	EXPECT_EQ(0, output.get_bit(2, 0));
	EXPECT_EQ(0, output.get_bit(2, 1));
	EXPECT_EQ(0, output.get_bit(2, 2));
	EXPECT_EQ(1, output.get_bit(2, 3));
	EXPECT_EQ(0, output.get_bit(2, 4));
	EXPECT_EQ(1, output.get_bit(2, 5));
	EXPECT_EQ(0, output.get_bit(2, 6));
	EXPECT_EQ(1, output.get_bit(2, 7));
	EXPECT_EQ(0, output.get_bit(2, 8));
	EXPECT_EQ(0, output.get_bit(2, 9));

	EXPECT_EQ(0, output.get_bit(3, 0));
	EXPECT_EQ(0, output.get_bit(3, 1));
	EXPECT_EQ(0, output.get_bit(3, 2));
	EXPECT_EQ(1, output.get_bit(3, 3));
	EXPECT_EQ(0, output.get_bit(3, 4));
	EXPECT_EQ(0, output.get_bit(3, 5));
	EXPECT_EQ(0, output.get_bit(3, 6));
	EXPECT_EQ(0, output.get_bit(3, 7));
	EXPECT_EQ(0, output.get_bit(3, 8));
	EXPECT_EQ(0, output.get_bit(3, 9));

	EXPECT_EQ(0, output.get_bit(4, 0));
	EXPECT_EQ(0, output.get_bit(4, 1));
	EXPECT_EQ(0, output.get_bit(4, 2));
	EXPECT_EQ(1, output.get_bit(4, 3));
	EXPECT_EQ(0, output.get_bit(4, 4));
	EXPECT_EQ(0, output.get_bit(4, 5));
	EXPECT_EQ(0, output.get_bit(4, 6));
	EXPECT_EQ(0, output.get_bit(4, 7));
	EXPECT_EQ(0, output.get_bit(4, 8));
	EXPECT_EQ(0, output.get_bit(4, 9));

	EXPECT_EQ(0, output.get_bit(5, 0));
	EXPECT_EQ(0, output.get_bit(5, 1));
	EXPECT_EQ(0, output.get_bit(5, 2));
	EXPECT_EQ(1, output.get_bit(5, 3));
	EXPECT_EQ(0, output.get_bit(5, 4));
	EXPECT_EQ(0, output.get_bit(5, 5));
	EXPECT_EQ(0, output.get_bit(5, 6));
	EXPECT_EQ(0, output.get_bit(5, 7));
	EXPECT_EQ(0, output.get_bit(5, 8));
	EXPECT_EQ(0, output.get_bit(5, 9));
	DataParameters params(10, 10, 3, 3, 10);
	BiNAM_Container<uint8_t> cont(params);
	cont.trained_matrix(binam);
	cont.input_matrix(input);
	cont.output_matrix(compare);
	cont.recall();
	std::vector<SampleError> false_bits = cont.false_bits();
	SampleError sum_false_bits = cont.sum_false_bits(false_bits);
	EXPECT_EQ(1, false_bits[0].fp);
	EXPECT_EQ(0, false_bits[0].fn);
	EXPECT_EQ(0, false_bits[1].fp);
	EXPECT_EQ(0, false_bits[1].fn);
	EXPECT_EQ(0, false_bits[2].fp);
	EXPECT_EQ(0, false_bits[2].fn);
	EXPECT_EQ(0, false_bits[3].fp);
	EXPECT_EQ(2, false_bits[3].fn);
	EXPECT_EQ(0, false_bits[4].fp);
	EXPECT_EQ(2, false_bits[4].fn);
	EXPECT_EQ(1, false_bits[5].fp);
	EXPECT_EQ(3, false_bits[5].fn);

	EXPECT_EQ(2, sum_false_bits.fp);
	EXPECT_EQ(7, sum_false_bits.fn);
}
TEST(BiNAM, BiNAM3)
{
	BiNAM<uint8_t> binam(10, 10);
	binam.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 6)
	    .set_bit(2, 9)
	    .set_bit(2, 7)
	    .set_bit(2, 6)
	    .set_bit(3, 9)
	    .set_bit(3, 7)
	    .set_bit(3, 6)
	    .set_bit(3, 5)
	    .set_bit(3, 4)
	    .set_bit(3, 3)
	    .set_bit(4, 9)
	    .set_bit(4, 7)
	    .set_bit(4, 5)
	    .set_bit(5, 6)
	    .set_bit(5, 4)
	    .set_bit(5, 3)
	    .set_bit(6, 9)
	    .set_bit(6, 7)
	    .set_bit(6, 5)
	    .set_bit(7, 6)
	    .set_bit(7, 4)
	    .set_bit(7, 3);

	BinaryMatrix<uint8_t> input(6, 10);
	input.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 5)
	    .set_bit(2, 6)
	    .set_bit(2, 4)
	    .set_bit(2, 3)
	    .set_bit(3, 9)
	    .set_bit(3, 7)
	    .set_bit(3, 3)
	    .set_bit(4, 6)
	    .set_bit(4, 4)
	    .set_bit(4, 5)
	    .set_bit(5, 4)
	    .set_bit(5, 5)
	    .set_bit(5, 3);

	BinaryMatrix<uint8_t> compare(6, 10);
	compare.set_bit(0, 0)
	    .set_bit(0, 1)
	    .set_bit(0, 2)
	    .set_bit(1, 3)
	    .set_bit(1, 4)
	    .set_bit(1, 6)
	    .set_bit(2, 3)
	    .set_bit(2, 5)
	    .set_bit(2, 7)
	    .set_bit(3, 0)
	    .set_bit(3, 1)
	    .set_bit(3, 3)
	    .set_bit(4, 0)
	    .set_bit(4, 1)
	    .set_bit(4, 3)
	    .set_bit(5, 0)
	    .set_bit(5, 1)
	    .set_bit(5, 2);

	auto output = binam.recallMat(input);

	EXPECT_EQ(1, output.get_bit(0, 0));
	EXPECT_EQ(1, output.get_bit(0, 1));
	EXPECT_EQ(1, output.get_bit(0, 2));
	EXPECT_EQ(1, output.get_bit(0, 3));
	EXPECT_EQ(0, output.get_bit(0, 4));
	EXPECT_EQ(0, output.get_bit(0, 5));
	EXPECT_EQ(0, output.get_bit(0, 6));
	EXPECT_EQ(0, output.get_bit(0, 7));
	EXPECT_EQ(0, output.get_bit(0, 8));
	EXPECT_EQ(0, output.get_bit(0, 9));

	EXPECT_EQ(0, output.get_bit(1, 0));
	EXPECT_EQ(0, output.get_bit(1, 1));
	EXPECT_EQ(0, output.get_bit(1, 2));
	EXPECT_EQ(1, output.get_bit(1, 3));
	EXPECT_EQ(1, output.get_bit(1, 4));
	EXPECT_EQ(0, output.get_bit(1, 5));
	EXPECT_EQ(1, output.get_bit(1, 6));
	EXPECT_EQ(0, output.get_bit(1, 7));
	EXPECT_EQ(0, output.get_bit(1, 8));
	EXPECT_EQ(0, output.get_bit(1, 9));

	EXPECT_EQ(0, output.get_bit(2, 0));
	EXPECT_EQ(0, output.get_bit(2, 1));
	EXPECT_EQ(0, output.get_bit(2, 2));
	EXPECT_EQ(1, output.get_bit(2, 3));
	EXPECT_EQ(0, output.get_bit(2, 4));
	EXPECT_EQ(1, output.get_bit(2, 5));
	EXPECT_EQ(0, output.get_bit(2, 6));
	EXPECT_EQ(1, output.get_bit(2, 7));
	EXPECT_EQ(0, output.get_bit(2, 8));
	EXPECT_EQ(0, output.get_bit(2, 9));

	EXPECT_EQ(0, output.get_bit(3, 0));
	EXPECT_EQ(0, output.get_bit(3, 1));
	EXPECT_EQ(0, output.get_bit(3, 2));
	EXPECT_EQ(1, output.get_bit(3, 3));
	EXPECT_EQ(0, output.get_bit(3, 4));
	EXPECT_EQ(0, output.get_bit(3, 5));
	EXPECT_EQ(0, output.get_bit(3, 6));
	EXPECT_EQ(0, output.get_bit(3, 7));
	EXPECT_EQ(0, output.get_bit(3, 8));
	EXPECT_EQ(0, output.get_bit(3, 9));

	EXPECT_EQ(0, output.get_bit(4, 0));
	EXPECT_EQ(0, output.get_bit(4, 1));
	EXPECT_EQ(0, output.get_bit(4, 2));
	EXPECT_EQ(1, output.get_bit(4, 3));
	EXPECT_EQ(0, output.get_bit(4, 4));
	EXPECT_EQ(0, output.get_bit(4, 5));
	EXPECT_EQ(0, output.get_bit(4, 6));
	EXPECT_EQ(0, output.get_bit(4, 7));
	EXPECT_EQ(0, output.get_bit(4, 8));
	EXPECT_EQ(0, output.get_bit(4, 9));

	EXPECT_EQ(0, output.get_bit(5, 0));
	EXPECT_EQ(0, output.get_bit(5, 1));
	EXPECT_EQ(0, output.get_bit(5, 2));
	EXPECT_EQ(1, output.get_bit(5, 3));
	EXPECT_EQ(0, output.get_bit(5, 4));
	EXPECT_EQ(0, output.get_bit(5, 5));
	EXPECT_EQ(0, output.get_bit(5, 6));
	EXPECT_EQ(0, output.get_bit(5, 7));
	EXPECT_EQ(0, output.get_bit(5, 8));
	EXPECT_EQ(0, output.get_bit(5, 9));

	DataParameters params(10, 10, 3, 3, 10);
	BiNAM_Container<uint8_t> cont(params);
	cont.trained_matrix(binam);
	cont.input_matrix(input);
	cont.output_matrix(compare);
	cont.recall();
	std::vector<SampleError> false_bits = cont.false_bits();
	SampleError sum_false_bits = cont.sum_false_bits(false_bits);
	EXPECT_EQ(1, false_bits[0].fp);
	EXPECT_EQ(0, false_bits[0].fn);
	EXPECT_EQ(0, false_bits[1].fp);
	EXPECT_EQ(0, false_bits[1].fn);
	EXPECT_EQ(0, false_bits[2].fp);
	EXPECT_EQ(0, false_bits[2].fn);
	EXPECT_EQ(0, false_bits[3].fp);
	EXPECT_EQ(2, false_bits[3].fn);
	EXPECT_EQ(0, false_bits[4].fp);
	EXPECT_EQ(2, false_bits[4].fn);
	EXPECT_EQ(1, false_bits[5].fp);
	EXPECT_EQ(3, false_bits[5].fn);

	EXPECT_EQ(2, sum_false_bits.fp);
	EXPECT_EQ(7, sum_false_bits.fn);
}
TEST(BiNAM, BiNAM4)
{
	BiNAM<uint8_t> binam(10, 10);
	binam.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 6)
	    .set_bit(2, 9)
	    .set_bit(2, 7)
	    .set_bit(2, 6)
	    .set_bit(3, 9)
	    .set_bit(3, 7)
	    .set_bit(3, 6)
	    .set_bit(3, 5)
	    .set_bit(3, 4)
	    .set_bit(3, 3)
	    .set_bit(4, 9)
	    .set_bit(4, 7)
	    .set_bit(4, 5)
	    .set_bit(5, 6)
	    .set_bit(5, 4)
	    .set_bit(5, 3)
	    .set_bit(6, 9)
	    .set_bit(6, 7)
	    .set_bit(6, 5)
	    .set_bit(7, 6)
	    .set_bit(7, 4)
	    .set_bit(7, 3);

	BinaryMatrix<uint8_t> input(6, 10);
	input.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 5)
	    .set_bit(2, 6)
	    .set_bit(2, 4)
	    .set_bit(2, 3);

	BinaryMatrix<uint8_t> compare(6, 10);
	compare.set_bit(0, 0)
	    .set_bit(0, 1)
	    .set_bit(0, 2)
	    .set_bit(1, 3)
	    .set_bit(1, 4)
	    .set_bit(1, 6)
	    .set_bit(2, 3)
	    .set_bit(2, 5)
	    .set_bit(2, 7);
	BiNAM<uint8_t> binam2(10, 10);
	binam2.train_mat(input, compare);
	for (size_t i = 0; i < binam.rows(); i++) {
		for (size_t j = 0; j < binam.cols(); j++) {
			EXPECT_EQ(binam.get_bit(i, j), binam2.get_bit(i, j));
		}
	}
}
TEST(BiNAM, binam5)
{
	BiNAM<uint8_t> binam(10, 10);
	binam.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 6)
	    .set_bit(2, 9)
	    .set_bit(2, 7)
	    .set_bit(2, 6)
	    .set_bit(3, 9)
	    .set_bit(3, 7)
	    .set_bit(3, 6)
	    .set_bit(3, 5)
	    .set_bit(3, 4)
	    .set_bit(3, 3)
	    .set_bit(4, 9)
	    .set_bit(4, 7)
	    .set_bit(4, 5)
	    .set_bit(5, 6)
	    .set_bit(5, 4)
	    .set_bit(5, 3)
	    .set_bit(6, 9)
	    .set_bit(6, 7)
	    .set_bit(6, 5)
	    .set_bit(7, 6)
	    .set_bit(7, 4)
	    .set_bit(7, 3);

	BinaryMatrix<uint8_t> input(6, 10);
	input.set_bit(0, 9)
	    .set_bit(0, 7)
	    .set_bit(0, 6)
	    .set_bit(1, 9)
	    .set_bit(1, 7)
	    .set_bit(1, 5)
	    .set_bit(2, 6)
	    .set_bit(2, 4)
	    .set_bit(2, 3);
	BinaryMatrix<uint8_t> compare(6, 10);
	compare.set_bit(0, 0)
	    .set_bit(0, 1)
	    .set_bit(0, 2)
	    .set_bit(1, 3)
	    .set_bit(1, 4)
	    .set_bit(1, 6)
	    .set_bit(2, 3)
	    .set_bit(2, 5)
	    .set_bit(2, 7);
	BiNAM<uint8_t> binam2(10, 10);
	binam2.train_mat(input, compare);
	for (size_t i = 0; i < binam.rows(); i++) {
		for (size_t j = 0; j < binam.cols(); j++) {
			EXPECT_EQ(binam.get_bit(i, j), binam2.get_bit(i, j));
		}
	}
}
}
