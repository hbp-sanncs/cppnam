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

#include <gtest/gtest.h>

#include <cstdint>

#include <cppnam/binam/binary_matrix.hpp>

namespace nam {
TEST(BitReference, get_set)
{
	BinaryMatrixCell c1 = 2482104122392148131L;
	BinaryMatrixCell c2 = 1547305385928427978L;

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_EQ(bool(c1 & (1L << i)), bool(BitReference(c1, i)));
	}
	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		BitReference(c1, i) = c2 & (1L << i);
	}
	EXPECT_EQ(c1, c2);
}

TEST(BitIterator, comparison)
{
	BinaryMatrixCell cs[6] = {2482104122392148131L, 1547305385928427978L,
	                          2342843954723478901L, 4329528589312054200L,
	                          1249014785205430399L, 2232489051200534900L};

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) ==
		            BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) !=
		             BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) <
		             BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) >
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) <=
		            BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) >=
		            BitIterator<false>(&cs[0], i));
	}

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_TRUE(BitIterator<false>(&cs[0], 64 + i) ==
		            BitIterator<false>(&cs[1], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], 64 + i) !=
		             BitIterator<false>(&cs[1], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], 64 + i) <
		             BitIterator<false>(&cs[1], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], 64 + i) >
		             BitIterator<false>(&cs[1], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], 64 + i) <=
		            BitIterator<false>(&cs[1], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], 64 + i) >=
		            BitIterator<false>(&cs[1], i));
	}

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) ==
		             BitIterator<false>(&cs[1], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) !=
		            BitIterator<false>(&cs[1], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) <
		            BitIterator<false>(&cs[1], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) >
		             BitIterator<false>(&cs[1], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) <=
		            BitIterator<false>(&cs[1], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) >=
		             BitIterator<false>(&cs[1], i));
	}

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) ==
		             BitIterator<false>(&cs[0], i + 1));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) !=
		            BitIterator<false>(&cs[0], i + 1));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) <
		            BitIterator<false>(&cs[0], i + 1));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) >
		             BitIterator<false>(&cs[0], i + 1));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i) <=
		            BitIterator<false>(&cs[0], i + 1));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i) >=
		             BitIterator<false>(&cs[0], i + 1));
	}

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_FALSE(BitIterator<false>(&cs[1], i) ==
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[1], i) !=
		            BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[1], i) <
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[1], i) >
		            BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[1], i) <=
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[1], i) >=
		            BitIterator<false>(&cs[0], i));
	}

	for (size_t i = 0; i < sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_FALSE(BitIterator<false>(&cs[0], i + 1) ==
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i + 1) !=
		            BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i + 1) <
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i + 1) >
		            BitIterator<false>(&cs[0], i));
		EXPECT_FALSE(BitIterator<false>(&cs[0], i + 1) <=
		             BitIterator<false>(&cs[0], i));
		EXPECT_TRUE(BitIterator<false>(&cs[0], i + 1) >=
		            BitIterator<false>(&cs[0], i));
	}

	for (size_t i = 0; i < 6 * sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_EQ(bool(cs[i / 64] & (1L << (i % 64))),
		          *(BitIterator<false>(&cs[i / 64], i % 64)));
		EXPECT_EQ(bool(cs[i / 64] & (1L << (i % 64))),
		          *(BitIterator<true>(&cs[i / 64], i % 64)));
	}
}

TEST(RowBitIterator, comparison)
{
	BinaryMatrixCell cs[6] = {2482104122392148131L, 1547305385928427978L,
	                          2342843954723478901L, 4329528589312054200L,
	                          1249014785205430399L, 2232489051200534900L};

	auto it = BitRowIterator(cs, 0);
	auto cit = ConstBitRowIterator(cs, 0);
	for (size_t i = 0; i < 6 * sizeof(BinaryMatrixCell) * 8; i++) {
		EXPECT_EQ(bool(cs[i / 64] & (1L << (i % 64))), *it++);
		EXPECT_EQ(bool(cs[i / 64] & (1L << (i % 64))), *cit++);
		EXPECT_TRUE(it == cit);
	}
}

template <typename Mat>
static void test_get()
{
	Mat mat(BinaryArray<4, 7>{{{0, 0, 0, 1, 1, 0, 0},
	                           {1, 0, 1, 0, 0, 1, 0},
	                           {1, 0, 0, 0, 0, 1, 0},
	                           {1, 0, 0, 0, 0, 1, 1}}});

	EXPECT_EQ(28U, mat.size());
	EXPECT_EQ(4U, mat.rows());
	EXPECT_EQ(7U, mat.cols());

	EXPECT_EQ(0, mat(0, 0));
	EXPECT_EQ(0, mat(0, 1));
	EXPECT_EQ(0, mat(0, 2));
	EXPECT_EQ(1, mat(0, 3));
	EXPECT_EQ(1, mat(0, 4));
	EXPECT_EQ(0, mat(0, 5));
	EXPECT_EQ(0, mat(0, 6));

	EXPECT_EQ(1, mat(1, 0));
	EXPECT_EQ(0, mat(1, 1));
	EXPECT_EQ(1, mat(1, 2));
	EXPECT_EQ(0, mat(1, 3));
	EXPECT_EQ(0, mat(1, 4));
	EXPECT_EQ(1, mat(1, 5));
	EXPECT_EQ(0, mat(1, 6));

	EXPECT_EQ(1, mat(2, 0));
	EXPECT_EQ(0, mat(2, 1));
	EXPECT_EQ(0, mat(2, 2));
	EXPECT_EQ(0, mat(2, 3));
	EXPECT_EQ(0, mat(2, 4));
	EXPECT_EQ(1, mat(2, 5));
	EXPECT_EQ(0, mat(2, 6));

	EXPECT_EQ(1, mat(3, 0));
	EXPECT_EQ(0, mat(3, 1));
	EXPECT_EQ(0, mat(3, 2));
	EXPECT_EQ(0, mat(3, 3));
	EXPECT_EQ(0, mat(3, 4));
	EXPECT_EQ(1, mat(3, 5));
	EXPECT_EQ(1, mat(3, 6));
}

TEST(BinaryMatrix, get) { test_get<BinaryMatrix>(); }

TEST(BinaryMatrix, const_get)
{
	//	test_get<const BinaryMatrix>();
}

template <typename Mat>
static void test_set_get_iterate(size_t rows, size_t cols)
{
	// Reference data generation function
	auto f = [](size_t i, size_t j) {
		return ((i * 429) ^ (j * 176)) % 42 > 21;
	};

	// Create the matrix and make sure it has the correct size
	Mat mat(rows, cols);
	EXPECT_EQ(rows, mat.rows());
	EXPECT_EQ(cols, mat.cols());
	EXPECT_EQ(rows * cols, mat.size());

	// Make sure the matrix is initialized with zeros
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			EXPECT_FALSE(mat(i, j));
		}
	}

	// Write the given function to the matrix (force writing in case a constant
	// matrix type is given).
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			(const_cast<typename std::remove_const<Mat>::type &>(mat))(i, j) =
			    f(i, j);
		}
	}

	// Read the results back
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			EXPECT_EQ(mat(i, j), f(i, j));
		}
	}

	// Iterate over the rows
	for (size_t i = 0; i < mat.rows(); i++) {
		size_t j = 0;
		for (auto it = mat.begin_row(i); it != mat.end_row(i); it++) {
			EXPECT_EQ(f(i, j), *it);
			j++;
		}
		EXPECT_EQ(mat.cols(), j);
	}

	// Iterate over the columns
	for (size_t j = 0; j < mat.cols(); j++) {
		size_t i = 0;
		for (auto it = mat.begin_col(j); it != mat.end_col(j); it++) {
			EXPECT_EQ(f(i, j), *it);
			i++;
		}
		EXPECT_EQ(mat.rows(), i);
	}
}

TEST(BinaryMatrix, set_get_iterate)
{
	test_set_get_iterate<BinaryMatrix>(1022, 879);
	test_set_get_iterate<const BinaryMatrix>(1022, 879);
	test_set_get_iterate<BinaryMatrix>(1024, 2048);
	test_set_get_iterate<const BinaryMatrix>(1024, 2048);
	test_set_get_iterate<BinaryMatrix>(1024, 785);
	test_set_get_iterate<const BinaryMatrix>(1024, 785);
	test_set_get_iterate<BinaryMatrix>(11, 785);
	test_set_get_iterate<const BinaryMatrix>(11, 785);
	test_set_get_iterate<BinaryMatrix>(1, 63);
	test_set_get_iterate<const BinaryMatrix>(1, 63);
}

TEST(BinaryMatrix, print)
{
	auto f = [](size_t i, size_t j) {
		return ((i * 429) ^ (j * 176)) % 42 > 21;
	};

	// Fill the matrix and construct the reference string
	std::stringstream ss_ref;
	BinaryMatrix mat(1024, 546);
	for (size_t i = 0; i < mat.rows(); i++) {
		for (size_t j = 0; j < mat.cols(); j++) {
			mat(i, j) = f(i, j);
			ss_ref << int(f(i, j));
		}
		ss_ref << '\n';
	}

	// Print the matrix to a stringstream and compare the result
	std::stringstream ss_out;
	ss_out << mat;
	EXPECT_EQ(ss_ref.str(), ss_out.str());
}
}
