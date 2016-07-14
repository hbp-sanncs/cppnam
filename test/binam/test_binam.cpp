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

#include <cppnam/binam/binam.hpp>

namespace nam {

TEST(BiNAM, simple)
{
	const BiNAM bin_expected =
	    BiNAM(BinaryArray<3, 3>{{{0, 1, 0}, {0, 0, 0}, {0, 1, 0}}});
	const BinaryVector vec_in = {0, 1, 0};
	const BinaryVector vec_out = {1, 0, 1};
	const BinaryVector vec_expected = {1, 0, 1};

	BiNAM bin(3, 3);
	bin.train(vec_in, vec_out);
	const BinaryMatrix vec_rec = bin.recall_auto_threshold(vec_in);

	EXPECT_EQ(BinaryMatrix(bin_expected), BinaryMatrix(bin));
	EXPECT_EQ(vec_expected, vec_rec);
}

TEST(BiNAM, palm)
{
	const BiNAM bin_expected =
	    BiNAM(BinaryArray<10, 10>{{{1, 0, 1, 1, 0, 0, 0, 0, 0, 0},
	                               {1, 0, 1, 1, 0, 0, 0, 0, 0, 0},
	                               {1, 0, 1, 1, 0, 0, 0, 0, 0, 0},
	                               {1, 0, 1, 1, 1, 1, 1, 0, 0, 0},
	                               {1, 0, 1, 0, 1, 0, 0, 0, 0, 0},
	                               {0, 0, 0, 1, 0, 1, 1, 0, 0, 0},
	                               {1, 0, 1, 0, 1, 0, 0, 0, 0, 0},
	                               {0, 0, 0, 1, 0, 1, 1, 0, 0, 0},
	                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
	                               {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}});

	const BinaryVector vec1_in = {0, 0, 0, 1, 0, 1, 1, 0, 0, 0};
	const BinaryVector vec2_in = {1, 0, 1, 0, 1, 0, 0, 0, 0, 0};
	const BinaryVector vec3_in = {1, 0, 1, 1, 0, 0, 0, 0, 0, 0};

	const BinaryVector vec1_out = {0, 0, 0, 1, 0, 1, 0, 1, 0, 0};
	const BinaryVector vec2_out = {0, 0, 0, 1, 1, 0, 1, 0, 0, 0};
	const BinaryVector vec3_out = {1, 1, 1, 0, 0, 0, 0, 0, 0, 0};

	const BinaryVector vec1_expected = {0, 0, 0, 1, 0, 1, 0, 1, 0, 0};
	const BinaryVector vec2_expected = {0, 0, 0, 1, 1, 0, 1, 0, 0, 0};
	const BinaryVector vec3_expected = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0};

	// Train the BiNAM
	BiNAM bin(10, 10);
	bin.train(vec1_in, vec1_out);
	bin.train(vec2_in, vec2_out);
	bin.train(vec3_in, vec3_out);
	EXPECT_EQ(bin_expected, bin);

	// Recall the vectors and compare them to the expected result
	BinaryMatrix vec1_recall = bin.recall_auto_threshold(vec1_in);
	BinaryMatrix vec2_recall = bin.recall_auto_threshold(vec2_in);
	BinaryMatrix vec3_recall = bin.recall_auto_threshold(vec3_in);

	EXPECT_EQ(vec1_expected, vec1_recall);
	EXPECT_EQ(vec2_expected, vec2_recall);
	EXPECT_EQ(vec3_expected, vec3_recall);
}

TEST(BiNAM, large)
{
	auto fill = [](BinaryMatrix &mat, auto f) {
		for (size_t i = 0; i < mat.rows(); i++) {
			for (size_t j = 0; j < mat.cols(); j++) {
				mat(i, j) = f(i, j);
			}
		}
	};
	auto f_in = [](size_t i, size_t j) { return (i * 583 ^ j * 347) % 32 < 3; };
	auto f_out = [](size_t i, size_t j) {
		return (i * 213 ^ j * 123) % 32 < 3;
	};

	// Build the input and output matrix
	size_t n_samples = 1000;
	size_t d_in = 548;
	size_t d_out = 381;
	size_t th = 4;
	BinaryMatrix mat_in(n_samples, d_in);
	BinaryMatrix mat_out(n_samples, d_out);
	fill(mat_in, f_in);
	fill(mat_out, f_out);

	// Calculate the expected BiNAM matrix after training
	BinaryMatrix bin_expected(d_out, d_in);
	for (size_t i = 0; i < d_out; i++) {
		for (size_t j = 0; j < d_in; j++) {
			for (size_t n = 0; n < n_samples; n++) {
				bin_expected(i, j) =
				    bin_expected(i, j) | (f_in(n, j) & f_out(n, i));
			}
		}
	}

	// Calculate the expected recalled vectors
	BinaryMatrix mat_recall_expected(n_samples, d_out);
	BinaryMatrix mat_recall_auto_expected(n_samples, d_out);
	for (size_t n = 0; n < n_samples; n++) {
		for (size_t i = 0; i < d_out; i++) {
			size_t n_in = 0;
			size_t n_mul = 0;
			for (size_t j = 0; j < d_in; j++) {
				n_in += mat_in(n, j) ? 1 : 0;
				n_mul += mat_in(n, j) && bin_expected(i, j) ? 1 : 0;
			}
			mat_recall_expected(n, i) = n_mul >= th ? 1 : 0;
			mat_recall_auto_expected(n, i) = n_mul >= n_in ? 1 : 0;
		}
	}

	// Train the BiNAM and compare the result to the version computed above
	BiNAM bin(mat_out.cols(), mat_in.cols());
	bin.train(mat_in, mat_out);
	EXPECT_EQ(bin_expected, bin);

	// Recall the trained input and compare the result to the value
	BinaryMatrix mat_recall = bin.recall(mat_in, th);
	BinaryMatrix mat_recall_auto = bin.recall_auto_threshold(mat_in);
	EXPECT_EQ(mat_recall_expected, mat_recall);
	EXPECT_EQ(mat_recall_auto_expected, mat_recall_auto);
}
}
