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

/**
 * @file binam.hpp
 *
 * Contains the actual implementation of the binary associative memory. A binary
 * associative memory allows lossy mapping between an input bit vector and an
 * output bit vector. Operation of the BiNAM is separated into two phases: a
 * training phase and a recall phase. During training, the internal storage
 * matrix of the binary associative memory is filled with content, during the
 * recall stage
 *
 * @author Christoph Jenzen
 * @author Andreas Stöckel
 */

#pragma once

#ifndef CPPNAM_BINAM_BINAM_HPP
#define CPPNAM_BINAM_BINAM_HPP

#include <cppnam/binam/binary_matrix.hpp>
#include <cppnam/binam/population_count.hpp>

namespace nam {
/**
 * The BiNAM class is the BinaryMatrix class with additional instructions used
 * by the BiNAM_Container. This is basically still a simple matrix class, which
 * knows the concept of training and recalling.
 */
class BiNAM: public BinaryMatrix {
private:
	/**
	 * Trains the given pair of sample. This function is used internally by the
	 * BiNAM class.
	 */
	BiNAM &train_vec_internal(ConstBitRowIterator in, ConstBitRowIterator out);

public:
	using BinaryMatrix::BinaryMatrix;

	/**
	 * Constructor. Does the same as the standard binary matrix constructor, yet
	 * has semantic parameter name.
	 */
	BiNAM(size_t dim_out, size_t dim_in) : BinaryMatrix(dim_out, dim_in) {}

	/**
	 * Returns the dimensionality of the input vectors.
	 */
	size_t dim_in() { return cols(); }

	/**
	 * Returns the dimensionality of the output vectors.
	 */
	size_t dim_out() { return rows();}

	/**
	 * Trains a sample input output pair. Each row of the given matrices
	 * contains exactly one sample.
	 *
	 * @param in is the input matrix or vector that should be trained.
	 * @param out is the output matrix or vector that should be trained.
	 * @return a reference at this BiNAM instance for chained member function
	 * calls.
	 */
	BiNAM &train(const BinaryMatrix &in, const BinaryMatrix &out);

	/**
	 * Recall procedure for a single vector or a matrix of input vectors.
	 *
	 * @param in is a matrix containing the vectors that should be recalled.
	 * Each input vector is stored in one row, each row must have exactly
	 * dim_in() columns.
	 * @param threshold is minimal activation of a cell in the output vector
	 * for a "one" to be returned.
	 */
	BinaryMatrix recall(const BinaryMatrix &in, size_t threshold);

	/**
	 * Recall procedure for a single vector or a matrix of input vectors. Note
	 * that this method implements a special case with the threshold set to the
	 * number of one bits in the input vector.
	 *
	 * @param in is a matrix containing the vectors that should be recalled.
	 * Each input vector is stored in one row, each row must have exactly
	 * dim_in() columns.
	 */
	BinaryMatrix recall_auto_threshold(const BinaryMatrix &in);

	/**
	 * Calculation of false positives and negative for single sample.
	 *
	 * @param out is the original sample
	 * @param recall the one with errors (the recalled sample)
	 */
/*	SampleError false_bits(BinaryVector out, BinaryVector recall)
	{
		SampleError error;
		for (size_t i = 0; i < numberOfCells(out.size()); i++) {
			T temp = out.get_cell(i) ^ recall.get_cell(i);
			error.fp += population_count(temp & recall.get_cell(i));
			error.fn += population_count(temp & out.get_cell(i));
		}
		return error;
	}*/

	/**
	 * Calculation of false positives and negative for the matrix
	 * @param out is the original sample matrix
	 * @param recall the one with errors (the recalled one)
	 */
/*	std::vector<SampleError> false_bits_mat(BinaryMatrix out,
	                                        BinaryMatrix res)
	{
		if (res.rows() > out.rows()) {
			std::stringstream ss;
			ss << res.rows() << " out of range for output matrix of size "
			   << out.rows() << std::endl;
			throw std::out_of_range(ss.str());
		}
		std::vector<SampleError> error(res.rows());
		for (size_t i = 0; i < res.rows(); i++) {
			error[i] = false_bits(out.row_vec(i), res.row_vec(i));
		}
		return error;
	}*/
};
}

#endif /* CPPNAM_BINAM_BINAM_HPP */
