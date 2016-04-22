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

#pragma once

#ifndef CPPNAM_UTIL_BINAM_HPP
#define CPPNAM_UTIL_BINAM_HPP
#include <sstream>
#include <stdexcept>
#include "data.hpp"
#include "util/binary_matrix.hpp"

namespace nam {
/**
 * The BiNAM class is the BinaryMatrix class with additional instructions used
 * by the BiNAM_Container. This is basically still a simple matrix class, which
 * knows the concept of training and recalling.
 */
template <typename T>
class BiNAM : public BinaryMatrix<T> {
private:
	/**
	 * Training of a sample pair. Dimensions are not checked, is only for
	 * internal used
	 */
	BiNAM<T> &train_vec(BinaryVector<uint8_t> in, BinaryVector<uint8_t> out)
	{
		for (size_t i = 0; i < out.size(); i++) {
			if (out.get_bit(i) != 0) {
				for (size_t j = 0; j < Base::numberOfCells(Base::cols()); j++)
					Base::set_cell(i, j, Base::get_cell(i, j) |
					                         (Base::intMax & in.get_cell(j)));
			}
		}
		return *this;
	}

public:
	using Base = BinaryMatrix<T>;
	/**
	 * Constructor - nothing to do here
	 */
	BiNAM(){};
	BiNAM(size_t input, size_t output) : BinaryMatrix<T>(output, input){};

	/**
	 * Training of a sample pair with checking of dimensions
	 */
	BiNAM<T> &train_vec_check(BinaryVector<uint8_t> in,
	                          BinaryVector<uint8_t> out)
	{
		if (in.size() != Base::cols() || out.size() != Base::rows()) {
			std::stringstream ss;
			ss << "[" << in.size() << ", " << out.size()
			   << "] out of range for matrix of size " << Base::cols() << " x "
			   << Base::rows() << std::endl;
			throw std::out_of_range(ss.str());
		}
		return train_vec(in, out);
	}

	/**
	 * Training of whole matrices, should be favoured for using
	 */
	BiNAM<T> &train_mat(BinaryMatrix<T> in, BinaryMatrix<T> out)
	{
		if (in.cols() != Base::cols() || out.cols() != Base::rows() ||
		    in.rows() != out.rows()) {
			std::stringstream ss;
			ss << "[" << in.size() << ", " << out.size()
			   << "] out of range for matrix of size " << Base::size()
			   << std::endl;
			throw std::out_of_range(ss.str());
		}
		for (size_t i = 0; i < in.rows(); i++) {
			train_vec(in.row_vec(i), out.row_vec(i));
		};
		return *this;
	}

	/**
	 * Sum of all set bits of a BinaryVector. Used for recall
	 */
	size_t digit_sum(BinaryVector<T> vec)
	{
		size_t sum = 0;
		for (size_t i = 0; i < vec.cols(); i++) {
			sum += (int)vec.get_bit(i);
		};
		return sum;
	}

	/*
	 * Recall procedure for a single sample, @param thresh is the threshold
	 */
	BinaryVector<T> recall(BinaryVector<T> in, size_t thresh)
	{
		BinaryVector<T> vec(Base::rows());
		for (size_t i = 0; i < Base::rows(); i++) {
			size_t sum = digit_sum(in.VectorMult(Base::row_vec(i)));
			if (sum >= thresh) {
				vec.set_bit(i);
			};
		};

		return vec;
	}

	/*
	 * Recall procedure for a matrix of samples, @param thresh is the threshold
	 */
	BinaryMatrix<T> recallMat(BinaryMatrix<T> in, size_t thresh)
	{
		if (in.cols() != Base::cols()) {
			std::stringstream ss;
			ss << in.size() << " out of range for matrix of size "
			   << Base::cols() << std::endl;
			throw std::out_of_range(ss.str());
		}
		BinaryMatrix<T> res(in.rows(), Base::cols());
		for (size_t i = 0; i < res.rows(); i++) {
			res.write_vec(i, recall(in.row_vec(i), thresh));
		};
		return res;
	}
};

class BiNAM_Container;
}

#endif /* CPPNAM_UTIL_BINAM_HPP */
