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

#include <random>
#include <sstream>
#include <stdexcept>

#include <cppnam/binam/binam.hpp>
#include <cppnam/binam/population_count.hpp>

namespace nam {

BiNAM &BiNAM::train(const BinaryMatrix &in, const BinaryMatrix &out)
{
	// Make sure the dimensions are correct
	if (in.cols() != dim_in() || out.cols() != dim_out() ||
	    in.rows() != out.rows()) {
		throw std::out_of_range("Training data dimensionality missmatch");
	}

	// Iterate over each sample
	const size_t n_samples = in.rows();
	for (size_t n = 0; n < n_samples; n++) {
		// Iterate over each bit in the current output vector
		auto it_out = out.begin_row(n);
		for (size_t i = 0; i < dim_out(); i++, it_out++) {
			// Skip entries in the output vector which are set to zero
			if (!(*it_out)) {
				continue;
			}

			// Otherwise, "OR" the current input vector onto the i-th row of the
			// storage matrix.
			auto it_in = in.begin_row(n);
			auto it_tar = begin_row(i);
			const auto it_tar_end = end_row(i);
			for (; it_tar < it_tar_end; it_tar.next_cell(), it_in.next_cell()) {
				it_tar.cell(it_tar.cell() | it_in.cell());
			}
		}
	}
	return *this;
}

BinaryMatrix BiNAM::recall(const BinaryMatrix &in, size_t threshold)
{
	if (in.cols() != dim_in()) {
		std::stringstream ss;
		ss << "" << in.cols() << " out of range for matrix of size " << cols()
		   << std::endl;
		throw std::out_of_range(ss.str());
	}

	// Iterate over each sample in the input matrix and fill the output matrix
	const size_t n_samples = in.rows();
	BinaryMatrix res(n_samples, dim_out());
	for (size_t n = 0; n < n_samples; n++) {
		// Iterate over each row in the storage matrix
		for (size_t i = 0; i < dim_out(); i++) {
			// Iterate over each cell in the input vector and the i-th row
			// of the storage matrix
			auto it_in = in.begin_row(n);
			auto it_stor = begin_row(i);
			const auto it_stor_end = end_row(i);
			size_t sum = 0;
			for (; it_stor < it_stor_end;
			     it_stor.next_cell(), it_in.next_cell()) {
				sum += population_count<BinaryMatrixCell>(it_in.cell() &
				                                          it_stor.cell());
				if (sum >= threshold) {
					res(n, i) = true;
					break;
				}
			}
		}
	}
	return res;
}

BinaryMatrix BiNAM::recall_auto_threshold(const BinaryMatrix &in)
{
	if (in.cols() != dim_in()) {
		std::stringstream ss;
		ss << "" << in.cols() << " out of range for matrix of size " << cols()
		   << std::endl;
		throw std::out_of_range(ss.str());
	}

	// Iterate over each sample in the input matrix and fill the output matrix
	const size_t n_samples = in.rows();
	BinaryMatrix res(n_samples, dim_out());
	for (size_t n = 0; n < n_samples; n++) {
		// Iterate over each row in the storage matrix
		for (size_t i = 0; i < dim_out(); i++) {
			// Flag which indicates whether the i-th bit in the output vector
			// should be set to one.
			bool b = true;

			// Iterate over each cell in the input vector and the i-th row
			// of the storage matrix
			auto it_in = in.begin_row(n);
			auto it_stor = begin_row(i);
			const auto it_stor_end = end_row(i);
			for (; it_stor < it_stor_end;
			     it_stor.next_cell(), it_in.next_cell()) {
				const BinaryMatrixCell v = it_in.cell();
				const BinaryMatrixCell w = it_stor.cell();
				if ((v & w) != v) {
					b = false;
					break;
				}
			}
			if (b) {
				res(n, i) = true;
			}
		}
	}
	return res;
}
}

