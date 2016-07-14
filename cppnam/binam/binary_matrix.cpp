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

#include <cppnam/binam/binary_matrix.hpp>

namespace nam {
#ifndef NDEBUG
void BinaryMatrix::check_range(size_t row, size_t col) const
{
	if (row >= m_rows || col >= m_cols) {
		std::stringstream ss;
		ss << "(" << row << ", " << col
		   << ") out of range for matrix of size " << m_rows << " x "
		   << m_cols;
		throw std::out_of_range(ss.str());
	}
}

void BinaryMatrix::check_range_row(size_t row) const
{
	if (row >= m_rows) {
		std::stringstream ss;
		ss << "Row index " << row << "out of bounds for matrix of size "
		   << m_rows << " x " << m_cols;
		throw std::out_of_range(ss.str());
	}
}

void BinaryMatrix::check_range_col(size_t col) const
{
	if (col >= m_cols) {
		std::stringstream ss;
		ss << "Column index " << col << "out of bounds for matrix of size "
		   << m_rows << " x " << m_cols;
		throw std::out_of_range(ss.str());
	}
}
#endif /* NDEBUG */
}
