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

#include "binary_matrix.hpp"

namespace nam {/*
template <typename T>
std::vector<T> BinaryMatrix<T>::bitVector(const std::vector<bool> &data)
{
	const uint32_t n = data.size();
	const uint32_t nCells = numberOfCells(n);

	std::vector<T> res(nCells);
	for (uint32_t i = 0, cell = 0; cell < nCells; cell++) {
		T c = 0;
		for (uint32_t j = 0; j < intWidth && i < n; j++, i++) {
			if (data[i]) {
				c = c | 1 << j;
			}
		}
		res[cell] = c;
	}
	return res;
}

template <typename T>
std::vector<T> BinaryMatrix<T>::cellVector(const std::vector<bool> &data)
{
	const uint32_t n = data.size();

	std::vector<T> res(n);
	for (uint32_t i = 0; i < n; i++) {
		res[i] = data[i] ? intMax : 0;
	}
	return res;
}*/
}
