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

#include "parameters.hpp"
#include "entropy.hpp"
#include "optimisation.hpp"

namespace nam {

DataParameters DataParameters::optimal(const size_t n_bits,
                                       const size_t n_samples,
                                       const size_t n_bits_in,
                                       const size_t n_bits_out)
{
}

size_t DataParameters::optimal_sample_count(const DataParameters &params)
{
	// TODO
}
}