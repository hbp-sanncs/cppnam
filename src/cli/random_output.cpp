/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Christoph Jenzen
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

#include <cmath>

#include "util/binary_matrix.hpp"
#include "core/binam.hpp"
#include "core/entropy.hpp"
#include "core/parameters.hpp"

using namespace nam;

/**
 * This program calculates the information stored in a BiNAM with random output
 */
int main(int argc, char *argv[])
{
	if (argc != 6) {
		std::cerr << "Usage: ./random_output <BITS_IN> <BITS_OUT> <ONES_IN> "
		             "<ONES_OUT> <SAMPLES>"
		          << std::endl;
		return 1;
	}

	int n_bits_in = std::stoi(argv[1]);
	int n_bits_out = std::stoi(argv[2]);
	int n_ones_in = std::stoi(argv[3]);
	int n_ones_out = std::stoi(argv[4]);
	int n_samples = std::stoi(argv[5]);

	auto binam = BiNAM_Container<uint64_t>(
	    DataParameters(n_bits_in, n_bits_out, n_ones_in, n_ones_out, n_samples),
	    DataGenerationParameters(1234, 1, 1, 1));
	binam.set_up().recall();
	size_t info_th = binam.analysis().Info;
	double average = 0.0;
	double deviation = 0.0;
	for (size_t i = 0; i < 50; i++) {
		for (size_t j = 0; j < 10; j++) {
			auto res_mat =
			    DataGenerator(true, false, false)
			        .generate<uint64_t>(n_bits_out, n_ones_out, n_samples);
			auto res = binam.analysis(res_mat);
			float info_res = float(res.Info) / float(info_th);
			average += info_res;
			deviation += info_res * info_res;
			std::cout << i << ", " << info_res << std::endl;
		}
	}
	deviation = std::sqrt((deviation - average * average / 500) / 499);
	average = average / 500;
	std::cout << "Average : " << average << std::endl;
	std::cout << "Standard deviation : " << deviation << std::endl;
	return 0;
}