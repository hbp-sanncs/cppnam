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

#include <core/binam.hpp>
#include <core/entropy.hpp>
#include <core/parameters.hpp>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace nam;

void show_progress(float progress)
{
	const int WIDTH = 50;
	float perc = progress * 100.0;
	std::cerr << std::setw(8) << std::setprecision(4) << perc << "% [";
	for (int i = 0; i < WIDTH; i++) {
		bool cur = i * 100 / WIDTH < perc;
		bool prev = (i - 1) * 100 / WIDTH < perc;
		if (cur && prev) {
			std::cerr << "=";
		}
		else if (prev) {
			std::cerr << ">";
		}
		else {
			std::cerr << " ";
		}
	}
	std::cerr << "]   \r";
}

void information_graph(size_t bits_in, size_t bits_out, size_t ones_in,
                       size_t ones_out, size_t max_sample)
{
	std::ofstream file;
	file.open("data.txt", std::ios::out);
	for (size_t i = 1; i <= max_sample; i++) {
		DataParameters params(bits_in, bits_out, ones_in, ones_out, i);
		BiNAM_Container<uint64_t> binam(params);
		binam.set_up().recall();
		auto se = binam.false_bits();
		double info = entropy_hetero(params, se);
		file << i << "," << info << ","
		     << binam.sum_false_bits(binam.false_bits()).fp << "\n";
		show_progress(double(i) / double(max_sample));
	}
	std::cerr << std::endl;
	file.close();
}

int main(int argc, char *argv[])
{
	if (argc != 4 && argc != 6 && argc != 7) {
		std::cerr
		    << "Usage: ./data_generator <BITS> <ONES> <SAMPLES> "
		    << "or <BITS_IN> <BITS_OUT> <ONES_IN> <ONES_OUT> <MAX_SAMPLES>"
		    << "or <BITS_IN> <BITS_OUT> <ONES_IN> <ONES_OUT> <SAMPLES> <bla>"
		    << std::endl;
		return 1;
	}
	if (argc == 6) {
		information_graph(std::stoi(argv[1]), std::stoi(argv[2]),
		                  std::stoi(argv[3]), std::stoi(argv[4]),
		                  std::stoi(argv[5]));
		return 0;
	}

	int n_bits = std::stoi(argv[1]);
	int n_bits_out = std::stoi(argv[1]);
	int n_ones = std::stoi(argv[2]);
	int n_ones_out = std::stoi(argv[2]);
	int n_samples = std::stoi(argv[3]);
	if (argc == 7) {
		n_bits = std::stoi(argv[1]);
		n_bits_out = std::stoi(argv[2]);
		n_ones = std::stoi(argv[3]);
		n_ones_out = std::stoi(argv[4]);
		n_samples = std::stoi(argv[5]);
	}
	auto params =
	    DataParameters(n_bits, n_bits_out, n_ones, n_ones_out, n_samples);
	std::cout << "The optimal sample count: "
	          << DataParameters::optimal_sample_count(params) << std::endl;
	std::cout << n_bits << " bits, " << n_ones << " ones and " << n_samples
	          << " samples" << std::endl;
	BiNAM_Container<uint64_t> binam(params);
	binam.set_up().recall().analysis();
	binam.trained_matrix().print();
	// binam.print();
	return 0;
}
