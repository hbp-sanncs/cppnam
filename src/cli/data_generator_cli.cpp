/*
 *  CppNAM -- C++ Neural Associative Memory Simulator
 *  Copyright (C) 2016  Andreas Stöckel
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

#include <csignal>
#include <iomanip>
#include <iostream>
#include <string>

#include <util/data.hpp>

using namespace nam;

/**
 * SIGINT handler. Sets the global "cancel" flag to true when called once,
 * terminates the program if called twice. This allows to terminate the program,
 * even if it is not responsive (the cancel flag is not checked).
 */
static bool cancel = false;
void int_handler(int)
{
	if (cancel) {
		exit(1);
	}
	cancel = true;
}

bool show_progress(float progress)
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
	return !cancel;
}

int main(int argc, char *argv[])
{
	signal(SIGINT, int_handler);

	if (argc != 4) {
		std::cerr << "Usage: ./data_generator <BITS> <ONES> <SAMPLES>"
		          << std::endl;
		return 1;
	}

	int n_bits = std::stoi(argv[1]);
	int n_ones = std::stoi(argv[2]);
	int n_samples = std::stoi(argv[3]);

	if (n_bits < 0 || n_ones < 0 || n_samples < 0) {
		std::cerr << "Invalid parameter combination, all arguments "
		          << "must be positive!" << std::endl;
		return 1;
	}

	if (n_ones > n_bits) {
		std::cerr << "<ONES> must be smaller than <BITS>!" << std::endl;
		return 1;
	}

	// Generate the requested data
	std::cerr << "Generating data..." << std::endl;
	auto data =
	    DataGenerator().generate(n_bits, n_ones, n_samples, show_progress);
	std::cerr << std::endl;

	// Print a non-sparse version of the matrix to std::cout
	for (size_t i = 0; i < data.n_rows; i++) {
		for (size_t j = 0; j < data.n_cols; j++) {
			std::cout << (int)(data(i, j));
		}
		std::cout << std::endl;
	}
	return 0;
}
