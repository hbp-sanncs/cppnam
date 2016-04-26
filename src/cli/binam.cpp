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

#include <csignal>
#include <iostream>

#include <util/binam.hpp>
#include <util/parameters.hpp>

using namespace nam;

int main(int argc, char *argv[])
{
	if (argc != 4) {
		std::cerr << "Usage: ./data_generator <BITS> <ONES> <SAMPLES>"
		          << std::endl;
		return 1;
	}
	int n_bits = std::stoi(argv[1]);
	int n_ones = std::stoi(argv[2]);
	int n_samples = std::stoi(argv[3]);
	DataParameters params(n_bits, n_bits, n_ones, n_ones, n_samples);
	BiNAM_Container<uint32_t> binam(params);
	binam.set_up().recall().analysis();
	
	
}