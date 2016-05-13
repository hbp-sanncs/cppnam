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
#include <fstream>
#include <string>

#include "cypress/cypress.hpp"
#include "util/spiking_binam.hpp"

using namespace nam;
int main(int argc, const char *argv[])
{
	if (argc != 4 && argc != 3 && !cypress::NMPI::check_args(argc, argv)) {
		std::cout << "Usage: " << argv[0] << " <SIMULATOR> <FILE> [NMPI]"
		          << std::endl;
		return 1;
	}

	if (argc == 4 && std::string(argv[3]) == "NMPI" &&
	    !cypress::NMPI::check_args(argc, argv)) {
		cypress::NMPI(argv[1], argc, argv);
		return 0;
	}
	
	std::ifstream ifs(argv[2], std::ifstream::in);
	cypress::Json json(ifs);
	std::ofstream ofs("data.txt", std::ofstream::app);
	
	auto time = std::time(NULL);
	ofs << "Spiking Binam from " << std::ctime(&time) << std::endl;
	SpikingBinam binam(json, ofs);
	binam.build();
	std::cout << "Building complete" << std::endl;
	binam.run(argv[1]);
	std::cout << "Run complete" << std::endl;
	binam.eval_output(ofs);
	// NMPI 3 argument list of files

	ofs << std::endl << "____________________________________________" 
	<< std::endl;

	return 0;
}
