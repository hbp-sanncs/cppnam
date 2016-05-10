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

#include "cypress/cypress.hpp"
#include "util/spiking_binam.hpp"


using namespace nam;
int main(int argc,const char *argv[])
{
	if (argc != 2 && !cypress::NMPI::check_args(argc, argv)) {
		std::cout << "Usage: " << argv[0] << " <SIMULATOR>" << std::endl;
		return 1;
	}
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	SpikingBinam binam(json);
	binam.build();
	std::cout << "Building complete"<< std::endl;
	binam.run(argv[1]);
	std::cout << "Run complete"<< std::endl;
	binam.eval_output();
	
	
	return 0;
}