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
#include <fstream>
#include <functional>
#include <memory>
#include <string>

#include "cypress/cypress.hpp"
#include "core/experiment.hpp"
#include "core/spiking_binam.hpp"
#include "recurrent/spiking_rec_binam.hpp"

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

	signal(SIGINT, int_handler);

	cypress::Json json;
	{
		std::ifstream ifs(argv[2], std::ifstream::in);
		json << ifs;
	}
	
	std::string mode;
	if(json.find("mode")!=json.end()){
		mode = json["mode"];
	}else {
		mode = "standard";
	}
	
	if(mode=="standard"){
		Experiment exp(
	    json, argv[1], [] (cypress::Json json, DataParameters params,
	        DataGenerationParameters data_params, std::ostream &out, bool one,
	        bool two) { return std::make_unique<SpikingBinam>(json, params, data_params, out, one, two); });
		exp.run(argv[2]);
	}else if (mode == "recurrent"){
		Experiment exp(
	    json, argv[1], [] (cypress::Json json, DataParameters params,
	        DataGenerationParameters data_params, std::ostream &out, bool one,
	        bool two) { return std::make_unique<SpikingRecBinam>(json, params, data_params, out, one, two); });
		exp.run(argv[2]);
	}
	return 0;
}
