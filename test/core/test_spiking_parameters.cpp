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

#include "gtest/gtest.h"

#include <fstream>
#include <vector>

#include <cypress/cypress.hpp>
#include "core/spiking_parameters.hpp"

namespace nam {

TEST(SpikingParameters, NeuronParameters)
{
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	cypress::IfCondExp neurontype = cypress::IfCondExp::inst();
	auto params = NeuronParameters(neurontype, json["network"]);
	std::vector<float> parameter = params.parameter();
	EXPECT_NEAR(float(0.2), parameter[0], 1e-8);
	EXPECT_NEAR(float(50), parameter[1], 1e-8);
	EXPECT_NEAR(float(2), parameter[2], 1e-8);
	EXPECT_NEAR(float(5), parameter[3], 1e-8);  // Cypress standard
	EXPECT_NEAR(float(0.0), parameter[4], 1e-8);
	EXPECT_NEAR(float(-70), parameter[5], 1e-8);
	EXPECT_NEAR(float(-57), parameter[6], 1e-8);
	EXPECT_NEAR(float(-80), parameter[7], 1e-8);
	EXPECT_NEAR(float(0), parameter[8], 1e-8);
	EXPECT_NEAR(float(-70), parameter[9], 1e-8);  // Cypress standard
	EXPECT_NEAR(float(0), parameter[10], 1e-8);
}

TEST(SpikingParameters, NetworkParameters)
{
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	auto params = NetworkParameters(json["network"]);
	EXPECT_NEAR(1, params.input_burst_size(), 1e-8);
	EXPECT_NEAR(1, params.output_burst_size(), 1e-8);
	EXPECT_NEAR(100, params.time_window(), 1e-8);
	EXPECT_NEAR(2.0, params.isi(), 1e-8);
	EXPECT_NEAR(2.0, params.sigma_t(), 1e-8);
	EXPECT_NEAR(0.0, params.sigma_offs(), 1e-8);
	EXPECT_NEAR(0, params.p0(), 1e-8);
	EXPECT_NEAR(0, params.p1(), 1e-8);
	EXPECT_NEAR(0.01, params.weight(), 1e-8);
}



}