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
	EXPECT_EQ(float(0.2), parameter[0]);
	EXPECT_EQ(float(50), parameter[1]);
	EXPECT_EQ(float(2), parameter[2]);
	EXPECT_EQ(float(5), parameter[3]);  // Cypress standard
	EXPECT_EQ(float(0.0), parameter[4]);
	EXPECT_EQ(float(-70), parameter[5]);
	EXPECT_EQ(float(-57), parameter[6]);
	EXPECT_EQ(float(-80), parameter[7]);
	EXPECT_EQ(float(0), parameter[8]);
	EXPECT_EQ(float(-70), parameter[9]);  // Cypress standard
	EXPECT_EQ(float(0), parameter[10]);
}

TEST(SpikingParameters, NetworkParameters)
{
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	auto params = NetworkParameters(json["network"]);
	EXPECT_EQ(1, params.input_burst_size());
	EXPECT_EQ(1, params.output_burst_size());
	EXPECT_EQ(100, params.time_window());
	EXPECT_EQ(2.0, params.isi());
	EXPECT_EQ(2.0, params.sigma_t());
	EXPECT_EQ(0.0, params.sigma_offs());
	EXPECT_EQ(0, params.p0());
	EXPECT_EQ(0, params.p1());
	EXPECT_EQ(0.01, params.weight());
}



}