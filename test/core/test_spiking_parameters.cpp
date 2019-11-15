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

#include <cypress/cypress.hpp>

#include "gtest/gtest.h"

#include <fstream>
#include <sstream>
#include <vector>

#include "core/spiking_parameters.hpp"

namespace nam {
using cypress::Real;

static const std::string test_json =
    "{\n"
    "\t\"data\": {\n"
    "\t\t\"n_bits_in\": 100,\n"
    "\t\t\"n_bits_out\": 100,\n"
    "\t\t\"n_ones_in\": 4,\n"
    "\t\t\"n_ones_out\": 4,\n"
    "\t\t\"n_samples\" : 1000\n"
    "\t},\n"
    "\n"
    "\t\"network\": {\n"
    "\t\t\"params\": {\n"
    "\t\t\t\"e_rev_E\": 0.0,\n"
    "\t\t\t\"v_rest\": -70.0,\n"
    "\t\t\t\"v_reset\": -80.0,\n"
    "\t\t\t\"v_thresh\": -57.0,\n"
    "\t\t\t\"tau_syn_E\": 2.0,\n"
    "\t\t\t\"tau_refrac\": 0.0,\n"
    "\t\t\t\"tau_m\": 50.0,\n"
    "\t\t\t\"cm\": 0.2\n"
    "\t\t},\n"
    "\t\t\"neuron_type\": \"IF_cond_exp\",\n"
    "\t\t\"weight\": 0.01,\n"
    "\t\t\"input_burst_size\": 1,\n"
    "\t\t\"output_burst_size\": 1,\n"
    "\t\t\"time_window\": 100.0,\n"
    "\t\t\"isi\": 2.0,\n"
    "\t\t\"sigma_t\": 2.0,\n"
    "\t\t\"sigma_offs\": 0.0,\n"
    "\t\t\"p0\": 0.0,\n"
    "\t\t\"p1\": 0.0,\n"
    "\t\t\"general_offset\" : 100\n"
    "\t}\n"
    "}\n"
    "";

TEST(SpikingParameters, NeuronParameters)
{
	std::stringstream ss(test_json);
	cypress::Json json = cypress::Json::parse(ss);
	const cypress::IfCondExp &neurontype = cypress::IfCondExp::inst();
	std::ofstream out;
	auto params = NeuronParameters(neurontype, json["network"], out);
	std::vector<cypress::Real> parameter = params.parameter();
	EXPECT_NEAR(Real(0.2), parameter[0], 1e-8);
	EXPECT_NEAR(Real(50), parameter[1], 1e-8);
	EXPECT_NEAR(Real(2), parameter[2], 1e-8);
	EXPECT_NEAR(Real(5), parameter[3], 1e-8);  // Cypress standard
	EXPECT_NEAR(Real(0.0), parameter[4], 1e-8);
	EXPECT_NEAR(Real(-70), parameter[5], 1e-8);
	EXPECT_NEAR(Real(-57), parameter[6], 1e-8);
	EXPECT_NEAR(Real(-80), parameter[7], 1e-8);
	EXPECT_NEAR(Real(0), parameter[8], 1e-8);
	EXPECT_NEAR(Real(-70), parameter[9], 1e-8);  // Cypress standard
	EXPECT_NEAR(Real(0), parameter[10], 1e-8);
}

TEST(SpikingParameters, NetworkParameters)
{
	std::stringstream ss(test_json);
	cypress::Json json = cypress::Json::parse(ss);
	std::ofstream out;
	auto params = NetworkParameters(json["network"], out);
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
