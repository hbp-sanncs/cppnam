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

#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "util/read_json.hpp"
#include "core/spiking_parameters.hpp"

namespace nam {

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

TEST(ReadJSON, json_to_map)
{
	std::stringstream ss(test_json);
	cypress::Json json(ss);

	auto map = json_to_map<float>(json["network"]);
	EXPECT_EQ(map.end(), map.find("neuron_type"));
	EXPECT_NEAR(0.01, map.find("weight")->second, 1e-8);
	EXPECT_NEAR(1.0, map.find("input_burst_size")->second, 1e-8);
	EXPECT_NEAR(1.0, map.find("output_burst_size")->second, 1e-8);
	EXPECT_NEAR(100.0, map.find("time_window")->second, 1e-8);
	EXPECT_NEAR(2.0, map.find("isi")->second, 1e-8);
	EXPECT_NEAR(2.0, map.find("sigma_t")->second, 1e-8);
	EXPECT_NEAR(0.0, map.find("sigma_offs")->second, 1e-8);
	EXPECT_NEAR(0.0, map.find("p0")->second, 1e-8);
	EXPECT_NEAR(0.0, map.find("p1")->second, 1e-8);
	EXPECT_NEAR(100.0, map.find("general_offset")->second, 1e-8);
}

TEST(ReadJSON, read_check)
{
	std::stringstream ss(test_json);
	cypress::Json json(ss);

	auto map = json_to_map<float>(json["network"]);
	auto res = read_check<float>(
	    map, NetworkParameters::names,
	    std::vector<float>(NetworkParameters::names.size(), 0));
	EXPECT_NEAR(0.01, res[8], 1e-8);
	EXPECT_NEAR(1.0, res[0], 1e-8);
	EXPECT_NEAR(1.0, res[1], 1e-8);
	EXPECT_NEAR(100.0, res[2], 1e-8);
	EXPECT_NEAR(2.0, res[3], 1e-8);
	EXPECT_NEAR(2.0, res[4], 1e-8);
	EXPECT_NEAR(0.0, res[5], 1e-8);
	EXPECT_NEAR(0.0, res[6], 1e-8);
	EXPECT_NEAR(0.0, res[7], 1e-8);
	EXPECT_NEAR(0.0, res[9], 1e-8);
	EXPECT_NEAR(100.0, res[10], 1e-8);

	map = json_to_map<float>(json["network"]);
	EXPECT_ANY_THROW(
	    read_check<float>(map, std::vector<std::string>({"input_burst_size"}),
	                      std::vector<float>({0})));
}
}
