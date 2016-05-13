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

#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "util/read_json.hpp"
#include "util/spiking_parameters.hpp"

namespace nam {
TEST(ReadJSON, json_to_map)
{
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	auto map = json_to_map<float>(json["network"]);
	EXPECT_EQ(map.end(), map.find("neuron_type"));
	EXPECT_EQ(0.01, map.find("weight")->second);
	EXPECT_EQ(1.0, map.find("input_burst_size")->second);
	EXPECT_EQ(1.0, map.find("output_burst_size")->second);
	EXPECT_EQ(100.0, map.find("time_window")->second);
	EXPECT_EQ(2.0, map.find("isi")->second);
	EXPECT_EQ(2.0, map.find("sigma_t")->second);
	EXPECT_EQ(0.0, map.find("sigma_offs")->second);
	EXPECT_EQ(0.0, map.find("p0")->second);
	EXPECT_EQ(0.0, map.find("p1")->second);
	EXPECT_EQ(100.0, map.find("general_offset")->second);
}

TEST(ReadJSON, read_check)
{
	std::ifstream ifs("test.json", std::ifstream::in);
	cypress::Json json(ifs);
	auto map = json_to_map<float>(json["network"]);
	EXPECT_ANY_THROW(
	    read_check<float>(map, std::vector<std::string>({"input_burst_size"}),
	                      std::vector<float>({0})));

	auto res =
	    read_check<float>(map, NetworkParameters::names,
	                      std::vector<float>(NetworkParameters::names.size(), 0));
	EXPECT_EQ(0.01, res[8]);
	EXPECT_EQ(1.0, res[0]);
	EXPECT_EQ(1.0, res[1]);
	EXPECT_EQ(100.0, res[2]);
	EXPECT_EQ(2.0, res[3]);
	EXPECT_EQ(2.0, res[4]);
	EXPECT_EQ(0.0, res[5]);
	EXPECT_EQ(0.0, res[6]);
	EXPECT_EQ(0.0, res[7]);
	EXPECT_EQ(100.0, res[9]);
}
}