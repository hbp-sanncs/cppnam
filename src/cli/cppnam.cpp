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

#include <array>
#include <iostream>
#include <limits>

#include <binnf/serialiser.hpp>

using namespace nam;
using namespace nam::binnf;

const auto INT = Serialiser::NumberType::INT;
const auto FLOAT = Serialiser::NumberType::FLOAT;

const int TYPE_SOURCE = 0;
const int TYPE_IF_COND_EXP = 1;
const int TYPE_AD_EX = 2;

const int ALL_NEURONS = std::numeric_limits<int>::max();

static const Serialiser::Header POPULATIONS_HEADER = {
    {"count", "type", "record_spikes", "record_v", "record_gsyn_exc",
     "record_gsyn_inh"},
    {INT, INT, INT, INT, INT, INT}};

static const Serialiser::Header CONNECTIONS_HEADER = {
    {"pid_src", "pid_tar", "nid_src", "nid_tar", "weight", "delay"},
    {INT, INT, INT, INT, FLOAT, FLOAT}};

static const Serialiser::Header PARAMETERS_HEADER = {
    {"pid", "nid", "v_rest", "cm", "tau_m", "tau_refrac", "tau_syn_E",
     "tau_syn_I", "e_rev_E", "e_rev_I", "v_thresh", "v_reset", "i_offset"},
    {INT, INT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT,
     FLOAT, FLOAT}};

static const Serialiser::Header TARGET_HEADER = {{"pid", "nid"}, {INT, INT}};

static const Serialiser::Header SPIKE_TIMES_HEADER = {{"times"}, {FLOAT}};

int main()
{
	Serialiser::serialise(
	    std::cout, {"populations", POPULATIONS_HEADER,
	                make_matrix<Serialiser::Number, 2, 6>({{
	                    {10, TYPE_SOURCE, false, false, false, false},
	                    {20, TYPE_IF_COND_EXP, false, false, false, false},
	                }})});

	Serialiser::serialise(std::cout, {"connections", CONNECTIONS_HEADER,
	                                  make_matrix<Serialiser::Number, 3, 6>({{
	                                      {0, 1, 0, 0, 0.1, 0.0},
	                                      {0, 1, 1, 1, 0.1, 0.0},
	                                      {0, 1, 1, 2, 0.1, 0.0},
	                                  }})});

	Serialiser::serialise(std::cout, {"parameters", PARAMETERS_HEADER,
	                                  make_matrix<Serialiser::Number, 3, 13>({{
	                                      {1, 0, -65.0, 1.0, 20.0, 0.0, 5.0,
	                                       5.0, 0.0, -70.0, -50.0, -65.0, 0.0},
	                                      {1, 1, -65.0, 1.0, 20.0, 0.0, 5.0,
	                                       5.0, 0.0, -70.0, -50.0, -65.0, 0.0},
	                                      {1, 2, -65.0, 1.0, 20.0, 0.0, 5.0,
	                                       5.0, 0.0, -70.0, -50.0, -65.0, 0.0},
	                                  }})});

	Serialiser::serialise(
	    std::cout,
	    {"target", TARGET_HEADER,
	     make_matrix<Serialiser::Number, 1, 2>({{{0, ALL_NEURONS}}})});

	Serialiser::serialise(
	    std::cout, {"spike_times", SPIKE_TIMES_HEADER,
	                make_matrix<Serialiser::Number>(
	                    {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0})});
}

