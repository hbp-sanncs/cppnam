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

#include <string>

#include <cypress/cypress.hpp>

#include "util/binary_matrix.hpp"
#include "entropy.hpp"
#include "parameters.hpp"
#include "spike_trains.hpp"
#include "spiking_binam.hpp"

using namespace cypress;

namespace nam {
namespace {
/**
 * Helper to return an instance of a certain neuron type
 */
NeuronType detect_type(std::string neuron_type_str)
{
	if (neuron_type_str == "IF_cond_exp") {
		return IfCondExp::inst();
	}
	else if (neuron_type_str == "IfFacetsHardware1") {
		return IfFacetsHardware1::inst();
	}
	else if (neuron_type_str == "AdExp") {
		return EifCondExpIsfaIsta::inst();
	}
	throw CypressException("Invalid neuron type \"" + neuron_type_str + "\"");
}
}

std::vector<std::vector<float>> SpikingBinam::build_spike_times()
{
	BinaryMatrix<uint64_t> mat = m_BiNAM_Container.input_matrix();
	std::vector<std::vector<float>> res;
	for (size_t i = 0; i < mat.cols(); i++) {  // over all neruons
		std::vector<float> vec;
		for (size_t j = 0; j < mat.rows(); j++) {  // over all samples
			auto vec2 =
			    build_spike_train(m_networkParams, mat.get_bit(j, i),
			                      m_networkParams.general_offset() +
			                          j * m_networkParams.time_window());
			vec.insert(vec.end(), vec2.begin(), vec2.end());
		}
		res.emplace_back(vec);
	}

	return res;
}

SpikingBinam::SpikingBinam(Json &json,bool recall, std::ostream &out)
    : m_pop_source(m_net, 0), m_pop_output(m_net, 0)
{
	m_dataParams = DataParameters(json["data"]);
	m_dataParams.print(out);
	m_BiNAM_Container = BiNAM_Container<uint64_t>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	auto neuronType = detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	m_networkParams = NetworkParameters(json["network"], out);
	if (recall){
		m_BiNAM_Container.set_up().recall();
	}
}

SpikingBinam::SpikingBinam(Json &json, DataParameters params, std::ostream &out)
: m_pop_source(m_net, 0), m_pop_output(m_net, 0), m_dataParams(params)
{
	m_dataParams.print(out);
	m_BiNAM_Container = BiNAM_Container<uint64_t>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	auto neuronType = detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	m_networkParams = NetworkParameters(json["network"], out);
	m_BiNAM_Container.set_up().recall();
}

template <typename T>
PopulationBase SpikingBinam::add_typed_population()
{
	using Signals = typename T::Signals;
	using Parameters = typename T::Parameters;
	return m_net.create_population<T>(m_dataParams.bits_out(),
	                                  Parameters(m_neuronParams.parameter()),
	                                  Signals().record_spikes());
}

PopulationBase SpikingBinam::add_population(std::string neuron_type_str)
{
	if (neuron_type_str == "IF_cond_exp") {
		return add_typed_population<IfCondExp>();
	}
	else if (neuron_type_str == "IfFacetsHardware1") {
		return add_typed_population<IfFacetsHardware1>();
	}
	else if (neuron_type_str == "AdExp") {
		return add_typed_population<EifCondExpIsfaIsta>();
	}

	throw CypressException("Invalid neuron type \"" + neuron_type_str + "\"");
}

SpikingBinam &SpikingBinam::build()
{
	m_pop_source =
	    m_net.create_population<SpikeSourceArray>(m_dataParams.bits_in());

	auto input_spike_times = build_spike_times();
	for (size_t i = 0; i < m_pop_source.size(); i++) {
		m_pop_source[i].parameters().spike_times(input_spike_times[i]);
	}

	m_pop_output = add_population(m_neuronType);

	const auto &mat = m_BiNAM_Container.trained_matrix();
	m_pop_source.connect_to(
	    m_pop_output, Connector::functor([&](NeuronIndex src, NeuronIndex tar) {
		    return mat.get_bit(tar, src);
		}, m_networkParams.weight()));
	return *this;
}

void SpikingBinam::run(std::string backend) { m_net.run(PyNN(backend)); }

BinaryMatrix<uint64_t> SpikingBinam::spikes_to_matrix()
{
	BinaryMatrix<uint64_t> res(m_dataParams.samples(), m_dataParams.bits_out());
	size_t counter = 0;
	for (auto neuron : m_pop_output) {
		auto spikes = neuron.signals().data(
		    0);  // Get the data for signal zero (the spikes)
		res.write_col_vec(counter,
		                  spikes_to_vector(spikes, m_dataParams.samples(),
		                                   m_networkParams));  // neuron.nid()
		counter++;
	}
	return res;
}

void SpikingBinam::evaluate_neat(std::ostream &out)
{
	BinaryMatrix<uint64_t> output = spikes_to_matrix();
	auto res_spike = m_BiNAM_Container.analysis(output);
	auto res_theo = m_BiNAM_Container.analysis();
	out << "Result of the analysis" << std::endl;
	out << "\tInfo \t nInfo \t fp \t fn" << std::endl;
	out << "theor: \t" << res_theo.Info << "\t" << 1.00 << "\t" << res_theo.fp
	    << "\t" << res_theo.fn << std::endl;
	out << "exp: \t" << res_spike.Info << "\t" << res_spike.Info / res_theo.Info
	    << "\t" << res_spike.fp << "\t" << res_spike.fn << std::endl;
}
void SpikingBinam::evaluate_csv(std::ostream &out)
{

	BinaryMatrix<uint64_t> output = spikes_to_matrix();
	auto res_spike = m_BiNAM_Container.analysis(output);
	auto res_theo = m_BiNAM_Container.analysis();
	out << res_spike.Info << "," << res_theo.Info << "," << res_spike.fp << ","
	    << res_theo.fp << "," << res_spike.fn << "," << res_theo.fn << ",";
}
}
