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

#include <string>
#include <cypress/backend/power/netio4.hpp>

#include "entropy.hpp"
#include "parameters.hpp"
#include "spiking_binam.hpp"
#include "spiking_utils.hpp"
#include "util/binary_matrix.hpp"

using namespace cypress;

namespace nam {

SpikingBinam::SpikingBinam(Json &json, std::ostream &out, bool recall)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_dataParams(json["data"]),
      m_networkParams(json["network"], out)
{
	m_dataParams.print(out);
	m_BiNAM_Container = std::make_shared<BiNAM_Container<uint64_t>>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	m_BiNAM_Container->set_up();
	if (recall) {
		m_BiNAM_Container->recall();
	}
}

SpikingBinam::SpikingBinam(Json &json, DataParameters params, std::ostream &out,
                           bool recall, bool read)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_dataParams(params),
      m_networkParams(json["network"], out)
{
	m_dataParams.print(out);
	m_BiNAM_Container = std::make_shared<BiNAM_Container<uint64_t>>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	if (read) {
		m_BiNAM_Container->set_up_from_file();
	}
	else {
		m_BiNAM_Container->set_up();
	}
	if (recall) {
		m_BiNAM_Container->recall();
	}
}

SpikingBinam::SpikingBinam(Json &json, DataParameters params,
                           DataGenerationParameters gen_params,
                           std::ostream &out, bool recall, bool warn)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_dataParams(params)
{
	m_dataParams.print(out);
	m_BiNAM_Container =
	    std::make_shared<BiNAM_Container<uint64_t>>(m_dataParams, gen_params);
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out, warn);
	m_networkParams = NetworkParameters(json["network"], out, warn);
	m_BiNAM_Container->set_up();
	if (recall) {
		m_BiNAM_Container->recall();
	}
}

SpikingBinam &SpikingBinam::build()
{
	size_t multi = m_networkParams.multiplicity();
	m_pop_source = m_net.create_population<SpikeSourceArray>(
	    m_dataParams.bits_in() * multi);

	auto input_spike_times = SpikingUtils::build_spike_times(
	    m_BiNAM_Container->input_matrix(), m_networkParams, 1234);
	for (size_t i = 0; i < m_dataParams.bits_in(); i++) {
		for (size_t j = 0; j < multi; j++) {
			m_pop_source[i * multi + j].parameters().spike_times(
			    input_spike_times[i * multi + j]);
		}
	}

	m_pop_output = SpikingUtils::add_population(
	    m_neuronType, m_net, m_dataParams, m_networkParams, m_neuronParams);

	const auto &mat = m_BiNAM_Container->trained_matrix();

	m_pop_source.connect_to(
	    m_pop_output, Connector::functor(
	                      [&](NeuronIndex src, NeuronIndex tar) {
		                      size_t mult = m_networkParams.multiplicity();
		                      return mat.get_bit(
		                          std::floor(double(tar) / double(mult)),
		                          std::floor(double(src) / double(mult)));
		                  },
	                      m_networkParams.weight()));
	return *this;
}

SpikingBinam &SpikingBinam::build(cypress::Network &network)
{
	size_t multi = m_networkParams.multiplicity();
	m_pop_source = network.create_population<SpikeSourceArray>(
	    m_dataParams.bits_in() * multi);
	auto input_spike_times = SpikingUtils::build_spike_times(
	    m_BiNAM_Container->input_matrix(), m_networkParams, 1234);

	for (size_t i = 0; i < m_dataParams.bits_in(); i++) {
		for (size_t j = 0; j < multi; j++) {
			m_pop_source[i * multi + j].parameters().spike_times(
			    input_spike_times[i * multi + j]);
		}
	}

	m_pop_output = SpikingUtils::add_population(
	    m_neuronType, network, m_dataParams, m_networkParams, m_neuronParams);

	const auto &mat = m_BiNAM_Container->trained_matrix();

	size_t mult = m_networkParams.multiplicity();
	m_pop_source.connect_to(m_pop_output,
	                        Connector::functor(
	                            [&, mult](NeuronIndex src, NeuronIndex tar) {
		                            return mat.get_bit(
		                                std::floor(float(tar) / float(mult)),
		                                std::floor(float(src) / float(mult)));
		                        },
	                            m_networkParams.weight()));
	return *this;
}
void SpikingBinam::run(std::string backend) { m_net.run(backend); }
/*void SpikingBinam::run(std::string backend, cypress::Network &netw)
{
	cypress::PowerManagementBackend pwbackend(
	    std::make_shared<cypress::NetIO4>(),
	    cypress::Network::make_backend(backend));
	netw.run(pwbackend);
}*/

void SpikingBinam::evaluate_neat(std::ostream &out)
{
	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	auto res_spike = m_BiNAM_Container->analysis(output);
	auto res_theo = m_BiNAM_Container->analysis();
	out << "Result of the analysis" << std::endl;
	out << "\tInfo \t nInfo \t fp \t fn" << std::endl;
	out << "theor: \t" << res_theo.Info << "\t" << 1.00 << "\t" << res_theo.fp
	    << "\t" << res_theo.fn << std::endl;
	out << "exp: \t" << res_spike.Info << "\t" << res_spike.Info / res_theo.Info
	    << "\t" << res_spike.fp << "\t" << res_spike.fn << std::endl;
}
void SpikingBinam::evaluate_csv(std::ostream &out)
{

	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	ExpResults res_spike = m_BiNAM_Container->analysis(output);
	ExpResults res_theo = m_BiNAM_Container->analysis();
	out << res_spike.Info << "," << res_theo.Info << ","
	    << res_spike.Info / res_theo.Info << "," << res_spike.fp << ","
	    << res_theo.fp << "," << res_spike.fn << "," << res_theo.fn;
}
std::pair<ExpResults, ExpResults> SpikingBinam::evaluate_res()
{
	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	ExpResults res_spike = m_BiNAM_Container->analysis(output);
	ExpResults res_theo = m_BiNAM_Container->analysis();
	return std::pair<ExpResults, ExpResults>(res_theo, res_spike);
}
}
