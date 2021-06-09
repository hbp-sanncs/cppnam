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
#include <cypress/cypress.hpp>

#include <cypress/backend/power/netio4.hpp>

#include "core/spiking_utils.hpp"
#include "rec_binam.hpp"
#include "spiking_rec_binam.hpp"

namespace nam {
using namespace cypress;

SpikingRecBinam::SpikingRecBinam(Json &json, std::ostream &out, bool recall)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_pop_control(m_net, 0),
      m_dataParams(json["data"]),
      m_networkParams(json["network"], out)
{
	m_dataParams.print(out);
	m_recBinam = std::make_shared<RecBinam>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	m_recBinam->set_up();
	if (recall) {
		m_recBinam->recall();
	}
}

SpikingRecBinam::SpikingRecBinam(cypress::Json &json, DataParameters params,
                                 std::ostream &out, bool recall, bool read)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_pop_control(m_net, 0),
      m_dataParams(params)
{
	m_dataParams.print(out);
	m_recBinam = std::make_shared<RecBinam>(
	    m_dataParams, DataGenerationParameters(json["data_generator"]));
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out);
	if (read) {
		m_recBinam->set_up_from_file();
	}
	else {
		m_recBinam->set_up();
	}
	if (recall) {
		m_recBinam->recall();
	}
}

SpikingRecBinam::SpikingRecBinam(cypress::Json &json, DataParameters params,
                                 DataGenerationParameters gen_params,
                                 std::ostream &out, bool, bool warn)
    : m_pop_source(cypress::PopulationBase(m_net, 0)),
      m_pop_output(m_net, 0),
      m_pop_control(m_net, 0),
      m_dataParams(params)
{
	m_dataParams.print(out);
	m_recBinam = std::make_shared<RecBinam>(m_dataParams, gen_params);
	m_neuronType = json["network"]["neuron_type"];
	const auto &neuronType = SpikingUtils::detect_type(m_neuronType);

	m_neuronParams = NeuronParameters(neuronType, json["network"], out, warn);
	m_networkParams = NetworkParameters(json["network"], out, warn);
	m_recBinam->set_up(false, true);
}

SpikingRecBinam &SpikingRecBinam::build(cypress::Network &network)
{
	size_t multi = m_networkParams.multiplicity();
	m_pop_source = network.create_population<SpikeSourceArray>(
	    m_dataParams.bits_in() * multi);
	auto input_spike_times = SpikingUtils::build_spike_times(
	    m_recBinam->input_matrix(), m_networkParams, 1234);

	for (size_t i = 0; i < m_dataParams.bits_in(); i++) {
		for (size_t j = 0; j < multi; j++) {
			m_pop_source[i * multi + j].parameters().spike_times(
			    input_spike_times[i * multi + j]);
		}
	}

	m_pop_output = SpikingUtils::add_population(
	    m_neuronType, network, m_dataParams, m_networkParams, m_neuronParams);
	m_pop_output.signals().record(1, true);

	m_pop_control = SpikingUtils::add_population(
	    m_neuronType, network, m_dataParams, m_networkParams, m_neuronParams);
	m_pop_control.signals().record(1, true);

	const auto &mat = m_recBinam->trained_matrix();
	const auto &matRec = m_recBinam->trained_matrix_rec();

	size_t mult = m_networkParams.multiplicity();
	m_pop_source.connect_to(m_pop_output,
	                        cypress::Connector::functor(
	                            [&, mult](NeuronIndex src, NeuronIndex tar) {
		                            return mat.get_bit(
		                                std::floor(float(tar) / float(mult)),
		                                std::floor(float(src) / float(mult)));
		                        },
	                            m_networkParams.weight()));

	m_pop_output.connect_to(m_pop_output,
	                        cypress::Connector::functor(
	                            [&, mult](NeuronIndex src, NeuronIndex tar) {
		                            return matRec.get_bit(
		                                std::floor(float(tar) / float(mult)),
		                                std::floor(float(src) / float(mult)));
		                        },
	                            m_networkParams.weight_rec(),
								m_networkParams.delay_rec()));

	m_pop_output.connect_to(m_pop_control, cypress::Connector::all_to_all(
	                                           m_networkParams.weight_inhib(),
	                                           m_networkParams.delay_inhib()));

	m_pop_control.connect_to(m_pop_output, cypress::Connector::all_to_all(
										m_networkParams.weight_control(),
	                                    m_networkParams.delay_control()));
	return *this;
}

void SpikingRecBinam::run(std::string backend)
{
	m_net.run(cypress::PyNN(backend));
}

Real SpikingRecBinam::spikes_to_recurrency_rate(PopulationBase &popOutput,
                                                DataParameters &dataParams,
                                                NetworkParameters &netwParams,
                                                BinaryMatrix<uint64_t> output)
{
	std::pair<Real, Real> rr;
	auto res_spike = m_recBinam->analysis(output);
	size_t multi = netwParams.multiplicity();
	for (size_t i = 0; i < dataParams.bits_out(); i++) {
		Vector<uint8_t> spike_vec(dataParams.samples(), MatrixFlags::ZEROS);
		for (size_t j = 0; j < multi; j++) {
			auto spikes = popOutput[i * multi + j].signals().data(0);
			auto temp_vec = SpikingUtils::spikes_to_vector(
			    spikes, dataParams.samples(), netwParams);

			for (size_t k = 0; k < temp_vec.size(); k++) {
				// printf("temp_vec[%ld]: %d \n", k, temp_vec[k]);
				spike_vec[k] += temp_vec[k];
				if (spike_vec[k] > 1 &&
				    !(m_recBinam->m_output.get_bit(k, i) == 0 &&
				      output.get_bit(k, i) == 1)) {
					// printf("spike_vec[%ld]: %d\n", k, spike_vec[k]);
					rr.first++;
				}
				if (spike_vec[k] >= 1 
					/* && !(m_recBinam->m_output.get_bit(i, k) == 0 && output.get_bit(i, k) == 1) */) {
					// printf("spike_vec[%ld]: %d\n", k, spike_vec[k]);
					rr.second++;
				}
			}
		}
	}
	// std::cout << rr.first << std::endl << rr.second-res_spike.fp <<
	// std::endl;
	return rr.first / (rr.second - res_spike.fp);
}

void SpikingRecBinam::evaluate_neat(std::ostream &out)
{
	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	Real rec_rate = spikes_to_recurrency_rate(m_pop_output, m_dataParams,
	                                          m_networkParams, output);
	auto res_spike = m_recBinam->analysis(output);
	auto res_theo = m_recBinam->analysis();
	auto res_theo_non_rec = m_recBinam->analysis(m_recBinam->recall_matrix());
	out << "Result of the analysis" << std::endl;
	out << "\t\t\tInfo \t\tnInfo \t\tfp \t\tfn" << std::endl;
	out << "theo: \t\t" << res_theo_non_rec.Info << "\t\t" << 100 << "%\t\t"
	    << res_theo_non_rec.fp << "\t\t" << res_theo_non_rec.fn << std::endl;
	out << "rec: \t\t" << res_theo.Info << "\t\t" << 100 << "%\t\t"
	    << res_theo.fp << "\t\t" << res_theo.fn << std::endl;
	out << "exp: \t\t" << res_spike.Info << "\t\t"
	    << (res_spike.Info / res_theo.Info) * 100 << "%\t" << res_spike.fp
	    << "\t\t" << res_spike.fn << std::endl;
	out << "recurrency_rate: \t\t" << rec_rate << std::endl;
	// plot_spikes(m_pop_output);
}

void SpikingRecBinam::plot_spikes(PopulationBase &popOutput)
{
	size_t s = popOutput.size();
	std::vector<std::vector<Real>> spks;
	Real time_limit = 0;
	for (size_t i = 0; i < s; i++) {
		spks.push_back(popOutput[i].signals().data(0));
		if (spks.at(i).size() > 0) {
			for (size_t j = 0; j < spks.at(i).size(); j++) {
				time_limit = spks.at(i).at(j);
			}
		}
	}

	pyplot::figure();
	pyplot::eventplot(spks);
	pyplot::title("Spike Times");
	pyplot::xlabel("Time in ms");
	pyplot::ylabel("Neuron ID");
	pyplot::xlim(0, int(time_limit + 500));
	if (spks.size() > 1) {
		pyplot::ylim(-0.5, spks.size() - 0.5);
	}
	else {
		pyplot::ylim(0.5, 1.5);
	}
	pyplot::tight_layout();
	pyplot::show();
	pyplot::save("spikes.pdf");  // or.png
}

void SpikingRecBinam::evaluate_csv(std::ostream &out)
{

	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	ExpResults res_spike = m_recBinam->analysis(output);
	ExpResults res_theo = m_recBinam->analysis();
	out << res_spike.Info << "," << res_theo.Info << ","
	    << res_spike.Info / res_theo.Info << "," << res_spike.fp << ","
	    << res_theo.fp << "," << res_spike.fn << "," << res_theo.fn;
}
std::pair<ExpResults, ExpResults> SpikingRecBinam::evaluate_res()
{
	BinaryMatrix<uint64_t> output = SpikingUtils::spikes_to_matrix(
	    m_pop_output, m_dataParams, m_networkParams);
	Real rec_rate = spikes_to_recurrency_rate(m_pop_output, m_dataParams,
	                                          m_networkParams, output);
	ExpResults res_spike = m_recBinam->analysis(output);
	res_spike.rr = rec_rate;
	ExpResults res_theo = m_recBinam->analysis();
	return std::pair<ExpResults, ExpResults>(res_theo, res_spike);
}
}
