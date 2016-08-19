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

#pragma once

#ifndef CPPNAM_UTIL_REC_BINAM_HPP
#define CPPNAM_UTIL_REC_BINAM_HPP

#include "core/binam.hpp"
#include "core/parameters.hpp"
#include "core/spiking_parameters.hpp"

namespace nam {

class rec_binam {
public:
	BiNAM<uint64_t> m_BiNAM, m_BiNAM_rec;
	DataParameters m_params;
	DataGenerationParameters m_datagen;
	BinaryMatrix<uint64_t> m_input, m_output, m_recall, m_recall_rec;
	// std::vector<SampleError> m_SampleError;

public:
	/**
	 * Constructor of the Container. Sets all parameters needed for ongoing
	 * Calculation. Therefore, one representation of the BiNAM_Container class
	 * corresponds exactly to one realisation of associative memory.
	 * @param params contains the BiNAM parameters like network size, number of
	 * samples,...
	 * @param seed for the random number generator which produces the data. Can
	 * be used to generate exactly the same data in consecutive runs.
	 * @param random: flag which (de)activates the randomization of data
	 * @param balanced: activates the algorithm for balanced data generation
	 * @param unique: Suppresses the multiple generation of the same pattern
	 */
	rec_binam(DataParameters params, DataGenerationParameters datagen)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_BiNAM_rec(params.bits_out(), params.bits_out()),
	      m_params(params),
	      m_datagen(datagen){};
	rec_binam(DataParameters params)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_BiNAM_rec(params.bits_out(), params.bits_out()),
	      m_params(params),
	      m_datagen(){};
	rec_binam(){};

	rec_binam &set_up(bool train_res = true)
	{
		size_t seed =
		    m_datagen.seed() ? m_datagen.seed() : std::random_device()();

		std::thread input_thread([this, seed]() mutable {
			m_input =
			    DataGenerator(seed, m_datagen.random(), m_datagen.balanced(),
			                  m_datagen.unique())
			        .generate<uint64_t>(m_params.bits_in(), m_params.ones_in(),
			                            m_params.samples());
		});
		std::thread output_thread([this, seed]() mutable {
			m_output = DataGenerator(seed + 5, m_datagen.random(),
			                         m_datagen.balanced(), m_datagen.unique())
			               .generate<uint64_t>(m_params.bits_out(),
			                                   m_params.ones_out(),
			                                   m_params.samples());
		});

		input_thread.join();
		output_thread.join();

		m_BiNAM.train_mat(m_input, m_output);
		m_recall = m_BiNAM.recallMat(m_input);
		if (train_res) {
			m_BiNAM_rec.train_mat(m_recall, m_output);
		}
		else {
			m_BiNAM_rec.train_mat(m_output, m_output);
		}
		m_recall_rec = m_BiNAM_rec.recallMat(m_recall);
		return *this;
	};

	/**
	 * Calculate the false positives and negatives as well as the stored
	 * information.
	 * @param recall_matrix: recalled matrix from experiment. If nothing is
	 * given, it takes the member recall matrix.
	 */
	ExpResults analysis(
	    const BinaryMatrix<uint64_t> &recall_matrix = BinaryMatrix<uint64_t>())
	{
		auto recall_mat = &recall_matrix;
		if (recall_matrix.size() == 0) {
			recall_mat = &m_recall_rec;
		}
		std::vector<SampleError> se =
		    m_BiNAM.false_bits_mat(m_output, *recall_mat);
		double info = entropy_hetero(m_params, se);
		SampleError sum = BiNAM_Container<uint64_t>::sum_false_bits(se);
		return ExpResults(info, sum);
	};

	/**
	 * Getter for member matrices
	 */
	const BiNAM<uint64_t> &trained_matrix() const { return m_BiNAM; };
	const BiNAM<uint64_t> &trained_matrix_rec() const { return m_BiNAM_rec; };
	const BinaryMatrix<uint64_t> &input_matrix() const { return m_input; };
	const BinaryMatrix<uint64_t> &output_matrix() const { return m_output; };
	const BinaryMatrix<uint64_t> &recall_matrix() const { return m_recall; };
	const BinaryMatrix<uint64_t> &recall_matrix_rec() const
	{
		return m_recall_rec;
	};

	void trained_matrix(BiNAM<uint64_t> mat) { m_BiNAM = mat; };
	void trained_matrix_rec(BiNAM<uint64_t> mat) { m_BiNAM_rec = mat; };
	void input_matrix(BinaryMatrix<uint64_t> mat) { m_input = mat; };
	void output_matrix(BinaryMatrix<uint64_t> mat) { m_output = mat; };
	void recall_matrix(BinaryMatrix<uint64_t> mat) { m_recall = mat; };
	void recall_matrix_rec(BinaryMatrix<uint64_t> mat) { m_recall_rec = mat; };
};
}
#endif /* CPPNAM_UTIL_REC_BINAM_HPP */