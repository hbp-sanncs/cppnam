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

#ifndef CPPNAM_RECURRENT_REC_BINAM_HPP
#define CPPNAM_RECURRENT_REC_BINAM_HPP

#include <cypress/cypress.hpp>
#include "core/binam.hpp"
#include "core/parameters.hpp"
#include "core/spiking_parameters.hpp"

namespace nam {

class RecBinam {
public:
	BiNAM<uint64_t> m_binam, m_binam_rec;
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
	 * @param datagen for the data generation
	 */
	RecBinam(DataParameters params, DataGenerationParameters datagen)
	    : m_binam(params.bits_out(), params.bits_in()),
	      m_binam_rec(params.bits_out(), params.bits_out()),
	      m_params(params),
	      m_datagen(datagen){};
	RecBinam(DataParameters params)
	    : m_binam(params.bits_out(), params.bits_in()),
	      m_binam_rec(params.bits_out(), params.bits_out()),
	      m_params(params),
	      m_datagen(){};
	RecBinam(){};

	RecBinam &set_up(bool train_res = true, bool recall = true);
	
	RecBinam &set_up_from_file(bool train_res=true);

	/**
	 * Recalls the patterns with the input matrix
	 */
	RecBinam &recall();

	/**
	 * Calculate the false positives and negatives as well as the stored
	 * information.
	 * @param recall_matrix: recalled matrix from experiment. If nothing is
	 * given, it takes the member recall matrix.
	 */
	ExpResults analysis(
	    const BinaryMatrix<uint64_t> &recall_matrix = BinaryMatrix<uint64_t>());

	/**
	 * Getter for member matrices
	 */
	const BiNAM<uint64_t> &trained_matrix() const { return m_binam; };
	const BiNAM<uint64_t> &trained_matrix_rec() const { return m_binam_rec; };
	const BinaryMatrix<uint64_t> &input_matrix() const { return m_input; };
	const BinaryMatrix<uint64_t> &output_matrix() const { return m_output; };
	const BinaryMatrix<uint64_t> &recall_matrix() const { return m_recall; };
	const BinaryMatrix<uint64_t> &recall_matrix_rec() const
	{
		return m_recall_rec;
	};

	void trained_matrix(BiNAM<uint64_t> mat) { m_binam = mat; };
	void trained_matrix_rec(BiNAM<uint64_t> mat) { m_binam_rec = mat; };
	void input_matrix(BinaryMatrix<uint64_t> mat) { m_input = mat; };
	void output_matrix(BinaryMatrix<uint64_t> mat) { m_output = mat; };
	void recall_matrix(BinaryMatrix<uint64_t> mat) { m_recall = mat; };
	void recall_matrix_rec(BinaryMatrix<uint64_t> mat) { m_recall_rec = mat; };
};
}
#endif /* CPPNAM_RECURRENT_REC_BINAM_HPP */