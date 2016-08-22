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

#include "rec_binam.hpp"

namespace nam {
RecBinam &RecBinam::set_up(bool train_res, bool recall)
{
	size_t seed = m_datagen.seed() ? m_datagen.seed() : std::random_device()();

	std::thread input_thread([this, seed]() mutable {
		m_input =
		    DataGenerator(seed, m_datagen.random(), m_datagen.balanced(),
		                  m_datagen.unique())
		        .generate<uint64_t>(m_params.bits_in(), m_params.ones_in(),
		                            m_params.samples());
	});
	std::thread output_thread([this, seed]() mutable {
		m_output =
		    DataGenerator(seed + 5, m_datagen.random(), m_datagen.balanced(),
		                  m_datagen.unique())
		        .generate<uint64_t>(m_params.bits_out(), m_params.ones_out(),
		                            m_params.samples());
	});

	input_thread.join();
	output_thread.join();

	m_binam.train_mat(m_input, m_output);
	m_recall = m_binam.recallMat(m_input);
	if (train_res) {
		m_binam_rec.train_mat(m_recall, m_output);
	}
	else {
		m_binam_rec.train_mat(m_output, m_output);
	}
	if (recall) {
		m_recall_rec = m_binam_rec.recallMat(m_recall);
	}
	return *this;
}

RecBinam &RecBinam::recall()
{
	m_recall_rec = m_binam_rec.recallMat(m_recall);
	return *this;
}

ExpResults RecBinam::analysis(const BinaryMatrix<uint64_t> &recall_matrix)
{
	auto recall_mat = &recall_matrix;
	if (recall_matrix.size() == 0) {
		recall_mat = &m_recall_rec;
	}
	std::vector<SampleError> se = m_binam.false_bits_mat(m_output, *recall_mat);
	double info = entropy_hetero(m_params, se);
	SampleError sum = BiNAM_Container<uint64_t>::sum_false_bits(se);
	return ExpResults(info, sum);
}

RecBinam &RecBinam::set_up_from_file(bool train_res)
{
	size_t height;
	size_t width;
	std::fstream ss("../data/data_in", std::fstream::in);
	ss.read((char *)&width, sizeof(width));
	ss.read((char *)&height, sizeof(height));
	m_input = BinaryMatrix<uint64_t>(height, width);
	ss.read((char *)m_input.cells().data(),
	        m_input.cells().size() * sizeof(uint64_t));
	ss.close();
	if (m_input.cols() != m_params.bits_in() ||
	    m_input.rows() != m_params.samples()) {
		std::stringstream s;
		s << "Input data size " << m_input.cols() << " and " << m_input.rows()
		  << " differs from given Parameters " << m_params.bits_in() << " and "
		  << m_params.samples() << " !" << std::endl;
		throw std::out_of_range(s.str());
	}

	ss.open("../data/data_out", std::fstream::in);
	ss.read((char *)&width, sizeof(width));
	ss.read((char *)&height, sizeof(height));
	m_output = BinaryMatrix<uint64_t>(height, width);
	ss.read((char *)m_output.cells().data(),
	        m_output.cells().size() * sizeof(uint64_t));
	ss.close();
	if (m_output.cols() != m_params.bits_out() ||
	    m_output.rows() != m_params.samples()) {
		std::stringstream s;
		s << "Output data size " << m_output.cols() << " and "
		  << m_output.rows() << " differs from given Parameters "
		  << m_params.bits_out() << " and " << m_params.samples() << " !"
		  << std::endl;
		throw std::out_of_range(s.str());
	}
	m_binam.train_mat(m_input, m_output);
	m_recall = m_binam.recallMat(m_input);
	if (train_res) {
		m_binam_rec.train_mat(m_recall, m_output);
	}
	else {
		m_binam_rec.train_mat(m_output, m_output);
	}
	m_recall_rec = m_binam_rec.recallMat(m_recall);
	return *this;
}
}
