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

#include "recurrent/rec_binam.hpp"

#include <cstring>
#include <iomanip>
#include <thread>

#include "core/binam.hpp"
#include "core/entropy.hpp"

using namespace nam;

void show_progress(float progress)
{
	const int WIDTH = 50;
	float perc = progress * 100.0;
	std::cerr << std::setw(8) << std::setprecision(4) << perc << "% [";
	for (int i = 0; i < WIDTH; i++) {
		bool cur = i * 100 / WIDTH < perc;
		bool prev = (i - 1) * 100 / WIDTH < perc;
		if (cur && prev) {
			std::cerr << "=";
		}
		else if (prev) {
			std::cerr << ">";
		}
		else {
			std::cerr << " ";
		}
	}
	std::cerr << "]   \r";
}

void information_graph(size_t bits_in, size_t bits_out, size_t ones_in,
                       size_t ones_out, size_t max_sample, bool rec = false)
{
	std::ofstream file;
	file.open("binam_data.csv", std::ios::out);
	file << "Samples, info, fp";
	if (rec) {
		file << ", info_rec, fp_rec, fn_rec";
		file << ", info_rec_opti, fp_rec_opti, fn_rec_opti";
	}
	file << std::endl;
	for (size_t i = 1; i <= max_sample; i++) {
		DataParameters params(bits_in, bits_out, ones_in, ones_out, i);
		BiNAM_Container<uint64_t> binam(params);
		ExpResults se1, se2, se3;
		std::thread normal_binam([&]() mutable {
			binam.set_up().recall();
			se1 = binam.analysis();
		});
		RecBinam binam2, binam3;

		if (rec) {
			binam2 = RecBinam(params);
			binam3 = RecBinam(params);
			std::thread true_rec_binam([&]() mutable {
				binam2.set_up(true);
				se2 = binam2.analysis();
			});
			std::thread false_rec_binam([&]() mutable {
				binam3.set_up(false);
				se3 = binam3.analysis();
			});

			true_rec_binam.join();
			false_rec_binam.join();
		}

		normal_binam.join();

		// double info = entropy_hetero(params, se1);
		file << i << "," << se1.Info << "," << se1.fp;
		if (rec) {
			file << "," << se2.Info << "," << se2.fp << "," << se2.fn;
			file << "," << se3.Info << "," << se3.fp << "," << se3.fn;
		}
		file << std::endl;
		show_progress(double(i) / double(max_sample));
	}
	std::cerr << std::endl;
	file.close();
}
int main(int argc, char *argv[])
{
	if (argc < 4 || argc >= 9) {
		std::cerr
		    << "Usage: ./recurrent_BiNAM <BITS> <ONES> <SAMPLES> (<RECURRENT>)"
		    << std::endl
		    << "or <BITS_IN> <BITS_OUT> <ONES_IN> <ONES_OUT> <MAX_SAMPLES> "
		       "(<RECURRENT>) (sweep)"
		    << std::endl
		    << "or <BITS_IN> <BITS_OUT> <ONES_IN> <ONES_OUT> <SAMPLES> "
		       "(<RECURRENT>)"
		    << std::endl;
		return 1;
	}

	bool rec = false;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "RECURRENT") == 0 ||
		    strcmp(argv[i], "recurrent") == 0) {
			rec = true;
		}
	}
	if ((argc == 7 && !rec) || (argc == 8 && rec)) {
		information_graph(std::stoi(argv[1]), std::stoi(argv[2]),
		                  std::stoi(argv[3]), std::stoi(argv[4]),
		                  std::stoi(argv[5]), rec);
		return 0;
	}
	else {

		int n_bits = std::stoi(argv[1]);
		int n_bits_out = std::stoi(argv[1]);
		int n_ones = std::stoi(argv[2]);
		int n_ones_out = std::stoi(argv[2]);
		int n_samples = std::stoi(argv[3]);
		if (argc > 5) {
			n_bits = std::stoi(argv[1]);
			n_bits_out = std::stoi(argv[2]);
			n_ones = std::stoi(argv[3]);
			n_ones_out = std::stoi(argv[4]);
			n_samples = std::stoi(argv[5]);
		}

		std::cout << n_bits << " bits, " << n_ones << " ones and " << n_samples
		          << " samples" << std::endl;
		DataParameters params(n_bits, n_bits_out, n_ones, n_ones_out,
		                      n_samples);
		if (rec) {
			RecBinam binam(params);
			auto res = binam.set_up(false).analysis();
			res.print();
		}
		else {
			BiNAM_Container<uint64_t> binam(params);
			auto res = binam.set_up().recall().analysis();
			res.print();
		}
	}
	return 0;
}
