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

#include <algorithm>
#include <map>
#include <limits>
#include <random>

#include "data.hpp"
#include "ncr.hpp"

namespace nam {
namespace {

class PermutationTrieNode {
private:
	int m_idx;
	int m_remaining;
	uint64_t m_total;
	Vector<uint32_t> m_max_permutations;
	Vector<uint32_t> m_permutations;
	std::map<int, PermutationTrieNode> m_children;
	static constexpr uint32_t MAX_PERMS = std::numeric_limits<uint32_t>::max();

public:
	PermutationTrieNode(int idx, int remaining)
	    : m_idx(idx), m_remaining(remaining)
	{
		// Binary search for the point at which ncr(n, rem - 1) = MAX_PERMS
		int min = 0;
		int max = m_idx - 1;
		while (max - min > 10) {  // Accuracy is not imporant here
			uint32_t n = (min + max) / 2;
			uint32_t res = ncr_clamped32(n, remaining - 1);
			if (res < MAX_PERMS) {
				min = n;
			}
			else {
				max = n;
			}
		}

		// Reserve as many entries as estimated above and fill them
		m_max_permutations = Vector<uint32_t>(max + 1);
		for (int i = 0; i <= max; i++) {
			if (i > 0 && m_max_permutations[i - 1] == MAX_PERMS) {
				m_max_permutations(i) = MAX_PERMS;
			}
			else {
				m_max_permutations(i) = ncr_clamped32(i, remaining - 1);
			}
		}
		m_permutations = m_max_permutations;
		m_total = std::accumulate<uint32_t *, uint64_t>(
		    m_max_permutations.begin(), m_max_permutations.end(), uint64_t(0));
	}

	int idx() const { return m_idx; }
	int remaining() const { return m_remaining; }
	int total() const { return m_total; }
	PermutationTrieNode &fetch(int idx)
	{
		auto it = m_children.find(idx);
		if (it != m_children.end()) {
			return it->second;
		}
		return m_children.emplace(idx,
		                          PermutationTrieNode({idx, m_remaining - 1}))
		    .first->second;
	}

	bool has_permutation(int idx)
	{
		return idx >= int(m_permutations.size()) ? true
		                                         : m_permutations[idx] > 0;
	}

	uint32_t permutation_count(int idx)
	{
		if (idx >= int(m_permutations.size())) {
			return MAX_PERMS;
		}
		return m_permutations[idx];
	}

	bool decrement_permutation(int idx)
	{
		if (idx >= int(m_permutations.size())) {
			return true;
		}

		if (m_permutations[idx] == MAX_PERMS) {
			return true;
		}

		if (m_total > 1) {
			m_permutations(idx)--;
			m_total--;
			return true;
		}

		m_permutations = m_max_permutations;
		m_total = std::accumulate<uint32_t *, uint64_t>(
		    m_max_permutations.begin(), m_max_permutations.end(), 0);
		return false;
	}
};

template <typename RandomEngine>
Matrix<uint8_t> generate_balanced(
    RandomEngine &re, uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    bool random, bool balance, bool unique,
    const DataGenerator::ProgressCallback &progress)
{
	Matrix<uint8_t> res(n_samples, n_bits,
	                    MatrixFlags::ZEROS);  // Result matrix
	Vector<uint32_t> usage(
	    n_bits,
	    MatrixFlags::ZEROS);  // Vector tracking how often each bit is used
	Vector<uint32_t> allowed(n_bits,
	                         MatrixFlags::ZEROS);  // Temporary vector holding
	                                               // the number of ones which
	                                               // can be inserted
	Vector<uint8_t> balancable(n_bits, MatrixFlags::ZEROS);
	Vector<uint8_t> selected(n_bits, MatrixFlags::ZEROS);
	Vector<double> weights(n_bits, MatrixFlags::ZEROS);

	PermutationTrieNode root(n_bits, n_ones);
	for (size_t i = 0; i < n_samples; i++) {
		PermutationTrieNode *node = &root;
		for (size_t j = 0; j < n_ones; j++) {
			const size_t idx = node->idx();

			// Select those elements for which a permutation is left
			uint32_t min_usage = std::numeric_limits<uint32_t>::max();
			for (size_t k = 0; k < idx; k++) {
				selected(k) = node->has_permutation(k);
				if (selected[k]) {
					min_usage = std::min(min_usage, usage[k]);
				}
			}

			// Try to balance the bit usage
			if (balance) {
				// Check which indices are minimally used, only select those
				// which are
				for (size_t k = 0; k < idx; k++) {
					balancable[k] = (usage[k] == min_usage) ? 1 : 0;
					selected[k] = selected[k] && balancable[k];
				}

				// Calculate how many balanced insertions are still allowed
				// after an index has been inserted at a given position
				uint32_t max_allowed = 0;
				uint32_t cum_balancable = 0;
				for (size_t k = 0; k < idx; k++) {
					cum_balancable = cum_balancable + balancable[k];
					allowed(k) = std::min<uint32_t>(n_ones - j, cum_balancable);
					max_allowed = std::max(max_allowed, allowed[k]);
				}

				// Try to select those indices for which the following
				// insertions of ones still allow balancing
				bool has_best = false;
				for (size_t k = 0; k < idx; k++) {
					balancable[k] =
					    (allowed[k] == max_allowed) && selected[k] ? 1 : 0;
					has_best = has_best || balancable[k];
				}
				if (has_best) {
					for (size_t k = 0; k < idx; k++) {
						selected[k] = balancable[k];
					}
				}
			}

			size_t chosen_idx = 0;
			if (random) {
				// Weight the entries with the possible permutations the
				// corresponding path still can generate. Calculate the
				// probabilities (weights) with which the indices are selected
				size_t total = 0;
				for (size_t k = 0; k < idx; k++) {
					weights[k] = selected[k] ? node->permutation_count(k) : 0;
					total += weights[k];
				}

				// Normalise the weights
				for (size_t k = 0; k < idx; k++) {
					weights[k] = weights[k] / (double)(total);
				}

				// Select the indices
				const double rnd =
				    std::uniform_real_distribution<double>(0, 1)(re);
				double w_sum = 0;
				for (size_t k = 0; k < idx; k++) {
					w_sum += weights[k];
					if (w_sum >= rnd) {
						chosen_idx = k;
						break;
					}
				}
			}
			else {
				chosen_idx = idx - 1;
				while (not selected[chosen_idx]) {
					chosen_idx--;
				}
			}

			// Set the corresponding output bit to one and update the trie
			res(i, chosen_idx) = 1;
			usage[chosen_idx]++;
			if (unique) {
				node->decrement_permutation(chosen_idx);
			}
			node = &node->fetch(chosen_idx);
		}

		// Regularly call the progress function
		if ((i == 0) || (i == n_samples - 1) || (i % 100 == 0)) {
			if (!progress(float(i) / float(n_samples - 1))) {
				break;
			}
		}
	}

	return res;
}

template <typename RandomEngine>
Matrix<uint8_t> generate_random(RandomEngine &re, size_t n_bits, size_t n_ones,
                                size_t n_samples,
                                const DataGenerator::ProgressCallback &progress)
{
	Matrix<uint8_t> res(n_samples, n_bits, MatrixFlags::ZEROS);
	for (size_t i = 0; i < n_samples; i++) {
		for (size_t j = n_bits - n_ones; j < n_bits; j++) {
			size_t idx = std::uniform_int_distribution<size_t>(0, j)(re);
			if (res(i, idx) == 1) {
				res(i, j) = 1;
			}
			else {
				res(i, idx) = 1;
			}
		}

		// Regularly call the progress function
		if ((i == 0) || (i == n_samples - 1) || (i % 100 == 0)) {
			if (!progress(float(i) / float(n_samples - 1))) {
				break;
			}
		}
	}
	return res;
}
}

DataGenerator::DataGenerator(bool random, bool balance, bool unique)
    : m_seed(std::random_device()()),
      m_random(random),
      m_balance(balance),
      m_unique(unique)
{
}

DataGenerator::DataGenerator(size_t seed, bool random, bool balance,
                             bool unique)
    : m_seed(seed), m_random(random), m_balance(balance), m_unique(unique)
{
}

Matrix<uint8_t> DataGenerator::generate(
    uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    const DataGenerator::ProgressCallback &progress)
{
	std::default_random_engine re(m_seed);
	if (m_random && !m_balance && !m_unique) {
		return generate_random(re, n_bits, n_ones, n_samples, progress);
	}
	else {
		return generate_balanced(re, n_bits, n_ones, n_samples, m_random,
		                         m_balance, m_unique, progress);
	}
}
}
