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

#include <cppnam/binam/data_generator.hpp>
#include <cppnam/binam/ncr.hpp>

namespace nam {
namespace {
/**
 * Class used internally in the data generator to represent a node in the trie
 * which represents already generated permutations.
 */
class PermutationTrieNode {
public:
	static constexpr uint32_t MAX_PERMS = std::numeric_limits<uint32_t>::max();

	int m_idx;
	int m_remaining;
	int m_min;
	int m_max;
	uint64_t m_total;
	std::vector<uint32_t> m_permutations;
	std::map<int, PermutationTrieNode> m_children;

	/**
	 * Reinitializes the permutation list.
	 */
	void initialize_permutations()
	{
		// Initialize the m_permutations list, calculate
		//                      binom(i, m_remaining - 1)
		// for each i from zero to m_idx - 1.
		m_permutations.clear();
		m_min = 0;
		m_max = m_idx;
		for (int i = 0; i < m_max; i++) {
			if (i < m_remaining - 1 || m_remaining == 0) {
				++m_min;
			}
			else if (i == m_remaining - 1) {
				m_permutations.push_back(1);
			}
			else {
				uint64_t n = (uint64_t(m_permutations.back()) * uint64_t(i)) /
				             uint64_t(i - m_remaining + 1);
				if (n >= MAX_PERMS) {
					m_max = i;
					break;
				}
				m_permutations.push_back(n);
			}
		}
		m_total =
		    std::accumulate<typename std::vector<uint32_t>::iterator, uint64_t>(
		        m_permutations.begin(), m_permutations.end(), uint64_t(0));
	}

public:
	PermutationTrieNode(int idx, int remaining)
	    : m_idx(idx), m_remaining(remaining)
	{
		initialize_permutations();
	}

	int idx() const { return m_idx; }
	int remaining() const { return m_remaining; }
	uint64_t total() const { return m_total; }
	int max() const { return m_max; }
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
		return idx < m_min
		           ? false
		           : (idx >= m_max ? true : m_permutations[idx - m_min] > 0);
	}

	uint32_t permutation_count(int idx)
	{
		return idx < m_min ? 0 : (idx >= m_max ? MAX_PERMS
		                                       : m_permutations[idx - m_min]);
	}

	bool decrement_permutation(int idx)
	{
		// Do not decrement the number of permutations if it equals MAX_PERMS
		if (idx >= m_max) {
			return true;
		}

		if (m_total > 1) {
			if (idx >= m_min) {
				m_permutations[idx - m_min]--;
				m_total--;
			}
			return true;
		}

		initialize_permutations();
		return false;
	}
};
}

template <typename RandomEngine>
BinaryMatrix DataGenerator::generate_random(
    RandomEngine &re, size_t n_bits, size_t n_ones, size_t n_samples,
    const DataGenerator::ProgressCallback &progress)
{
	BinaryMatrix res(n_samples, n_bits);
	for (size_t i = 0; i < n_samples; i++) {
		for (size_t j = n_bits - n_ones; j < n_bits; j++) {
			size_t idx = std::uniform_int_distribution<size_t>(0, j)(re);
			if (res(i, idx)) {
				res(i, j) = true;
			}
			else {
				res(i, idx) = true;
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

template <typename RandomEngine>
BinaryMatrix DataGenerator::generate_balanced(
    RandomEngine &re, uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    bool random, bool balance, bool unique,
    const DataGenerator::ProgressCallback &progress)
{
	auto approximate_weight = [](uint32_t k, uint32_t r_ones,
	                             uint32_t r_bits) -> double {
		//                                (k | r_ones - 1)
		// exact solution for -------------------------------------
		//                     sum((j | r_ones - 1), j=r_ones-1...r_bits-1)
		const uint32_t num_f0 = k - r_ones + 2;
		const uint32_t den_f0 = r_bits - r_ones + 1;
		double res = double(r_ones) / double(r_bits);
		for (int i = 0; i <= int(r_ones) - 2; i++) {
			res *= double(num_f0 + i) / double(den_f0 + i);
		}
		return res;
	};

	BinaryMatrix res(n_samples, n_bits);  // Result matrix
	std::vector<uint32_t> usage(
	    n_bits);  // vector tracking how often each bit is used
	std::vector<uint32_t> allowed(n_bits);  // Temporary vector holding
	                                        // the number of ones which
	                                        // can be inserted
	std::vector<uint8_t> balancable(n_bits);
	std::vector<uint8_t> selected(n_bits);
	std::vector<double> weights(n_bits);

	PermutationTrieNode root(n_bits, n_ones);
	for (size_t i = 0; i < n_samples; i++) {
		size_t ones = 0;
		PermutationTrieNode *node = &root;
		for (size_t j = 0; j < n_ones; j++) {
			const size_t idx = node->idx();

			// Select those elements for which a permutation is left
			uint32_t min_usage = std::numeric_limits<uint32_t>::max();
			for (size_t k = 0; k < idx; k++) {
				selected[k] = node->has_permutation(k);
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
					allowed[k] = std::min<uint32_t>(n_ones - j, cum_balancable);
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
				// probabilities (weights) with which the indices are
				// selected.
				double total = 0.0;
				for (size_t k = 0; k < size_t(node->max()); k++) {
					weights[k] = selected[k] ? node->permutation_count(k) : 0;
					total += weights[k];
				}

				// Approximate the remaining weights
				double large_weight_total = 0.0;
				double large_weight_total_renorm = 0.0;
				for (size_t k = node->max(); k < idx; k++) {
					const float w =
					    approximate_weight(k, node->remaining(), idx);
					weights[k] = selected[k] ? w : 0;
					large_weight_total += weights[k];
					large_weight_total_renorm += w;
				}

				if (large_weight_total_renorm > 0.0) {
					double inv_total =
					    large_weight_total_renorm / large_weight_total;
					for (size_t k = node->max(); k < idx; k++) {
						weights[k] *= inv_total;
					}
				}

				// Normalise the weights
				if (total > 0.0) {
					double inv_total =
					    (1.0 - large_weight_total_renorm) / total;
					for (size_t k = 0; k < size_t(node->max()); k++) {
						weights[k] = std::max(0.0, weights[k] * inv_total);
					}
				}
				else if (large_weight_total > 0.0) {
					double inv_total = 1.0 / large_weight_total;
					for (size_t k = node->max(); k < idx; k++) {
						weights[k] *= inv_total;
					}
				}

				// Select the indices
				const double rnd =
				    std::uniform_real_distribution<double>(0.0, 1.0)(re);
				double w_sum = 0.0;
				for (size_t k = 0; k < idx; k++) {
					w_sum += weights[k];
					if (selected[k]) {
						chosen_idx = k;  // Ensures sensible selection if sum
						                 // weights[k] < 1.0
					}
					if (w_sum >= rnd) {
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
			ones += res(i, chosen_idx) ? 0 : 1;
			if (res(i, chosen_idx)) {
				exit(1);
			}
			res(i, chosen_idx) = true;
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

BinaryMatrix DataGenerator::generate(
    uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    const DataGenerator::ProgressCallback &progress)
{
	using engine = std::default_random_engine;
	std::default_random_engine re(m_seed);
	if (m_random && !m_balance && !m_unique) {
		return generate_random<engine>(re, n_bits, n_ones, n_samples, progress);
	}
	else {
		return generate_balanced<engine>(re, n_bits, n_ones, n_samples,
		                                 m_random, m_balance, m_unique,
		                                 progress);
	}
}
}
