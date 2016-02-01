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

#include <map>
#include <limits>
#include <random>

#include "data.hpp"
#include "ncr.hpp"

using namespace arma;

namespace nam {
namespace {

class PermutationTrieNode {
private:
	int m_idx;
	int m_remaining;
	uint64_t m_total;
	Row<uint32_t> m_max_permutations;
	Row<uint32_t> m_permutations;
	std::map<int, PermutationTrieNode> m_children;

public:
	PermutationTrieNode(int idx, int remaining)
	    : m_idx(idx), m_remaining(remaining), m_max_permutations(m_idx)
	{
		static constexpr uint32_t MAX_PERMS =
		    std::numeric_limits<uint32_t>::max();
		for (int i = 0; i < m_idx; i++) {
			if (i > 0 && m_max_permutations[i - 1] == MAX_PERMS) {
				m_max_permutations[i] = MAX_PERMS;
			}
			else {
				m_max_permutations[i] = ncr_clamped32(i, remaining - 1);
			}
		}
		m_permutations = m_max_permutations;
		m_total = accu(m_max_permutations);
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
		return m_children.emplace(m_idx,
		                          PermutationTrieNode({idx, m_remaining - 1}))
		    .first->second;
	}

	bool has_permutation(int idx) { return m_permutations[idx] > 0; }

	bool decrement_permutation(int idx)
	{
		if (m_total > 1) {
			m_permutations[idx]--;
			m_total--;
			return true;
		}

		m_permutations = m_max_permutations;
		m_total = accu(m_max_permutations);
		return false;
	}
};

template <typename RandomEngine>
DataGenerator::MatrixType generate_balanced(
    RandomEngine &re, uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    bool random, bool balance, bool unique,
    const DataGenerator::ProgressCallback &progress)
{
	DataGenerator::MatrixType res(n_samples, n_bits,
	                              fill::zeros);  // Result matrix
	Row<uint32_t> usage(
	    n_bits, fill::zeros);  // Vector tracking how often each bit is used
	Row<uint32_t> allowed(n_bits, fill::zeros);  // Temporary vector holding the
	                                             // number of ones which can be
	                                             // inserted
	Row<uint8_t> balancable(n_bits, fill::zeros);
	Row<uint8_t> selected(n_bits, fill::zeros);

	PermutationTrieNode root(n_bits, n_samples);
	for (size_t i = 0; i < n_samples; i++) {
		PermutationTrieNode *node = &root;
		for (size_t j = 0; j < n_ones; j++) {
			const size_t idx = node->idx();

			// Select those elements for which a permutation is left
			for (size_t k = 0; k < idx; k++) {
				selected[k] = node->has_permutation(k);
			}

			// Try to balance the bit usage
			if (balance) {
				// Check which indices are minimally used, only select those
				// which are
				const uint32_t min_usage = min(usage.subvec(0, idx - 1));
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
					selected.subvec(0, idx - 1) = balancable.subvec(0, idx - 1);
				}
			}

			size_t chosen_idx = 0;
			if (random) {
				// TODO
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
			node->decrement_permutation(chosen_idx);
			node = &node->fetch(chosen_idx);
		}

		// Regularly call the progress function
		if ((i == 0) || (i == n_samples - 1) || (i % 100 == 0)) {
			if (!progress(float(i) / float(n_samples - 1))) {
				break;
			}
		}
	}

	return std::move(res);
}

template <typename RandomEngine>
DataGenerator::MatrixType generate_random(
    RandomEngine &re, size_t n_bits, size_t n_ones, size_t n_samples,
    const DataGenerator::ProgressCallback &progress)
{
	DataGenerator::MatrixType res(n_samples, n_bits, fill::zeros);
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
	return std::move(res);
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

DataGenerator::MatrixType DataGenerator::generate(
    uint32_t n_bits, uint32_t n_ones, uint32_t n_samples,
    const DataGenerator::ProgressCallback &progress)
{
	std::default_random_engine re(m_seed);
	if (m_random && !m_balance && !m_unique) {
		return std::move(
		    generate_random(re, n_bits, n_ones, n_samples, progress));
	}
	else {
		return std::move(generate_balanced(re, n_bits, n_ones, n_samples,
		                                       m_random, m_balance, m_unique,
		                                       progress));
	}
}
}
