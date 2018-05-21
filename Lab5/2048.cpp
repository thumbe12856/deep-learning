/**
 * Temporal Difference Learning Demo for Game 2048
 * use 'g++ -std=c++0x -O3 -g -o 2048 2048.cpp' to compile the source
 * https://github.com/moporgic/TDL2048-Demo
 *
 * Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
 * http://www.aigames.nctu.edu.tw
 *
 * References:
 * [1] Szubert, Marcin, and Wojciech Jaśkowski. "Temporal difference learning of n-tuple networks for the game 2048."
 * Computational Intelligence and Games (CIG), 2014 IEEE Conference on. IEEE, 2014.
 * [2] Wu, I-Chen, et al. "Multi-stage temporal difference learning for 2048."
 * Technologies and Applications of Artificial Intelligence. Springer International Publishing, 2014. 366-378.
 * [3] Oka, Kazuto, and Kiminori Matsuzaki. "Systematic selection of n-tuple networks for 2048."
 * International Conference on Computers and Games. Springer International Publishing, 2016.
 */
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <vector>
#include <array>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>

/**
 * output streams
 * to enable debugging (more output), just change the line to 'std::ostream& debug = std::cout;'
 */
std::ostream& info = std::cout;
std::ostream& error = std::cerr;
std::ostream& debug = *(new std::ofstream);

//std::ostream& debug = std::cerr;

#include "board.h"
#include "feature.h"
#include "pattern.h"
#include "state.h"
#include "learn.h"

int main(int argc, const char* argv[]) {
	info << "TDL2048-Demo" << std::endl;
	learning tdl;

	// set the learning parameters
	float alpha = 0.1;
	size_t total = 1000;
	unsigned seed;
	__asm__ __volatile__ ("rdtsc" : "=a" (seed));
	info << "alpha = " << alpha << std::endl;
	info << "total = " << total << std::endl;
	info << "seed = " << seed << std::endl;
	std::srand(seed);

	// initialize the features
	// pattern({index})
	// 1 個 board 有 4 個 features 組成
	// 1 個 feature  有 8 個 isomorphic 組成 (旋轉、垂直翻轉)
	tdl.add_feature(new pattern({ 0, 1, 2, 3, 4, 5 }));
	tdl.add_feature(new pattern({ 4, 5, 6, 7, 8, 9 }));
	tdl.add_feature(new pattern({ 0, 1, 2, 4, 5, 6 }));
	tdl.add_feature(new pattern({ 4, 5, 6, 8, 9, 10 }));

	// restore the model from file
	tdl.load("");

	// train the model
	std::vector<state> path;
	path.reserve(20000);

	std::ofstream recordFile;
	recordFile.open("T-state.csv");

	recordFile << "mean, sum, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384";

	for (size_t n = 1; n <= total; n++) {
		board b;
		int score = 0;

		// play an episode
		debug << "begin episode" << std::endl;
		b.init();
		while (true) {
			debug << "state" << std::endl << b;
			state best = tdl.select_best_move(b, n);
			path.push_back(best);

			if (best.is_valid()) {
				debug << "best " << best;
				score += best.reward();
				
				// S(t) -> S(t)'
				b = best.after_state();
				
				// S(t)' -> S(t)'' = S(t+1)
				b.popup();
			} else {
				break;
			}
		}
		debug << "end episode" << std::endl;

		// update by TD(0)
		tdl.update_episode(path, alpha);
		tdl.make_statistic(n, recordFile, b, score);
		path.clear();

		/*
		if(n > 20) {
			recordFile.close();
		}
		*/
	}

	recordFile.close();
	// store the model into file
	tdl.save("model");

	return 0;
}
