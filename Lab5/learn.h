
class learning {
public:
	learning() {}
	~learning() {}

	/**
	 * add a feature into tuple networks
	 *
	 * note that feats is std::vector<feature*>,
	 * therefore you need to keep all the instances somewhere
	 */
	void add_feature(feature* feat) {
		feats.push_back(feat);

		info << feat->name() << ", size = " << feat->size();
		size_t usage = feat->size() * sizeof(float);
		if (usage >= (1 << 30)) {
			info << " (" << (usage >> 30) << "GB)";
		} else if (usage >= (1 << 20)) {
			info << " (" << (usage >> 20) << "MB)";
		} else if (usage >= (1 << 10)) {
			info << " (" << (usage >> 10) << "KB)";
		}
		info << std::endl;
	}

	/**
	 * accumulate the total value of given state
	 */
	float estimate(const board& b) const {
		debug << "estimate " << std::endl << b;
		float value = 0;
		for (feature* feat : feats) {

			// in pattern.h
			value += feat->estimate(b);
		}
		return value;
	}

	/**
	 * update the value of given state and return its new value
	 */
	float update(const board& b, float u) const {
		float u_split = u / feats.size();
		float value = 0;
		for (feature* feat : feats) {
			
			// in pattern.h
			value += feat->update(b, u_split);
		}
		return value;
	}

	/**
	 * select a best move of a before state b
	 *
	 * return should be a state whose
	 *  before_state() is b
	 *  after_state() is b's best successor (after state)
	 *  action() is the best action
	 *  reward() is the reward of performing action()
	 *  value() is the estimated value of after_state()
	 *
	 * you may simply return state() if no valid move
	 */
	state select_best_move(const board& b, int epoch) const {
		state after[4] = { 0, 1, 2, 3 }; // up, right, down, left
		//std::cin.get(); 
		state* best = after;
		float e = 0;

		for (state* move = after; move != after + 4; move++) {
			if (move->assign(b)) {
				e = move->reward() + estimate(move->after_state());
				//e = move->reward() + estimate(move->before_state());
				//debug << "estimate: " << e << std::endl;
				move->set_value(e);

				if (move->value() > best->value()) {
					best = move;
				}
			} else {
				move->set_value(-std::numeric_limits<float>::max());
			}

			//debug << "test " << *move;
			//debug << "test before state" << std::endl << move->before_state();
			//debug << "test after state" << std::endl << move->after_state();
			debug << "epoch " << epoch;
		}
		return *best;
	}

	/**
	 * update the tuple network by an episode
	 *
	 * path is the sequence of states in this episode,
	 * the last entry in path (path.back()) is the final state
	 *
	 * for example, a 2048 games consists of
	 *  (initial) s0 --(a0,r0)--> s0' --(popup)--> s1 --(a1,r1)--> s1' --(popup)--> s2 (terminal)
	 *  where sx is before state, sx' is after state
	 *
	 * its path would be
	 *  { (s0,s0',a0,r0), (s1,s1',a1,r1), (s2,s2,x,-1) }
	 *  where (x,x,x,x) means (before state, after state, action, reward)
	 */
	void update_episode(std::vector<state>& path, float alpha = 0.1) const {
		
		// V(terminal state) = 0
		// V(S(t+1))
		float exact = 0;

		for (path.pop_back() /* terminal state */; path.size(); path.pop_back()) {
			if(path.size() <= 1) {
				break;
			}

			/**
			 * move.after_state() = S(t + 1)
			 * move.before_state() = S(t)
			 */
			state& move = path.back();

			/**
			 * before_movemove.after_state() = S(t)
			 * before_move.before_state() = S(t - 1)
			 */			
			state& before_move = *(path.end()-2);
			
			/**
			 * after state:
			 * r(t+1) + V(S'(t+1)) - V(S'(t))
			 */
			//float errorr = move.reward() + exact - move.value();

			/**
			 * 
			 * V(S'(t)) = V(S'(t)) + alpha * error
			 */
			//exact = move.reward() + update(move.after_state(), alpha * errorr);


			/**
			 * state:
			 * r(t+1) + V(S(t+1)) - V(S(t))
			 */
			float errorr = move.reward() + exact - before_move.value();

			/**
			 * V(S(t)) = V(S(t)) + alpha * error
			 */
			exact = move.reward() + update(before_move.after_state(), alpha * errorr);
		}
	}

	/**
	 * update the statistic, and display the status once in 1000 episodes by default
	 *
	 * the format would be
	 * 1000   mean = 273901  max = 382324
	 *        512     100%   (0.3%)
	 *        1024    99.7%  (0.2%)
	 *        2048    99.5%  (1.1%)
	 *        4096    98.4%  (4.7%)
	 *        8192    93.7%  (22.4%)
	 *        16384   71.3%  (71.3%)
	 *
	 * where (let unit = 1000)
	 *  '1000': current iteration (games trained)
	 *  'mean = 273901': the average score of last 1000 games is 273901
	 *  'max = 382324': the maximum score of last 1000 games is 382324
	 *  '93.7%': 93.7% (937 games) reached 8192-tiles in last 1000 games (a.k.a. win rate of 8192-tile)
	 *  '22.4%': 22.4% (224 games) terminated with 8192-tiles (the largest) in last 1000 games
	 */
	void make_statistic(size_t n, std::ofstream& recordFile, const board& b, int score, int unit = 1000) {
		scores.push_back(score);
		maxtile.push_back(0);
		for (int i = 0; i < 16; i++) {
			maxtile.back() = std::max(maxtile.back(), b.at(i));
		}

		if (n % unit == 0) { // show the training process
			if (scores.size() != size_t(unit) || maxtile.size() != size_t(unit)) {
				error << "wrong statistic size for show statistics" << std::endl;
				std::exit(2);
			}

		
			int sum = std::accumulate(scores.begin(), scores.end(), 0);
			int max = *std::max_element(scores.begin(), scores.end());
			int stat[16] = { 0 };
			for (int i = 0; i < 16; i++) {
				stat[i] = std::count(maxtile.begin(), maxtile.end(), i);
			}
			float mean = float(sum) / unit;
			float coef = 100.0 / unit;
			info << n;
			info << "\t" "mean = " << mean;
			info << "\t" "max = " << max;

			int terminalTile[10] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
            float tileIndex[10] ={ 100,  0,   0,   0,   0,    0,    0,    0,    0,     0};

            for (int t = 1, c = 0; c < unit; c += stat[t++]) {
                if (stat[t] == 0) continue;
                int accu = std::accumulate(stat + t, stat + 16, 0);
                info << "\t" << ((1 << t) & -2u) << "\t" << (accu * coef) << "%";
                info << "\t(" << (stat[t] * coef) << "%)" << std::endl;
                for(int i=0; i<10; i++) {
                    if(terminalTile[i] == ((1 << t) & -2u)) {
                            tileIndex[i] = (accu * coef);
                    }
				}
            }

            recordFile << "\n" << mean << "," << max << ",";
            for(int i = 0; i < 10; i++) {
                recordFile << tileIndex[i] << ",";
            }

			scores.clear();
			maxtile.clear();
		}
	}

	/**
	 * display the weight information of a given board
	 */
	void dump(const board& b, std::ostream& out = info) const {
		out << b << "estimate = " << estimate(b) << std::endl;
		for (feature* feat : feats) {
			out << feat->name() << std::endl;
			feat->dump(b, out);
		}
	}

	/**
	 * load the weight table from binary file
	 * you need to define all the features (add_feature(...)) before call this function
	 */
	void load(const std::string& path) {
		std::ifstream in;
		in.open(path.c_str(), std::ios::in | std::ios::binary);
		if (in.is_open()) {
			size_t size;
			in.read(reinterpret_cast<char*>(&size), sizeof(size));
			if (size != feats.size()) {
				error << "unexpected feature count: " << size << " (" << feats.size() << " is expected)" << std::endl;
				std::exit(1);
			}
			for (feature* feat : feats) {
				in >> *feat;
				info << feat->name() << " is loaded from " << path << std::endl;
			}
			in.close();
		}
	}

	/**
	 * save the weight table to binary file
	 */
	void save(const std::string& path) {
		std::ofstream out;
		out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (out.is_open()) {
			size_t size = feats.size();
			out.write(reinterpret_cast<char*>(&size), sizeof(size));
			for (feature* feat : feats) {
				out << *feat;
				info << feat->name() << " is saved to " << path << std::endl;
			}
			out.flush();
			out.close();
		}
	}

private:
	std::vector<feature*> feats;
	std::vector<int> scores;
	std::vector<int> maxtile;
};
