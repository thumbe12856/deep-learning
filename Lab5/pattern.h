
/**
 * the pattern feature
 * including isomorphic (rotate/mirror)
 *
 * index:
 *  0  1  2  3
 *  4  5  6  7
 *  8  9 10 11
 * 12 13 14 15
 *
 * usage:
 *  pattern({ 0, 1, 2, 3 })
 *  pattern({ 0, 1, 2, 3, 4, 5 })
 */
class pattern : public feature {
public:
	pattern(const std::vector<int>& p, int iso = 8) : feature(1 << (p.size() * 4)), iso_last(iso) {
		if (p.empty()) {
			error << "no pattern defined" << std::endl;
			std::exit(1);
		}

		/**
		 * isomorphic patterns can be calculated by board
		 *
		 * take pattern { 0, 1, 2, 3 } as an example
		 * apply the pattern to the original board (left), we will get 0x1372
		 * if we apply the pattern to the clockwise rotated board (right), we will get 0x2131,
		 * which is the same as applying pattern { 12, 8, 4, 0 } to the original board
		 * { 0, 1, 2, 3 } and { 12, 8, 4, 0 } are isomorphic patterns
		 * +------------------------+       +------------------------+
		 * |     2     8   128     4|       |     4     2     8     2|
		 * |     8    32    64   256|       |     2     4    32     8|
		 * |     2     4    32   128| ----> |     8    32    64   128|
		 * |     4     2     8    16|       |    16   128   256     4|
		 * +------------------------+       +------------------------+
		 *
		 * therefore if we make a board whose value is 0xfedcba9876543210ull (the same as index)
		 * we would be able to use the above method to calculate its 8 isomorphisms
		 */
		for (int i = 0; i < 8; i++) {
			board idx = 0xfedcba9876543210ull;
			if (i >= 4) idx.mirror();
			idx.rotate(i);
			for (int t : p) {
				isomorphic[i].push_back(idx.at(t));
			}
		}
	}
	pattern(const pattern& p) = delete;
	virtual ~pattern() {}
	pattern& operator =(const pattern& p) = delete;

public:

	/**
	 * estimate the value of a given board
	 */
	virtual float estimate(const board& b) const {
		float value = 0;
		for (int i = 0; i < iso_last; i++) {
			size_t index = indexof(isomorphic[i], b);
			value += operator[](index);
		}
		return value;
	}

	/**
	 * update the value of a given board, and return its updated value
	 */
	virtual float update(const board& b, float u) {
		float u_split = u / iso_last;
		float value = 0;
		for (int i = 0; i < iso_last; i++) {
			size_t index = indexof(isomorphic[i], b);
			
			// real update
			operator[](index) += u_split;
			value += operator[](index);
		}
		return value;
	}

	/**
	 * get the name of this feature
	 */
	virtual std::string name() const {
		return std::to_string(isomorphic[0].size()) + "-tuple pattern " + nameof(isomorphic[0]);
	}

public:

	/*
	 * set the isomorphic level of this pattern
	 * 1: no isomorphic
	 * 4: enable rotation
	 * 8: enable rotation and reflection
	 */
	void set_isomorphic(int i = 8) { iso_last = i; }

	/**
	 * display the weight information of a given board
	 */
	void dump(const board& b, std::ostream& out = info) const {
		for (int i = 0; i < iso_last; i++) {
			out << "#" << i << ":" << nameof(isomorphic[i]) << "(";
			size_t index = indexof(isomorphic[i], b);
			for (size_t i = 0; i < isomorphic[i].size(); i++) {
				out << std::hex << ((index >> (4 * i)) & 0x0f);
			}
			out << std::dec << ") = " << operator[](index) << std::endl;
		}
	}

protected:

	size_t indexof(const std::vector<int>& patt, const board& b) const {
		size_t index = 0;
		for (size_t i = 0; i < patt.size(); i++)
			index |= b.at(patt[i]) << (4 * i);
		return index;
	}

	std::string nameof(const std::vector<int>& patt) const {
		std::stringstream ss;
		ss << std::hex;
		std::copy(patt.cbegin(), patt.cend(), std::ostream_iterator<int>(ss, ""));
		return ss.str();
	}

	std::array<std::vector<int>, 8> isomorphic;
	int iso_last;
};
