
/**
 * The BiNAM_Container is the general class to evaluate BiNAMs and the easy to
 * use interface.
 * Within the constructor one gives the general descritption of the memory and
 * data-generation. With several commands one can generate the data, train the
 * storage matrix and at last recall and analyse.
 */
template <typename T>
class BiNAM_Container {
public:
	BiNAM<T> m_BiNAM;
	DataParameters m_params;
	DataGenerationParameters m_datagen;
	BinaryMatrix<T> m_input, m_output, m_recall;
	std::vector<SampleError> m_SampleError;

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
	BiNAM_Container(DataParameters params, DataGenerationParameters datagen)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_params(params),
	      m_datagen(datagen){};
	BiNAM_Container(DataParameters params)
	    : m_BiNAM(params.bits_out(), params.bits_in()),
	      m_params(params),
	      m_datagen(){};
	BiNAM_Container(){};

	/**
	 * Generates input and output data, trains the storage matrix
	 */
	BiNAM_Container<T> &set_up()
	{
		size_t seed =
		    m_datagen.seed() ? m_datagen.seed() : std::random_device()();

		std::thread input_thread([this, seed]() mutable {
			m_input = DataGenerator(seed, m_datagen.random(),
			                        m_datagen.balanced(), m_datagen.unique())
			              .generate<T>(m_params.bits_in(), m_params.ones_in(),
			                           m_params.samples());
		});
		std::thread output_thread([this, seed]() mutable {
			m_output =
			    DataGenerator(seed + 5, m_datagen.random(),
			                  m_datagen.balanced(), m_datagen.unique())
			        .generate<T>(m_params.bits_out(), m_params.ones_out(),
			                     m_params.samples());
		});

		input_thread.join();
		output_thread.join();

		m_BiNAM.train_mat(m_input, m_output);
		return *this;
	};

	/**
	 * Recalls the patterns with the input matrix
	 */
	BiNAM_Container<T> &recall()
	{
		m_recall = m_BiNAM.recallMat(m_input);
		m_SampleError = m_BiNAM.false_bits_mat(m_output, m_recall);
		return *this;
	};

	/**
	 * Returns the vector of SampleError containing the number of false
	 * positives and negatives per sample which is calculated by the recall
	 * function.
	 */
	const std::vector<SampleError> &false_bits() { return m_SampleError; };

	/**
	 * Returns the number of all false positives and negatives of the recall.
	 */
	SampleError sum_false_bits(std::vector<SampleError> vec_err = {
	                               SampleError(-1, -1)})
	{
		if (vec_err[0].fp < 0) {
			vec_err = m_SampleError;
		}

		SampleError sum(0, 0);
		for (size_t i = 0; i < vec_err.size(); i++) {
			sum.fp += vec_err[i].fp;
			sum.fn += vec_err[i].fn;
		}
		return sum;
	};

	/*
	 * Gives back an approximate number of expected false positives
	 */
	SampleError theoretical_false_bits()
	{
		SampleError se(expected_false_positives(m_params), 0);
		return se;
	};

	/**
	 * Calculate the false psotives and negatives as well as the stored
	 * information.
	 * @param recall_matrix: recalled matrix from experiment. If nothing is
	 * given, it takes the member recall matrix.
	 */
	ExpResults analysis(
	    const BinaryMatrix<T> &recall_matrix = BinaryMatrix<T>())
	{
		BinaryMatrix<T> recall_mat;
		if (recall_matrix.size() == 0) {
			recall_mat = m_recall;
		}
		else {
			recall_mat = recall_matrix;
		}
		std::vector<SampleError> se =
		    m_BiNAM.false_bits_mat(m_output, recall_mat);
		double info = entropy_hetero(m_params, se);
		SampleError sum = sum_false_bits(se);
		return ExpResults(info, sum);
	};

	/**
	 * Getter for member matrices
	 */
	const BiNAM<T> &trained_matrix() const { return m_BiNAM; };
	const BinaryMatrix<T> &input_matrix() const { return m_input; };
	const BinaryMatrix<T> &output_matrix() const { return m_output; };
	const BinaryMatrix<T> &recall_matrix() const { return m_recall; };

	void trained_matrix(BiNAM<T> mat) { m_BiNAM = mat; };
	void input_matrix(BinaryMatrix<T> mat) { m_input = mat; };
	void output_matrix(BinaryMatrix<T> mat) { m_output = mat; };
	void recall_matrix(BinaryMatrix<T> mat) { m_recall = mat; };

	/**
	 * Print out matrices for testing purposes
	 */
	void print()
	{
		m_BiNAM.print();
		m_input.print();
		m_output.print();
		m_recall.print();
	};
};
