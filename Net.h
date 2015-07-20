#include "Layer.h"
#include <vector>

class Net {
	public:
		int layer_num;
		vector<Layer*> layers;
		vector<int> ans;
		double alpha;
		int batch_size;
		
		Net(int ln, int un[], int b);
		void train(vector< vector<double> >& data, int start, int end);
		void setInput(vector< vector<double> >& data, int start, int end);
		void propagateNet();
		void backPropagateNet();
		void adjustWeight();
		bool test(vector<double>& item);
		void printParam();
};