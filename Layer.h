#ifndef __LAYER__H_
#define __LAYER__H_
#include <Eigen/Eigen>

using namespace Eigen;
using namespace std;

class Layer {
	public:
		int unit_num;
		int batch_size;
		MatrixXd output;
		MatrixXd weight;
		MatrixXd error;
		
		Layer(int un, int b) {
			unit_num = un;
			batch_size = b;
			output = MatrixXd::Constant(batch_size, unit_num + 1, 1);
		}
		
		Layer(int un, int pre_un, int b) {
			unit_num = un;
			batch_size = b;
			output = MatrixXd::Constant(batch_size, unit_num + 1, 1);
			error = MatrixXd::Constant(batch_size, unit_num + 1, 0);
			weight = MatrixXd::Constant(pre_un + 1, unit_num + 1, 1);
			for (int i = 0; i <= pre_un; i++) {
				for (int j = 0; j <= un; j++) {
					weight(i, j) = randomReal(-0.5, 0.5);
				}
			}
		}
		
		inline double randomReal(double low, double high) {
			return ((double) rand() / RAND_MAX) * (high-low) + low;
		}
};		
#endif	