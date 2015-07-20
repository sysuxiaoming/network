#include "Net.h"
#include <iostream>
#include <stdlib.h>

Net::Net(int ln, int un[], int b) {
	batch_size = b;
	ans.resize(b);
	alpha = 0.5;
	layer_num = ln;
	layers.resize(ln);
	layers[0] = new Layer(un[0], b);
	for (int i = 1; i < ln; i++) {
		layers[i] = new Layer(un[i], un[i - 1], b);
	}
}
		
void Net::train(vector< vector<double> >& data, int start, int end) {
	setInput(data, start, end);
	propagateNet();
	backPropagateNet();
	adjustWeight();
}
		
void Net::setInput(vector< vector<double> >& data, int start, int end) {
	batch_size = end - start;
	for (int i = 0; i < batch_size; i++) {
		int t = data[0].size();
		for (int j = 1; j < t; j++) {
			layers[0]->output(i, j) = data[start+i][j-1];
		}
		layers[0]->output(i, 0) = 1;
		ans[i] = data[start+i][t-1] + 1;
	}
}
		
void Net::propagateNet() {
	for (int i = 1; i < layer_num; i++) {
		MatrixXd temp = layers[i-1]->output * layers[i]->weight;
		layers[i]->output = (1 / (1 + (-temp.array()).exp())).matrix();
		for (int j = 0; j < batch_size; j++) {
			layers[i]->output(j, 0) = 1;
		}
	}
}
		
void Net::backPropagateNet() {
	Layer* outLayer = layers[layer_num-1];
	double error = 0;
	for (int m = 0; m < batch_size; m++) {
		for (int i = 1; i <= outLayer->unit_num; i++) {
			double t = 0;
			if (i == ans[m]) {
				t = 1;
			}
			double dot_out = t - outLayer->output(m, i);
			outLayer->error(m, i) = dot_out;
			error -= t * log(outLayer->output(m, i)) + (1-t) * log(1-outLayer->output(m, i));
		}
		outLayer->error(m, 0) = 0;
	}
	cout << error << endl;
	for (int i = layer_num - 2; i > 0; i--) {
		MatrixXd temp = layers[i+1]->error * (layers[i+1]->weight).transpose();
		ArrayXXd a = layers[i]->output.array();
		ArrayXXd dot_sigmoid = (a * (1 - a)).matrix();
		layers[i]->error = (dot_sigmoid * temp.array()).matrix();
		for (int j = 0; j < batch_size; j++) {
			layers[i]->error(j, 0) = 0;
		}
	}
}
		
void Net::adjustWeight() {
	double f = alpha / batch_size;
	for (int i = 1; i < layer_num; i++) {
		MatrixXd temp = layers[i-1]->output.transpose() * layers[i]->error;
		layers[i]->weight = (layers[i]->weight.array() + f * temp.array()).matrix();
	}
}
		
bool Net::test(vector<double>& value) {
	int t = value.size();
	for (int i = 1; i < t; i++) {
		layers[0]->output(0, i) = value[i - 1];
	}
	layers[0]->output(0, 0) = 1;
	int ans = value[t - 1] + 1;
	for (int i = 1; i < layer_num; i++) {
		for (int j = 1; j <= layers[i]->unit_num; j++) {
			double sum = 0;
			for (int k = 0; k < layers[i-1]->unit_num; k++) {
				sum += layers[i]->weight(k, j) * layers[i-1]->output(0, k);
			}
			layers[i]->output(0, j) = 1 / (1 + exp(-1 * sum));
		}
	}
	Layer* layer = layers[layer_num-1];
	int pred = 0;
	double max = -1;
	for (int i = 1; i <= layer->unit_num; i++) {
		if (max < layer->output(0, i)) {
			max = layer->output(0, i);
			pred = i;
		}
	}
	if (pred == ans) {
		return true;
	} else {
		return false;
	}
}
		
void Net::printParam() {
	for (int i = 1; i < layer_num; i++) {
		for (int j = 1; j <= layers[i]->unit_num; j++) {
			for (int k = 0; k <= layers[i-1]->unit_num; k++) {
				cout << layers[i]->weight(k, j) << " ";
			}
		}
	}
	cout << endl;
}