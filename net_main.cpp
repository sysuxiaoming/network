#include "Net.h"
#include <vector>
#include <iostream>
#include <string.h>
#include <fstream>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define uint32 unsigned int
#define BigtoLittle32(A)   ((( (uint32)(A) & 0xff000000) >> 24) | \
	(((uint32)(A)& 0x00ff0000) >> 8) | \
	(((uint32)(A)& 0x0000ff00) << 8) | \
	(((uint32)(A)& 0x000000ff) << 24))
using namespace std;

vector< vector<double> > train_data;
vector< vector<double> > test_data;
int batch_size = 1024;

int main() {
	ifstream fin("E:/项目/深度学习/train-images.idx3-ubyte", ios::binary);
	unsigned int temp;
	unsigned int train_num;
	unsigned int row_num;
	unsigned int col_num;
	fin.read((char*)&temp, 4);
	temp = BigtoLittle32(temp);
	fin.read((char*)&train_num, 4);
	train_num = BigtoLittle32(train_num);
	fin.read((char*)&row_num, 4);
	row_num = BigtoLittle32(row_num);
	fin.read((char*)&col_num, 4);
	col_num = BigtoLittle32(col_num);
	int class_index = row_num * col_num;
	train_data.resize(train_num);
	for (int i = 0; i < train_num; i++) {
		train_data[i].resize(class_index + 1);
	}
	for (int i = 0; i < train_num; i++) {
		for (int j = 0; j < class_index; j++) {
			char c;
			fin.read((char*)&c, 1);
			train_data[i][j] = (unsigned int)(c & 0x000000ff);
		}
	}
	fin.close();
	fin.open("E:/项目/深度学习/train-labels.idx1-ubyte", ios::binary);
	fin.read((char*)&temp, 4);
	temp = BigtoLittle32(temp);
	fin.read((char*)&train_num, 4);
	train_num = BigtoLittle32(train_num);
	for (int i = 0; i < train_num; i++) {
		char c;
		fin.read((char*)&c, 1);
		train_data[i][class_index] = (unsigned int)(c & 0x000000ff);
	}
	fin.close();

	int test_num;
	fin.open("E:/项目/深度学习/t10k-images.idx3-ubyte", ios::binary);
	fin.read((char*)&temp, 4);
	temp = BigtoLittle32(temp);
	fin.read((char*)&test_num, 4);
	test_num = BigtoLittle32(test_num);
	fin.read((char*)&row_num, 4);
	row_num = BigtoLittle32(row_num);
	fin.read((char*)&col_num, 4);
	col_num = BigtoLittle32(col_num);
	test_data.resize(test_num);
	for (int i = 0; i < test_num; i++) {
		test_data[i].resize(class_index + 1);
	}
	for (int i = 0; i < test_num; i++) {
		for (int j = 0; j < class_index; j++) {
			char c;
			fin.read((char*)&c, 1);
			test_data[i][j] = (unsigned int)(c & 0x000000ff);
		}
	}
	fin.close();
	fin.open("E:/项目/深度学习/t10k-labels.idx1-ubyte", ios::binary);
	fin.read((char*)&temp, 4);
	temp = BigtoLittle32(temp);
	fin.read((char*)&test_num, 4);
	test_num = BigtoLittle32(test_num);
	for (int i = 0; i < test_num; i++) {
		char c;
		fin.read((char*)&c, 1);
		test_data[i][class_index] = (unsigned int)(c & 0x000000ff);
	}
	fin.close();

	int un[] = { class_index, 300, 10 };
	Net net(3, un, batch_size);
	int pass = 1000;
	cout << "start train" << endl;
	while (pass--) {
		clock_t t1 = clock();
		int i = 0;
		while (true) {
			if (i + batch_size <= train_num) {
				net.train(train_data, i, i + batch_size);
				i += batch_size;
			}
			else {
				break;
			}
		}
		clock_t t2 = clock();
		cout << "pass" << pass << " cost time " << t2 - t1 << endl;
		int correct = 0;
		for (int i = 0; i < test_num; i++) {
			if (net.test(test_data[i])) {
				correct++;
			}
		}
		cout << "test correct rate" << (double)correct / (double)test_num << endl;
	}
}

