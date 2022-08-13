#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>
#include <filesystem>
#include <cmath>
#include <numeric>
using namespace cv;
using namespace std;

vector<vector<double>> lin = { {1,1} };
vector<vector<double>> kernel3(3, vector<double>(3, 1));
vector<vector<double>> kernel9(9, vector<double>(9, 1));
vector<vector<double>> kernel15(15, vector<double>(15, 1));
vector<vector<double>> kernel3_m = { {0,1,0},{1,1,1},{0,1,0} };

bool check(int x, int y, int h, int w) {
	if (x >= 0 && y >= 0 && x < h && y < w) {
		return true;
	}
	return false;
}


vector<double> flip(vector<double>& element, int h, int w) {
	vector<double> flipped(h * w);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			flipped[j * h + i] = element[i * w + j];
		}
	}
	return flipped;
}



double erosion(vector<double> data, vector<vector<double>> element) {
	vector<double> kernel;
	for (int i = 0; i < (int)element.size(); i++) {
		for (int j = 0; j < (int)element[0].size(); j++) {
			kernel.push_back(element[i][j]);
		}
	}
	double result = 255;
	for (int i = 0; i < (int)kernel.size(); i++) {
		if (kernel[i] == 1 && data[i] != 255) {
			result = 0;
			break;
		}
	}
	return result;
}


double dilation(vector<double> data, vector<vector<double>> element) {
	vector<double> kernel;
	for (int i = 0; i < (int)element.size(); i++) {
		for (int j = 0; j < (int)element[0].size(); j++) {
			kernel.push_back(element[i][j]);
		}
	}
	kernel = flip(kernel, (int)element.size(), (int)element[0].size());
	double result = 0;
	for (int i = 0; i < (int)kernel.size(); i++) {
		if (kernel[i] == 1 && data[i] == 255) {
			result = 255;
			break;
		}
	}
	return result;
}




double choose_operation(vector<double> data, string operation, vector<vector<double>> element) {
	double result;
	if (operation == "erosion") {
		result = erosion(data, element);
	}
	else if (operation == "dilation") {
		result = dilation(data, element);
	}
	return result;
}




double filter_vector(vector<uchar> pixel_values, int h, int w, int x, int y, string operation, vector<vector<double>> element, bool threshold = true) {
	vector<double> data;
	int row = (int)element.size();
	int col = (int)element[0].size();
	for (int i = -row / 2; i <= row / 2; i++) {
		for (int j = -col / 2; j <= col / 2; j++) {
			int x_dash = x + i;
			int y_dash = y + j;
			if (check(x_dash, y_dash, h, w)) {
				data.push_back(pixel_values[x_dash * w + y_dash]);
			}
			else data.push_back(0);
		}
	}

	double result = choose_operation(data, operation, element);
	if (threshold) {
		return max(result, 0.0);
	}
	else return result;
}




vector<uchar> perform_operation(vector<uchar> pixel_values, string operation, int h, int w, vector<vector<double>> element) {
	vector<uchar> filtered_val(h*w);

	if (operation == "erosion" || operation == "dilation") {
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				filtered_val[i * w + j] = (int)filter_vector(pixel_values, h, w, i, j, operation, element);
			}
		}
	}
	else if (operation == "opening") {
		vector<uchar> temp(h * w);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				temp[i * w + j] = (int)filter_vector(pixel_values, h, w, i, j, "erosion", element);
			}
		}
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				filtered_val[i * w + j] = (int)filter_vector(temp, h, w, i, j, "dilation", element);
			}
		}
	}
	else if (operation == "closing") {
		vector<uchar> temp(h * w);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				temp[i * w + j] = (int)filter_vector(pixel_values, h, w, i, j, "dilation", element);
			}
		}
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++) {
				filtered_val[i * w + j] = (int)filter_vector(temp, h, w, i, j, "erosion", element);
			}
		}
	}
	else {
		cout << "Operation not implemented\n";
		exit(0);
	}
	return filtered_val;
}




void morphological_processing(string name, string operation, string kernel, string dirname = ".", bool display = true) {
	string basename = name.substr(name.find_last_of("/\\") + 1);
	string identifier = basename.substr(0, basename.find_last_of("."));

	vector<vector<double>> element;
	if (kernel == "lin") {
		element = lin;
	}
	else if (kernel == "kernel3") {
		element = kernel3;
	}
	else if (kernel == "kernel9") {
		element = kernel9;
	}
	else if (kernel == "kernel15") {
		element = kernel15;
	}
	else if (kernel == "kernel3_m") {
		element = kernel3_m;
	}
	else {
		cout << "Kernel does not exist\n";
		exit(0);
	}

	Mat src = imread(name, IMREAD_GRAYSCALE);
	threshold(src, src, 100, 255, 0);

	if (display) {
		imshow(identifier, src);
		waitKey();
	}

	vector<uchar> pixel_values;
	pixel_values.assign(src.data, src.data + src.total());

	vector<uchar> filtered_val;
	filtered_val = perform_operation(pixel_values, operation, src.rows, src.cols, element);

	Mat res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
	if (display) {
		imshow(identifier + "_" + operation, res);
		waitKey();
	}
	imwrite(dirname + "//" + identifier + "_" + operation + "_" + kernel + ".jpg", res);
}




void generate_samples(string input_dir, string output_dir) {
	vector<string> kernels = { "lin","kernel3","kernel9","kernel15","kernel3_m" };
	for (const auto& file : filesystem::directory_iterator(input_dir)) {
		string filepath = file.path().string();

		vector<thread> threads;
		for (auto kernel : kernels) {
			threads.push_back(thread(morphological_processing, filepath, "erosion", kernel, output_dir, false));
			threads.push_back(thread(morphological_processing, filepath, "dilation", kernel, output_dir, false));
			threads.push_back(thread(morphological_processing, filepath, "opening", kernel, output_dir, false));
			threads.push_back(thread(morphological_processing, filepath, "closing", kernel, output_dir, false));
		}
		
		cout << "Threads spawned for " << filepath << "\n";
		for (auto& th : threads) {
			th.join();
		}
	}
}




int main()
{
	morphological_processing("ricegrains_mono.bmp", "opening", "kernel9");
	//generate_samples("Input", "Output");
	return 0;
}