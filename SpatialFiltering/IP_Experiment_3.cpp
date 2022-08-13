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

class tracker_in {
public:
	vector<string> images;
	int idx;
	vector<string> filter_names;
	int filter_idx;
	int f;
	string output_dir;

	tracker_in() {
		this->idx = 0;
		this->f = 0;
		this->filter_names = { "mean", "median", "gaussian" };
		this->filter_idx = 0;
		this->output_dir = ".";
	}
};


double filter_vector(vector<uchar> pixel_values, int h, int w, int x, int y, string filter_name, int f, bool threshold);
double filter_vector(vector<double> pixel_values, int h, int w, int x, int y, string filter_name, int f);



double convolve(vector<double> data, vector<double> filter) {
	assert(data.size() == filter.size());
	double result = 0;
	result = transform_reduce(data.begin(), data.end(), filter.begin(), 0.0);
	return result;
}


bool check(int x, int y, int h, int w) {
	if (x >= 0 && y >= 0 && x < h && y < w) {
		return true;
	}
	return false;
}


double mean_filter(vector<double> data, int f) {
	vector<double> filter(f * f, (1.0 / (f * f)));
	double result = convolve(data, filter);
	return result;
}


double median_filter(vector<double> data, int f) {
	sort(data.begin(), data.end());
	double result = data[f / 2];
	return result;
}



double gaussian_filter(vector<double> data, int f) {
	vector<double> filter;
	double norm_factor = 0;
	for (int i = -f / 2; i <= f / 2; i++) {
		for (int j = -f / 2; j <= f / 2; j++) {
			double val = exp(-double((i * i) + (j * j)) / 2);
			norm_factor += val;
			filter.push_back(val);
		}
	}

	for (int i = 0; i < filter.size(); i++) {
		filter[i] /= norm_factor;
	}

	double result = convolve(data, filter);
	return result;
}



double sobel_horizontal_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { -1,-2,-1,0,0,0,1,2,1 };
	double result = convolve(data, filter);
	return result;
}

double sobel_vertical_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { -1,0,1,-2,0,2,-1,0,1 };
	double result = convolve(data, filter);
	return result;
}

double sobel_diagonal_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { -2,-1,0,-1,0,1,0,1,2 };
	double result = convolve(data, filter);
	return result;
}



double prewitt_horizontal_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { -1,-1,-1,0,0,0,1,1,1 };
	double result = convolve(data, filter);
	return result;
}

double prewitt_vertical_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { -1,0,1,-1,0,1,-1,0,1 };
	double result = convolve(data, filter);
	return result;
}



double laplacian_filter(vector<double> data, int f) {
	if (f != 3) {
		cout << "Filter not implemented\n";
		exit(0);
	}
	vector<double> filter = { 0,1,0,1,-4,1,0,1,0 };
	double result = convolve(data, filter);
	return result;
}


double laplacian_of_gaussian_filter(vector<double> data, int f) {
	if (f != 5) {
		cout << "Filter not implemented\n";
		exit(0);
	}

	vector<double> filter = { 0,0,1,0,0,0,1,2,1,0,1,2,-16,2,1,0,1,2,1,0,0,0,1,0,0 };
	double result = convolve(data, filter);
	return result;
}


double high_pass(vector<double> data) {
	vector<double> filter = { -1,-1,-1,-1,8,-1,-1,-1,-1 };
	double result = convolve(data, filter);
	return result;
}




double choose_filter(vector<double> data, string filter_name, int f) {
	double result;
	if (filter_name == "mean") {
		result = mean_filter(data, f);
	}
	else if (filter_name == "median") {
		result = median_filter(data, f);
	}
	else if (filter_name == "gaussian") {
		result = gaussian_filter(data, f);
	}
	else if (filter_name == "sobel_h") {
		result = sobel_horizontal_filter(data, f);
	}
	else if (filter_name == "sobel_v") {
		result = sobel_vertical_filter(data, f);
	}
	else if (filter_name == "sobel_d") {
		result = sobel_diagonal_filter(data, f);
	}
	else if (filter_name == "prewitt_h") {
		result = prewitt_horizontal_filter(data, f);
	}
	else if (filter_name == "prewitt_v") {
		result = prewitt_vertical_filter(data, f);
	}
	else if (filter_name == "laplacian") {
		result = laplacian_filter(data, f);
	}
	else if (filter_name == "log") {
		result = laplacian_of_gaussian_filter(data, f);
	}
	else if (filter_name == "high_pass") {
		result = high_pass(data);
	}
	else {
		cout << "Filter not implemented\n";
		exit(0);
	}

	return result;
}




double filter_vector(vector<uchar> pixel_values, int h, int w, int x, int y, string filter_name, int f, bool threshold = true) {
	vector<double> data;
	for (int i = -f / 2; i <= f / 2; i++) {
		for (int j = -f / 2; j <= f / 2; j++) {
			int x_dash = x + i;
			int y_dash = y + j;
			if (check(x_dash, y_dash, h, w)) {
				data.push_back(pixel_values[x_dash * w + y_dash]);
			}
			else data.push_back(0);
		}
	}

	double result = choose_filter(data, filter_name, f);
	if (threshold) {
		return max(result, 0.0);
	}
	else return result;
}


double filter_vector(vector<double> pixel_values, int h, int w, int x, int y, string filter_name, int f) {
	vector<double> data;
	for (int i = -f / 2; i <= f / 2; i++) {
		for (int j = -f / 2; j <= f / 2; j++) {
			int x_dash = x + i;
			int y_dash = y + j;
			if (check(x_dash, y_dash, h, w)) {
				data.push_back(pixel_values[x_dash * w + y_dash]);
			}
			else data.push_back(0);
		}
	}

	double result = choose_filter(data, filter_name, f);
	return result;
}




void compute_filter(string name, string filter_name, int f, string dirname = ".", bool display = true) {
	string basename = name.substr(name.find_last_of("/\\") + 1);
	string identifier = basename.substr(0, basename.find_last_of("."));

	Mat src = imread(name, IMREAD_GRAYSCALE);
	if (display) {
		imshow(identifier, src);
		waitKey();
	}

	vector<uchar> pixel_values;
	pixel_values.assign(src.data, src.data + src.total());

	vector<uchar> filtered_val(src.total());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			filtered_val[i * src.rows + j] = (int)filter_vector(pixel_values, src.rows, src.cols, i, j, filter_name, f);
		}
	}

	Mat res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
	if (display) {
		imshow("Display", res);
		waitKey();
	}
	imwrite(dirname + "//" + identifier + "_" + filter_name + "_" + to_string(f) + ".jpg", res);
}




void prewitt_filter(string name, string dirname = ".", bool display = true) {
	string basename = name.substr(name.find_last_of("/\\") + 1);
	string identifier = basename.substr(0, basename.find_last_of("."));

	Mat src = imread(name, IMREAD_GRAYSCALE);
	if (display) {
		imshow(identifier, src);
		waitKey();
	}

	vector<uchar> pixel_values;
	pixel_values.assign(src.data, src.data + src.total());

	vector<uchar> intermediate_val(src.total());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			intermediate_val[i * src.rows + j] = (int)filter_vector(pixel_values, src.rows, src.cols, i, j, "prewitt_h", 3);
		}
	}

	vector<uchar> filtered_val(src.total());

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			filtered_val[i * src.rows + j] = (int)filter_vector(intermediate_val, src.rows, src.cols, i, j, "prewitt_v", 3);
		}
	}

	Mat res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
	if (display) {
		imshow("Display", res);
		waitKey();
	}
	imwrite(dirname + "//" + identifier + "_prewitt_3.jpg", res);
}




void tracker_callback(void* userData) {
	tracker_in obj = *(static_cast<tracker_in*>(userData));
	string name = obj.images[obj.idx];
	string filter_name = obj.filter_names[obj.filter_idx];
	string output_dir = obj.output_dir;

	int f = (obj.f <= 3) ? 3 : obj.f - 1 + (obj.f % 2);
	assert((f >= 3) && (f % 2 == 1));

	cout << "Image = " << name << " Filter name = " << filter_name << " Filter size = " << f << "\n";

	if (filter_name == "prewiit") {
		prewitt_filter(name, output_dir);
	}
	else {
		compute_filter(name, filter_name, f, output_dir);
	}
}


void image_callback(int slidervalue, void* userData) {
	tracker_callback(userData);
}

void filter_callback(int slidervalue, void* userData) {
	tracker_callback(userData);
}

void filter_name_callback(int slidervalue, void* userData) {
	tracker_callback(userData);
}




void spatial_filter(string input_dir, string output_dir) {
	tracker_in obj;
	for (const auto& file : filesystem::directory_iterator(input_dir)) {
		string filepath = file.path().string();
		obj.images.push_back(filepath);
	}
	obj.output_dir = output_dir;

	namedWindow("Display");
	createTrackbar("Image", "Display", &obj.idx, (int)obj.images.size() - 1, image_callback, &obj);
	createTrackbar("f", "Display", &obj.f, 9, filter_callback, &obj);
	createTrackbar("Filter", "Display", &obj.filter_idx, (int)obj.filter_names.size() - 1, filter_name_callback, &obj);
	waitKey();
}




vector<double> high_pass(vector<uchar> pixel_values, int h, int w) {
	vector<double> src_hp(pixel_values.size());
	
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			src_hp[i * h + j] = filter_vector(pixel_values, h, w, i, j, "high_pass", 3, false);
		}
	}

	return src_hp;
}


vector<double> adaptive_w(vector<double> src_hp, int h, int w, double sigma_r) {
	vector<double> Aw1;

	for (int i = 0; i < h * w; i++) {
		Aw1.push_back(exp(-abs(src_hp[i]) / (2 * pow(sigma_r, 2))));
	}

	vector<double> Aw;
	for (int i = 0; i < h * w; i++) {
		Aw.push_back(1 - Aw1[i]);
	}

	return Aw;
}


void adaptive_high_boost(string name, double b, double sigma_r, string dirname = ".") {
	string basename = name.substr(name.find_last_of("/\\") + 1);
	string identifier = basename.substr(0, basename.find_last_of("."));

	Mat src = imread(name, IMREAD_GRAYSCALE);
	imshow(identifier, src);
	waitKey();

	vector<uchar> pixel_values;
	pixel_values.assign(src.data, src.data + src.total());

	vector<double> src_hp = high_pass(pixel_values, src.rows, src.cols);
	vector<double> Aw = adaptive_w(src_hp, src.rows, src.cols, sigma_r);

	vector<uchar> filtered_val(src.total());
	for (int i = 0; i < src.total(); i++) {
		filtered_val[i] = pixel_values[i] + (int)(b * Aw[i] * src_hp[i]);
	}

	Mat res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
	imshow("Display", res);
	waitKey();
	imwrite(dirname + "//" + identifier + "_adaptive_high_boost_" + to_string(b) + "_" + to_string(sigma_r) + ".jpg", res);
}




void sharpening_filters(string input_dir, string output_dir) {
	for (const auto& file : filesystem::directory_iterator(input_dir)) {
		string filepath = file.path().string();

		vector<thread> threads;
		threads.push_back(thread(compute_filter, filepath, "sobel_h", 3, output_dir, false));
		threads.push_back(thread(compute_filter, filepath, "sobel_v", 3, output_dir, false));
		threads.push_back(thread(compute_filter, filepath, "sobel_d", 3, output_dir, false));
		threads.push_back(thread(prewitt_filter, filepath, output_dir, false));
		threads.push_back(thread(compute_filter, filepath, "laplacian", 3, output_dir, false));
		threads.push_back(thread(compute_filter, filepath, "log", 5, output_dir, false));
		cout << "Threads spawned for " << filepath << "\n";
		for (auto& th : threads) {
			th.join();
		}
	}
}




int main() {
	//compute_filter("lena_gray_512.jpg", "sobel_d", 3);
	//prewitt_filter("lena_gray_512.jpg");
	//spatial_filter("Noisy Images", "Filtered Images");
	adaptive_high_boost("jetplane.jpg", 0.4, 30);
	//sharpening_filters("Normal Images", "Filtered Images");
	return 0;
}