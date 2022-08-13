#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <map>
#include <filesystem>
#include <cmath>
using namespace cv;
using namespace std;

using cd = complex<double>;

const double PI = acos(-1);

class tracker_in {
public:
    vector<string> images;
    int idx;
    vector<string> filter_names;
    int filter_idx;
    int cutoff;
    string output_dir;

    tracker_in() {
        this->idx = 0;
        this->cutoff = 0;
        this->filter_names = { "LPF", "HPF", "gaussian_LPF", "gaussian_HPF", "butterworth_LPF", "butterworth_HPF" };
        this->filter_idx = 0;
        this->output_dir = ".";
    }
};




void fft(vector<cd>& a) {
    int n = a.size();
    if (n == 1) {
        return;
    }

    vector<cd> a0(n / 2), a1(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        a0[i] = a[2 * i];
        a1[i] = a[2 * i + 1];
    }

    fft(a0);
    fft(a1);

    double ang = 2 * PI / n;
    cd w(1), wn(cos(ang), -sin(ang));
    for (int i = 0; 2 * i < n; i++) {
        a[i] = a0[i] + w * a1[i];
        a[i + n / 2] = a0[i] - w * a1[i];
        w *= wn;
    }
}




vector<cd> forward_fft(vector<cd> input, int h, int w) {
    vector<cd> result;

    int p = 1;
    if (!(w && !(w & (w - 1)))) {
        while (p < w) {
            p <<= 1;
        }
    }
    else {
        p = w;
    }
    vector<cd> temp;

    for (int i = 0; i < h; i++) {
        temp.assign(p, 0);
        copy(input.begin() + i * w, input.begin() + i * w + w, temp.begin());
        
        fft(temp);

        for (int k = 0; k < w; k++) {
            result.push_back(temp[k]);
        }
    }

    if (!(h && !(h & (h - 1)))) {
        while (p < h) {
            p <<= 1;
        }
    }
    else {
        p = h;
    }
    for (int j = 0; j < w; j++) {
        temp.assign(p, 0);

        for (int i = 0; i < h; i++) {
            temp[i] = result[i * w + j];
        }

        fft(temp);
        for (int i = 0; i < h; i++) {
            result[i * w + j] = temp[i];
        }
    }

    return result;
}




vector<cd> ideal_LPF(vector<cd> input, int h, int w, double cutoff) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            if (d <= cutoff) {
                result[i * w + j] = input[i * w + j];
                filter[i * w + j] = 1;
            }
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * int(abs(x)));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}


vector<cd> ideal_HPF(vector<cd> input, int h, int w, double cutoff) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            if (d > cutoff) {
                result[i * w + j] = input[i * w + j];
                filter[i * w + j] = 1;
            }
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * int(abs(x)));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}


vector<cd> gaussian_LPF(vector<cd> input, int h, int w, double cutoff) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            filter[i * w + j] = exp(-(d * d / (2 * cutoff)));
            result[i * w + j] = input[i * w + j] * exp(-(d * d / (2 * cutoff)));
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * int(abs(x)));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}


vector<cd> gaussian_HPF(vector<cd> input, int h, int w, double cutoff) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            filter[i * w + j] = 1 - exp(-(d * d / (2 * cutoff)));
            result[i * w + j] = input[i * w + j] * filter[i * w + j];
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * max(0, int(x.real())));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}


vector<cd> butterworth_LPF(vector<cd> input, int h, int w, double cutoff, int order = 1) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            filter[i * w + j] = 1 / (1 + pow((d / cutoff), 2 * order));
            result[i * w + j] = input[i * w + j] * filter[i * w + j];
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * max(0, int(x.real())));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}


vector<cd> butterworth_HPF(vector<cd> input, int h, int w, double cutoff, int order = 1) {
    vector<cd> result(h * w, 0);
    vector<cd> filter(h * w, 0);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            double d = sqrt((i - h / 2) * (i - h / 2) + (j - w / 2) * (j - w / 2));
            filter[i * w + j] = 1 - (1 / (1 + pow((d / cutoff), 2 * order)));
            result[i * w + j] = input[i * w + j] * filter[i * w + j];
        }
    }

    vector<uchar> filter_spectrum;
    for (auto x : filter) {
        filter_spectrum.push_back(255 * max(0, int(x.real())));
    }
    Mat res = Mat(h, w, CV_8UC1, filter_spectrum.data());
    imshow("Filter Spectrum", res);
    waitKey();

    return result;
}




void frequency_transform(string name, string dirname = ".") {
    string basename = name.substr(name.find_last_of("/\\") + 1);
    string identifier = basename.substr(0, basename.find_last_of("."));

    Mat src = imread(name, IMREAD_GRAYSCALE);
    imshow(identifier, src);
    waitKey();

    vector<uchar> pixel_values;
    pixel_values.assign(src.data, src.data + src.total());

    vector<cd> input;
    input.assign(pixel_values.data(), pixel_values.data() + src.total());

    int h = src.rows;
    int w = src.cols;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            input[i * w + j] *= ((i + j) % 2 == 0) ? 1 : -1;
        }
    }

    vector<cd> filtered_input = forward_fft(input, h, w);

    vector<cd> inverse_filtered_input;
    for (auto x : filtered_input) {
        inverse_filtered_input.push_back(conj(x));
    }

    for (int i = 0; i < h * w; i++) {
        inverse_filtered_input[i] = conj(inverse_filtered_input[i]);
    }
    inverse_filtered_input = forward_fft(inverse_filtered_input, h, w);
    for (int i = 0; i < h * w; i++) {
        inverse_filtered_input[i] = conj(inverse_filtered_input[i]) / (1.0 * h * w);
    }

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if ((i + j) % 2 != 0) {
                inverse_filtered_input[i] = -conj(inverse_filtered_input[i]);
            }
        }
    }


    vector<uchar> filtered_val;
    for (auto x : inverse_filtered_input) {
        filtered_val.push_back(int(abs(x)));
    }

    Mat res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
    imshow("Display", res);
    waitKey();
    imwrite(dirname + "//" + identifier + ".jpg", res);
}




void filter_image(string name, string filter_name, double cutoff, string dirname = ".") {
    string basename = name.substr(name.find_last_of("/\\") + 1);
    string identifier = basename.substr(0, basename.find_last_of("."));

    Mat src = imread(name, IMREAD_GRAYSCALE);
    imshow(identifier, src);
    waitKey();

    Mat src_padded;
    copyMakeBorder(src, src_padded, 0, src.rows, 0, src.cols, BORDER_CONSTANT, 0);

    src = src_padded;

    vector<uchar> pixel_values;
    pixel_values.assign(src.data, src.data + src.total());

    vector<cd> input;
    input.assign(pixel_values.data(), pixel_values.data() + src.total());

    int h = src.rows;
    int w = src.cols;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            input[i * w + j] *= ((i + j) % 2 == 0) ? 1 : -1;
        }
    }

    vector<cd> filtered_input = forward_fft(input, h, w);

    vector<uchar> input_spectrum;
    for (auto x : filtered_input) {
        input_spectrum.push_back(max(0.0, x.real()));
    }

    Mat res = Mat(src.rows, src.cols, CV_8UC1, input_spectrum.data());
    imshow("Input Spectrum", res);
    waitKey();

    if (filter_name == "LPF") {
        filtered_input = ideal_LPF(filtered_input, h, w, cutoff);
    }
    else if (filter_name == "HPF") {
        filtered_input = ideal_HPF(filtered_input, h, w, cutoff);
    }
    else if (filter_name == "gaussian_LPF") {
        filtered_input = gaussian_LPF(filtered_input, h, w, cutoff);
    }
    else if (filter_name == "gaussian_HPF") {
        filtered_input = gaussian_HPF(filtered_input, h, w, cutoff);
    }
    else if (filter_name == "butterworth_LPF") {
        filtered_input = butterworth_LPF(filtered_input, h, w, cutoff);
    }
    else if (filter_name == "butterworth_HPF") {
        filtered_input = butterworth_HPF(filtered_input, h, w, cutoff);
    }
    else {
        cout << "Filter not implemented\n";
        return;
    }

    vector<uchar> filtered_spectrum;
    for (auto x : filtered_input) {
        filtered_spectrum.push_back(max(0.0, x.real()));
    }

    res = Mat(src.rows, src.cols, CV_8UC1, filtered_spectrum.data());
    imshow("Filtered Output Spectrum", res);
    waitKey();

    vector<cd> inverse_filtered_input;
    for (auto x : filtered_input) {
        inverse_filtered_input.push_back(conj(x));
    }
    inverse_filtered_input = forward_fft(inverse_filtered_input, h, w);
    for (int i = 0; i < h * w; i++) {
        inverse_filtered_input[i] = conj(inverse_filtered_input[i]) / (1.0 * h * w);
    }

    vector<double> result(h * w);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            result[i * w + j] = inverse_filtered_input[i * w + j].real();
            result[i * w + j] *= ((i + j) % 2 == 0) ? 1 : -1;
        }
    }

    vector<uchar> filtered_val;
    for (auto x : result) {
        filtered_val.push_back(int(x));
    }

    res = Mat(src.rows, src.cols, CV_8UC1, filtered_val.data());
    Mat res_cropped = res(Rect(0, 0, h / 2, w / 2));
    imshow("Filtered Image", res_cropped);
    waitKey();
    imwrite(dirname + "//" + identifier + "_" + filter_name + "_" + to_string(cutoff) + ".jpg", res_cropped);
}




void tracker_callback(void* userData) {
    tracker_in obj = *(static_cast<tracker_in*>(userData));
    string name = obj.images[obj.idx];
    string filter_name = obj.filter_names[obj.filter_idx];
    string output_dir = obj.output_dir;

    int cutoff = obj.cutoff;

    cout << "Image = " << name << " Filter name = " << filter_name << " Cutoff = " << cutoff << "\n";

    filter_image(name, filter_name, cutoff, output_dir);
}


void image_callback(int slidervalue, void* userData) {
    tracker_callback(userData);
}

void cutoff_callback(int slidervalue, void* userData) {
    tracker_callback(userData);
}

void filter_name_callback(int slidervalue, void* userData) {
    tracker_callback(userData);
}


void interactive_frequency_filter(string input_dir, string output_dir) {
    tracker_in obj;
    for (const auto& file : filesystem::directory_iterator(input_dir)) {
        string filepath = file.path().string();
        obj.images.push_back(filepath);
    }
    obj.output_dir = output_dir;

    namedWindow("Display");
    createTrackbar("Image", "Display", &obj.idx, (int)obj.images.size() - 1, image_callback, &obj);
    createTrackbar("Cutoff", "Display", &obj.cutoff, 512, cutoff_callback, &obj);
    createTrackbar("Filter", "Display", &obj.filter_idx, (int)obj.filter_names.size() - 1, filter_name_callback, &obj);
    waitKey();
}




int main()
{
    //frequency_transform("dip.tif");
    //filter_image("walkbridge.jpg", "butterworth_HPF", 5);
    interactive_frequency_filter("Input Images", "Filtered Images");
}