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


class histogram {
public:
    vector<vector<double>> hist;
    int total;
    vector<int> peak_freq;
};


histogram calculateHist(string name, string identifier, string dirname = ".") {
    Mat src = imread(name, IMREAD_COLOR);
    imshow(identifier, src);
    waitKey();

    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    vector<vector<uchar>> pixel_values(3);
    for (int k = 0; k < 3; k++) {
        pixel_values[k].assign(bgr_planes[k].data, bgr_planes[k].data + bgr_planes[k].total());
    }

    vector<vector<double>> hist(3, vector<double>(256, 0));
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < pixel_values[k].size(); i++) {
            hist[k][pixel_values[k][i]]++;
        }
    }

    vector<int> peak_freq(3, 0);
    for (int k = 0; k < 3; k++) {
        int sanity_check_sum = 0;
        for (int i = 0; i < hist[k].size(); i++) {
            sanity_check_sum += int(round(hist[k][i]));
            peak_freq[k] = max(peak_freq[k], int(round(hist[k][i])));
        }
        assert(sanity_check_sum == bgr_planes[k].total());
    }

    int hist_h = 512, hist_w = 512;
    int bin_w = cvRound((double)hist_w / hist[0].size());

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 1; i < hist[0].size(); i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[0][i - 1] / peak_freq[0])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[0][i] / peak_freq[0])), Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[1][i - 1] / peak_freq[1])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[1][i] / peak_freq[1])), Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[2][i - 1] / peak_freq[2])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[2][i] / peak_freq[2])), Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow(identifier + " Histogram", histImage);
    waitKey();
    imwrite(dirname + "//" + identifier + "_histogram.jpg", histImage);

    histogram source_hist;
    source_hist.hist = hist;
    source_hist.total = (int)bgr_planes[0].total();
    source_hist.peak_freq = peak_freq;

    return source_hist;
}




vector<double> equalize(vector<double> r, int total) {
    int L = 256;
    vector<double> s(r.size(), 0);
    for (int i = 0; i < r.size(); i++) {
        s[i] = (i == 0) ? r[i] / total : s[i - 1] + r[i] / total;
    }
    for (int i = 0; i < r.size(); i++) {
        s[i] *= (L - 1);
        s[i] = round(s[i]);
    }
    return s;
}




void histogram_equalize(string name, string dirname=".") {
    string basename = name.substr(name.find_last_of("/\\") + 1);
    string identifier = basename.substr(0, basename.find_last_of("."));
    histogram source_hist = calculateHist(name, identifier, dirname = dirname);
    histogram dest_hist;
    vector<vector<double>> transform(3);

    for (int k = 0; k < 3; k++) {
        transform[k] = equalize(source_hist.hist[k], source_hist.total);
    }
    
    vector<vector<double>> hist(3, vector<double>(256, 0));
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < transform[k].size(); i++) {
            hist[k][int(transform[k][i])] += source_hist.hist[k][i];
        }
    }

    vector<int> peak_freq(3, 0);
    for (int k = 0; k < 3; k++) {
        int sanity_check_sum = 0;
        for (int i = 0; i < hist[k].size(); i++) {
            sanity_check_sum += int(round(hist[k][i]));
            peak_freq[k] = max(peak_freq[k], int(round(hist[k][i])));
        }
        assert(sanity_check_sum == source_hist.total);
    }

    int hist_h = 512, hist_w = 512;
    int bin_w = cvRound((double)hist_w / hist[0].size());

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 1; i < hist[0].size(); i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[0][i - 1] / peak_freq[0])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[0][i] / peak_freq[0])), Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[1][i - 1] / peak_freq[1])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[1][i] / peak_freq[1])), Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(hist_h * hist[2][i - 1] / peak_freq[2])),
            Point(bin_w * (i), hist_h - cvRound(hist_h * hist[2][i] / peak_freq[2])), Scalar(0, 0, 255), 2, 8, 0);
    }

    imshow(identifier + " Equalized histogram", histImage);
    waitKey();
    imwrite(dirname + "//" + identifier + "_equalized_histogram.jpg", histImage);

    dest_hist.hist = hist;
    dest_hist.total = source_hist.total;
    dest_hist.peak_freq = source_hist.peak_freq;


    Mat src = imread(name, IMREAD_COLOR);
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    vector<vector<uchar>> src_pixel_values(3);
    for (int k = 0; k < 3; k++) {
        src_pixel_values[k].assign(bgr_planes[k].data, bgr_planes[k].data + bgr_planes[k].total());
    }

    vector<vector<uchar>> dest_pixel_values(3, vector<uchar>(src_pixel_values[0].size(), 0));
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < src_pixel_values[k].size(); i++) {
            dest_pixel_values[k][i] = int(round(transform[k][src_pixel_values[k][i]]));
        }
    }

    vector<Mat> dest_bgr_planes(3);
    for (int k = 0; k < 3; k++) {
        dest_bgr_planes[k] = Mat(src.rows, src.cols, CV_8UC1, dest_pixel_values[k].data());
    }
    
    Mat dest;
    merge(dest_bgr_planes, dest);
    imshow(identifier + " Equalized", dest);
    waitKey();
    imwrite(dirname + "//" + identifier + "_equalized.jpg", dest);
}




void histogram_match(string input_name, string target_name, string dirname = ".") {
    string in_basename = input_name.substr(input_name.find_last_of("/\\") + 1);
    string in_identifier = in_basename.substr(0, in_basename.find_last_of("."));
    histogram source_hist = calculateHist(input_name, in_identifier, dirname);

    string tar_basename = target_name.substr(target_name.find_last_of("/\\") + 1);
    string tar_identifier = tar_basename.substr(0, tar_basename.find_last_of("."));
    histogram target_hist = calculateHist(target_name, tar_identifier, dirname);

    vector<vector<double>> source_transform(3);
    vector<vector<double>> target_transform(3);

    for (int k = 0; k < 3; k++) {
        source_transform[k] = equalize(source_hist.hist[k], source_hist.total);
    }
    for (int k = 0; k < 3; k++) {
        target_transform[k] = equalize(target_hist.hist[k], target_hist.total);
    }

    vector<map<int, int>> inverse_target_transform(3);
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < target_transform[k].size(); i++) {
            int key = int(round(target_transform[k][i]));
            if (inverse_target_transform[k].find(key) == inverse_target_transform[k].end()) {
                inverse_target_transform[k][key] = i;
            }
            else {
                inverse_target_transform[k][key] = min(i, inverse_target_transform[k][key]);
            }
        }
    }

    Mat src = imread(input_name, IMREAD_COLOR);
    vector<Mat> bgr_planes;
    split(src, bgr_planes);

    vector<vector<uchar>> src_pixel_values(3);
    for (int k = 0; k < 3; k++) {
        src_pixel_values[k].assign(bgr_planes[k].data, bgr_planes[k].data + bgr_planes[k].total());
    }

    vector<vector<uchar>> dest_pixel_values(3, vector<uchar>(src_pixel_values[0].size(), 0));
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < src_pixel_values[k].size(); i++) {
            int si = int(round(source_transform[k][src_pixel_values[k][i]]));
            auto it = inverse_target_transform[k].lower_bound(si);
            int pixel_val = (*it).second;
            if (it != inverse_target_transform[k].begin()) {
                auto prev_it = prev(it);
                int prev_pixel_val = (*prev_it).second;
                pixel_val = (abs((*it).first - si) <= abs((*prev_it).first - si)) ? pixel_val : prev_pixel_val;
            }
            dest_pixel_values[k][i] = pixel_val;
        }
    }

    vector<Mat> dest_bgr_planes(3);
    for (int k = 0; k < 3; k++) {
        dest_bgr_planes[k] = Mat(src.rows, src.cols, CV_8UC1, dest_pixel_values[k].data());
    }

    Mat dest;
    merge(dest_bgr_planes, dest);
    string identifier = in_identifier + "_" + tar_identifier;
    imshow(identifier + " matched image", dest);
    waitKey();
    imwrite(dirname + "//" + identifier + "_matched.jpg", dest);

    calculateHist(dirname + "//" + identifier + "_matched.jpg", identifier, dirname);
}




void equalize_directory(string path, string output_path) {
    for (const auto& file : filesystem::directory_iterator(path)) {
        string filepath = file.path().string();
        histogram_equalize(filepath, output_path);
    }
}




void match_directories(string input_dir, string target_dir, string output_path) {
    for (const auto& infile : filesystem::directory_iterator(input_dir)) {
        string in_filepath = infile.path().string();
        for (const auto& tarfile : filesystem::directory_iterator(target_dir)) {
            string tar_filepath = tarfile.path().string();
            histogram_match(in_filepath, tar_filepath, output_path);
        }
    }
}




int main()
{
    //histogram_equalize("lena_color_512.jpg");
    //histogram_match("lena_color_512.jpg", "mandril_color.jpg");
    //equalize_directory("Equalization Input","Equalization Output");
    //match_directories("Match Input", "match Target", "Match Output");
    return 0;
}
