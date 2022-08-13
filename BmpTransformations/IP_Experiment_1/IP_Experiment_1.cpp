#define _USE_MATH_DEFINES
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
using namespace cv;
using namespace std;


class BMP {
public:
	unsigned char* data;
	unsigned char* info;
	int height;
	int width;
	int nb;
	int offset;
	int file_size;
};

void writeBMP(string name, unsigned char* header, unsigned char* pixel_arr, int height, int width,
	int offset, int bitwidth, int fileSize);
BMP rotateBMP(BMP input_image, double theta);
unsigned char* interpolate(unsigned char* data, int height, int width, int nb);


unsigned char* generate_color_img(unsigned char* color_p, unsigned char* data, int height, int width, int pal_s) {
	unsigned char* out = new unsigned char[3 * height * width];
	for (int i = 0; i < height * width; i++) {
		out[3 * i] = color_p[4 * data[i]];
		out[3 * i + 1] = color_p[4 * data[i] + 1];
		out[3 * i + 2] = color_p[4 * data[i] + 2];
	}
	return out;
}


bool isBackground(unsigned char* data, int i, int j, int height, int width, int nb) {
	for (int k = 0; k < nb; k++) {
		if (data[i * width * nb + j * nb + k] != 0) {
			return false;
		}
	}
	return true;
}


bool check(int i, int j, int height, int width) {
	if (i >= 0 && j >= 0 && i < height && j < width) {
		return true;
	}
	else return false;
}


vector<unsigned char> linear_interpolate(unsigned char* data, int i, int j, int height, int width, int nb) {
	vector<unsigned char> colors(nb);
	int dx[] = { 1,0,-1,0 };
	int dy[] = { 0,1,0,-1 };
	int num_neighbours = 0;
	int r = 0, b = 0, g = 0;
	for (int k = 0; k < 4; k++) {
		int ni = i + dx[k];
		int nj = j + dy[k];
		if (check(ni, nj, height, width)) {
			num_neighbours++;
			if (nb == 3) {
				b += data[ni * width * 3 + nj * 3];
				g += data[ni * width * 3 + nj * 3 + 1];
				r += data[ni * width * 3 + nj * 3 + 2];
			}
			else if (nb == 1) {
				b += data[ni * width + nj];
				g = b; r = b;
			}
		}
	}
	if (num_neighbours) {
		r /= num_neighbours;
		b /= num_neighbours;
		g /= num_neighbours;
	}
	if (nb == 3) {
		colors[0] = b; colors[1] = g; colors[2] = r;
	}
	else if (nb == 1) {
		colors[0] = b;
	}
	return colors;
}




BMP readBMP(const char* filename) {
	FILE* f = fopen(filename, "rb");
	if (f == NULL) {
		cout << "No file\n";
		cout << "Current working directory: " << filesystem::current_path() << endl;
		exit(0);
	}

	unsigned char* info = new unsigned char[54];
	fread(info, sizeof(unsigned char), 54, f);

	if (info[0] != 'B' || info[1] != 'M') {
		cout << "Not a bmp file\n";
		exit(0);
	}

	int height, width, _bit_width, file_size, offset;
	memcpy(&width, info + 18, sizeof(int));
	memcpy(&height, info + 22, sizeof(int));
	memcpy(&_bit_width, info + 28, sizeof(int));
	memcpy(&file_size, info + 2, sizeof(int));
	memcpy(&offset, info + 10, sizeof(int));
	bool flipped = height > 0;
	height = abs(height);

	cout << "width  = " << width << "\n";
	cout << "height  = " << height << "\n";
	cout << "bit width  = " << _bit_width << "\n";
	cout << "file size in bytes  = " << file_size << "\n";
	cout << "offset size  = " << offset << "\n";

	unsigned char* color_p = new unsigned char[10];

	if (offset > 54) {
		color_p = new unsigned char[offset - 54];
		fread(color_p, sizeof(unsigned char), offset - 54, f);
	}

	int nb;
	nb = _bit_width / 8;

	int data_size = nb * height * width;
	unsigned char* data = new unsigned char[data_size];
	int row_padded = ceil(float(width * _bit_width) / 32) * 4;
	unsigned char* row = new unsigned char[row_padded];

	if (flipped) {
		for (int i = height - 1; i >= 0; i--) {
			fread(row, sizeof(unsigned char), row_padded, f);
			for (int j = 0; j < width * nb; j++) {
				data[i * width * nb + j] = row[j];
			}
		}
	}
	else {
		for (int i = 0; i < height; i++) {
			fread(row, sizeof(unsigned char), row_padded, f);
			for (int j = 0; j < width * nb; j++) {
				data[i * width * nb + j] = row[j];
			}
		}
	}
	fclose(f);

	if (nb == 3) {
		Mat image = Mat(height, width, CV_8UC3, data);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", image);
		waitKey();
	}
	else if (nb == 1) {
		unsigned char* new_d = generate_color_img(color_p, data, height, width, offset - 54);
		data = new_d;
		nb = 3;
		offset = 54;
		file_size = 3 * height * width + 54;
		Mat image = Mat(height, width, CV_8UC3, new_d);
		namedWindow("image", WINDOW_AUTOSIZE);
		imshow("image", image);
		waitKey();
	}
	

	BMP input_image;
	input_image.data = data;
	input_image.info = info;
	input_image.height = height;
	input_image.width = width;
	input_image.nb = nb;
	input_image.file_size = file_size;
	input_image.offset = offset;

	return input_image;
}




void modifyBMP(BMP input_image, string prefix) {
	unsigned char* data = input_image.data;
	unsigned char* info = input_image.info;
	int height = input_image.height;
	int width = input_image.width;
	int nb = input_image.nb;
	int offset = input_image.offset;

	unsigned char* greydata = new unsigned char[height * width];
	for (int i = 0, j = 0; i < 3 * height * width; i+=3, j++) {
		greydata[j] = (data[i] + data[i + 1] + data[i + 2]) / 3;
	}
	Mat greyImg = Mat(height, width, CV_8UC1, greydata);
	namedWindow("Grey Image", WINDOW_AUTOSIZE);
	imshow("Grey Image", greyImg);
	waitKey();
	writeBMP(prefix+"_grey.bmp", info, greydata, height, width, offset, 8, 54 + height * width);

	unsigned char* flipped_color = new unsigned char[nb * height * width];
	int n_width = height;
	int n_height = width;
	int x_dash, y_dash;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			x_dash = j;
			y_dash = i;
			for (int k = 0; k < nb; k++) {
				flipped_color[x_dash * n_width * nb + y_dash * nb + k] = data[i * width * nb + j * nb + k];
			}
		}
	}
	Mat flipColorImg = Mat(n_height, n_width, CV_8UC3, flipped_color);
	namedWindow("Flipped Color Image", WINDOW_AUTOSIZE);
	imshow("Flipped Color Image", flipColorImg);
	waitKey();
	writeBMP(prefix + "_flipped.bmp", info, flipped_color, n_height, n_width, offset, 24, 54 + nb * n_height * n_width);

	unsigned char* flipped_grey = new unsigned char[height * width];
	n_width = height;
	n_height = width;
	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j ++) {
			x_dash = j;
			y_dash = i;
			flipped_grey[x_dash * n_width + y_dash] = greydata[i * width + j];
		}
	}
	Mat flipGreyImg = Mat(n_height, n_width, CV_8UC1, flipped_grey);
	namedWindow("Flipped Grey Image", WINDOW_AUTOSIZE);
	imshow("Flipped Grey Image", flipGreyImg);
	waitKey();
	writeBMP(prefix + "_grey_flipped.bmp", info, flipped_grey, n_height, n_width, offset, 8, 54 + height * width);

	int sf = 2;
	unsigned char* scaled_color = new unsigned char[sf * sf * 3 * height * width]();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int xdash = sf * i;
			int ydash = sf * j;
			scaled_color[xdash * sf * width * 3 + 3 * ydash] = data[i * width * 3 + 3 * j];
			scaled_color[xdash * sf * width * 3 + 3 * ydash + 1] = data[i * width * 3 + 3 * j + 1];
			scaled_color[xdash * sf * width * 3 + 3 * ydash + 2] = data[i * width * 3 + 3 * j + 2];
		}
	}
	scaled_color = interpolate(scaled_color, sf * height, sf * width, 3);
	Mat scaledColorImg = Mat(sf * height, sf * width, CV_8UC3, scaled_color);
	namedWindow("Scaled Color Image", WINDOW_AUTOSIZE);
	imshow("Scaled Color Image", scaledColorImg);
	waitKey();
	writeBMP(prefix + "_scaled.bmp", info, scaled_color, sf * height, sf * width, offset, 24, 54 + 3 * sf * sf * height * width);

	unsigned char* scaled_grey = new unsigned char[sf * sf * height * width]();
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			int xdash = sf * i;
			int ydash = sf * j;
			scaled_grey[xdash * sf * width + ydash] = greydata[i * width + j];
		}
	}
	scaled_grey = interpolate(scaled_grey, sf * height, sf * width, 1);
	Mat scaledGreyImg = Mat(sf * height, sf * width, CV_8UC1, scaled_grey);
	namedWindow("Scaled Grey Image", WINDOW_AUTOSIZE);
	imshow("Scaled Grey Image", scaledGreyImg);
	waitKey();
	writeBMP(prefix + "_grey_scaled.bmp", info, scaled_grey, sf * height, sf * width, offset, 8, 54 + sf * sf * height * width);

	BMP rot1_color = rotateBMP(input_image, M_PI_2);
	Mat rot1_ColorImg = Mat(rot1_color.height, rot1_color.width, CV_8UC3, rot1_color.data);
	namedWindow("Rotated 90 Color Image", WINDOW_AUTOSIZE);
	imshow("Rotated 90 Color Image", rot1_ColorImg);
	waitKey();
	writeBMP(prefix + "_rotated_pi_2.bmp", info, rot1_color.data, rot1_color.height, rot1_color.width, 
		offset, 24, 54 + 3 * rot1_color.height * rot1_color.width);

	BMP rot2_color = rotateBMP(input_image, M_PI_4);
	Mat rot2_ColorImg = Mat(rot1_color.height, rot1_color.width, CV_8UC3, rot2_color.data);
	namedWindow("Rotated 45 Color Image", WINDOW_AUTOSIZE);
	imshow("Rotated 45 Color Image", rot2_ColorImg);
	waitKey();
	writeBMP(prefix + "_rotated_pi_4.bmp", info, rot2_color.data, rot1_color.height, rot1_color.width, 
		offset, 24, 54 + 3 * rot1_color.height * rot1_color.width);

	BMP grey_input = input_image;
	grey_input.data = greydata;
	grey_input.nb = 1;

	BMP rot1_grey = rotateBMP(grey_input, M_PI_2);
	Mat rot1_GreyImg = Mat(rot1_grey.height, rot1_grey.width, CV_8UC1, rot1_grey.data);
	namedWindow("Rotated 90 Grey Image", WINDOW_AUTOSIZE);
	imshow("Rotated 90 Grey Image", rot1_GreyImg);
	waitKey();
	writeBMP(prefix + "_grey_rotated_pi_2.bmp", info, rot1_grey.data, rot1_grey.height, rot1_grey.width, 
		offset, 8, 54 + rot1_grey.height * rot1_grey.width);

	BMP rot2_grey = rotateBMP(grey_input, M_PI_4);
	Mat rot2_GreyImg = Mat(rot1_grey.height, rot1_grey.width, CV_8UC1, rot2_grey.data);
	namedWindow("Rotated 45 Grey Image", WINDOW_AUTOSIZE);
	imshow("Rotated 45 Grey Image", rot2_GreyImg);
	waitKey();
	writeBMP(prefix + "_grey_rotated_pi_4.bmp", info, rot2_grey.data, rot1_grey.height, rot1_grey.width, 
		offset, 8, 54 + rot1_grey.height * rot1_grey.width);
}




BMP rotateBMP(BMP input_image, double theta) {
	int width = input_image.width;
	int height = input_image.height;
	int nb = input_image.nb;

	int n_width = ceil(sqrt(width * width + height * height));
	int n_height = n_width;

	BMP rot_image;
	rot_image.data = new unsigned char[nb * n_height * n_width]();
	memset(rot_image.data, sizeof(unsigned char), 0);
	rot_image.height = n_height;
	rot_image.width = n_width;
	rot_image.offset = input_image.offset;
	rot_image.nb = nb;
	rot_image.info = input_image.info;
	rot_image.file_size = 54 + nb * n_height * n_width;

	for (int i = 0; i < height; i ++) {
		for (int j = 0; j < width; j ++) {
			double x = double(i - height / 2.0);
			double y = double(j - width / 2.0);
			int x_dash = int(x * cos(theta) + y * (-sin(theta)) + rot_image.height / 2);
			int y_dash = int(x * sin(theta) + y * cos(theta) + rot_image.width / 2);
			for (int k = 0; k < nb; k++) {
				if (x_dash < 0) {
					x_dash = 0;
				}
				if (y_dash < 0) {
					y_dash = 0;
				}
				rot_image.data[x_dash * rot_image.width * nb + y_dash * nb + k] = input_image.data[i * width * nb + j * nb + k];
			}
		}
	}

	rot_image.data = interpolate(rot_image.data, rot_image.height, rot_image.width, rot_image.nb);

	return rot_image;
}




void writeBMP(string name, unsigned char* header, unsigned char* pixel_arr, 
	int height, int width, int offset, int bitwidth, int fileSize) {
	const char* filename = name.c_str();
	FILE* out_f = fopen(filename, "wb");

	if (bitwidth == 8) {
		unsigned char* temp = new unsigned char[3 * height * width];
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				temp[i * width * 3 + j * 3] = pixel_arr[i * width + j];
				temp[i * width * 3 + j * 3 + 1] = pixel_arr[i * width + j];
				temp[i * width * 3 + j * 3 + 2] = pixel_arr[i * width + j];
			}
		}
		pixel_arr = temp;
		bitwidth = 24;
	}

	header[2] = fileSize;
	header[3] = fileSize >> 8;
	header[4] = fileSize >> 16;
	header[5] = fileSize >> 24;

	header[10] = offset;
	header[11] = offset >> 8;
	header[12] = offset >> 16;
	header[13] = offset >> 24;

	header[18] = width;
	header[19] = width >> 8;
	header[20] = width >> 16;
	header[21] = width >> 24;

	header[22] = height;
	header[23] = height >> 8;
	header[24] = height >> 16;
	header[25] = height >> 24;

	header[28] = bitwidth;
	header[29] = bitwidth >> 8;

	fwrite(header, 1, offset, out_f);

	int row_padded = ceil(float(width * bitwidth) / 32) * 4;
	int paddingSize = row_padded - (width * bitwidth/8);
	unsigned char* padding = new unsigned char[1];
	if (paddingSize > 0) {
		padding = new unsigned char[paddingSize];
		memset(padding, 0, sizeof(unsigned char));
	}

	for (int i = height - 1; i >= 0; i--) {
		fwrite(pixel_arr + (i * width * bitwidth/8), bitwidth/8, width, out_f);
		if (paddingSize > 0) {
			fwrite(padding, 1, paddingSize, out_f);
		}
	}
}




void colorChannel (BMP input_image, string prefix) {
	int height = input_image.height;
	int width = input_image.width;
	unsigned char* data = input_image.data;
	unsigned char* info = input_image.info;
	int nb = input_image.nb;
	int offset = input_image.offset;

	unsigned char* zeroR = new unsigned char[3 * height * width];
	for (int i = 0; i < 3 * height; i += 3) {
		for (int j = 0; j < 3 * width; j += 3) {
			zeroR[(i / 3) * width * 3 + j] = data[(i / 3) * width * 3 + j];
			zeroR[(i / 3) * width * 3 + j + 1] = data[(i / 3) * width * 3 + j + 1];
			zeroR[(i / 3) * width * 3 + j + 2] = 0;
		}
	}
	Mat zeroRImg = Mat(height, width, CV_8UC3, zeroR);
	namedWindow("Zero Red Image", WINDOW_AUTOSIZE);
	imshow("Zero Red Image", zeroRImg);
	waitKey();
	writeBMP(prefix + "_zeroR.bmp", info, zeroR, height, width, offset, 24, 54 + 3 * height * width);

	unsigned char* zeroB = new unsigned char[3 * height * width];
	for (int i = 0; i < 3 * height; i += 3) {
		for (int j = 0; j < 3 * width; j += 3) {
			zeroB[(i / 3) * width * 3 + j] = 0;
			zeroB[(i / 3) * width * 3 + j + 1] = data[(i / 3) * width * 3 + j + 1];
			zeroB[(i / 3) * width * 3 + j + 2] = data[(i / 3) * width * 3 + j + 2];
		}
	}
	Mat zeroBImg = Mat(height, width, CV_8UC3, zeroB);
	namedWindow("Zero Blue Image", WINDOW_AUTOSIZE);
	imshow("Zero Blue Image", zeroBImg);
	waitKey();
	writeBMP(prefix + "_zeroB.bmp", info, zeroB, height, width, offset, 24, 54 + 3 * height * width);

	unsigned char* zeroG = new unsigned char[3 * height * width];
	for (int i = 0; i < 3 * height; i += 3) {
		for (int j = 0; j < 3 * width; j += 3) {
			zeroG[(i / 3) * width * 3 + j] = data[(i / 3) * width * 3 + j];
			zeroG[(i / 3) * width * 3 + j + 1] = 0;
			zeroG[(i / 3) * width * 3 + j + 2] = data[(i / 3) * width * 3 + j + 2];
		}
	}
	Mat zeroGImg = Mat(height, width, CV_8UC3, zeroG);
	namedWindow("Zero Green Image", WINDOW_AUTOSIZE);
	imshow("Zero Green Image", zeroGImg);
	waitKey();
	writeBMP(prefix + "_zeroG.bmp", info, zeroG, height, width, offset, 24, 54 + 3 * height * width);
}




unsigned char* interpolate(unsigned char* data, int height, int width, int nb) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (isBackground(data, i, j, height, width, nb)) {
				vector<unsigned char> colors = linear_interpolate(data, i, j, height, width, nb);
				for (int k = 0; k < nb; k++) {
					data[i * width * nb + j * nb + k] = colors[k];
				}
			}
		}
	}
	return data;
}




int main() {
	BMP input_image = readBMP("lena_colored_256.bmp");
	modifyBMP(input_image, "lena");
	colorChannel(input_image, "lena");
	

	BMP corn = readBMP("corn.bmp");
	modifyBMP(corn, "corn");
	colorChannel(corn, "corn");

	BMP camera = readBMP("cameraman.bmp");
	modifyBMP(camera, "camera");
	colorChannel(camera, "camera");
	return 0;
}