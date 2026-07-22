#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#define WINDOW_SIZE 5
#define STEP_SIZE 10
#define MIN_EIGENVALUE 1.0e-4f

using namespace cv;

static Mat loadImage(const std::string& path, int flags)
{
    Mat image = imread(path, flags);
    if (!image.empty()) {
        return image;
    }

    return imread("test/" + path, flags);
}

int main()
{
    // read img
    Mat curr_frame = loadImage("data/img/scene00140.png", IMREAD_GRAYSCALE);
    Mat curr_frame_color = loadImage("data/img/scene00140.png", IMREAD_COLOR);
    Mat next_frame = loadImage("data/img/scene00141.png", IMREAD_GRAYSCALE);

    if (curr_frame.empty() || curr_frame_color.empty() || next_frame.empty()) {
        std::cerr << "Failed to load input images." << std::endl;
        return 1;
    }

    if (curr_frame.size() != next_frame.size()) {
        std::cerr << "Input images must have the same size." << std::endl;
        return 1;
    }

    // get img size
    int width = curr_frame.cols;
    int height = curr_frame.rows;

    // calc diff between curr_frame, next_frame
    Mat df_dy = Mat::zeros(height, width, CV_32F);
    Mat df_dx = Mat::zeros(height, width, CV_32F);
    Mat df_dt = Mat::zeros(height, width, CV_32F);

    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            df_dy.at<float>(y, x) = static_cast<float>(curr_frame.at<uchar>(y + 1, x)) - curr_frame.at<uchar>(y, x);
            df_dx.at<float>(y, x) = static_cast<float>(curr_frame.at<uchar>(y, x + 1)) - curr_frame.at<uchar>(y, x);
            df_dt.at<float>(y, x) = static_cast<float>(next_frame.at<uchar>(y, x)) - curr_frame.at<uchar>(y, x);
        }
    }

    // for all pixels
    int half_window = WINDOW_SIZE / 2;
    for (int y = half_window; y < height - half_window; y += STEP_SIZE) {
        for (int x = half_window; x < width - half_window; x += STEP_SIZE) {
            int index = 0;
            Mat matrix_A = Mat::zeros(WINDOW_SIZE * WINDOW_SIZE, 2, df_dt.type());
            Mat matrix_B = Mat::zeros(WINDOW_SIZE * WINDOW_SIZE, 1, df_dt.type());
            Mat motion_vectors = Mat::zeros(2, 1, df_dt.type());
            // adapt lucas-kanade algorithm
            for (int window_y = y - half_window; window_y <= y + half_window; window_y++) {
                for (int window_x = x - half_window; window_x <= x + half_window; window_x++) {

                    matrix_A.at<float>(index, 0) = df_dx.at<float>(window_y, window_x);
                    matrix_A.at<float>(index, 1) = df_dy.at<float>(window_y, window_x);
                    matrix_B.at<float>(index, 0) = -df_dt.at<float>(window_y, window_x);
                    index += 1;
                }
            }

            Mat normal_matrix = matrix_A.t() * matrix_A;
            Mat eigenvalues;
            eigen(normal_matrix, eigenvalues);
            if (eigenvalues.at<float>(1) < MIN_EIGENVALUE) {
                continue;
            }

            solve(matrix_A, matrix_B, motion_vectors, DECOMP_SVD);

            Point start(x, y);
            Point end(cvRound(x + motion_vectors.at<float>(0)), cvRound(y + motion_vectors.at<float>(1)));
            arrowedLine(curr_frame_color, start, end, Scalar(0, 0, 255), 1);
        }
    }

    // make window for display output
    namedWindow("image", WINDOW_AUTOSIZE);

    imshow("image", curr_frame_color);

    waitKey(0);
}
