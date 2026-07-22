#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#define WINDOW_SIZE 21
#define STEP_SIZE 30
#define MIN_EIGENVALUE 0.05f
#define MIN_EIGEN_RATIO 0.03f
#define MIN_MOTION 0.15f
#define MAX_MOTION 8.0f
#define DISPLAY_SCALE 3.0f

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
    Mat curr_frame_u8 = loadImage("data/img/scene00140.png", IMREAD_GRAYSCALE);
    Mat curr_frame_color = loadImage("data/img/scene00140.png", IMREAD_COLOR);
    Mat next_frame_u8 = loadImage("data/img/scene00141.png", IMREAD_GRAYSCALE);

    if (curr_frame_u8.empty() || curr_frame_color.empty() || next_frame_u8.empty()) {
        std::cerr << "Failed to load input images." << std::endl;
        return 1;
    }

    if (curr_frame_u8.size() != next_frame_u8.size()) {
        std::cerr << "Input images must have the same size." << std::endl;
        return 1;
    }

    Mat curr_frame;
    Mat next_frame;
    curr_frame_u8.convertTo(curr_frame, CV_32F, 1.0 / 255.0);
    next_frame_u8.convertTo(next_frame, CV_32F, 1.0 / 255.0);

    // get img size
    int width = curr_frame.cols;
    int height = curr_frame.rows;

    // calc diff between curr_frame, next_frame
    Mat df_dy = Mat::zeros(height, width, CV_32F);
    Mat df_dx = Mat::zeros(height, width, CV_32F);
    Mat df_dt = Mat::zeros(height, width, CV_32F);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float curr_dx = curr_frame.at<float>(y, x + 1) - curr_frame.at<float>(y, x - 1);
            float next_dx = next_frame.at<float>(y, x + 1) - next_frame.at<float>(y, x - 1);
            float curr_dy = curr_frame.at<float>(y + 1, x) - curr_frame.at<float>(y - 1, x);
            float next_dy = next_frame.at<float>(y + 1, x) - next_frame.at<float>(y - 1, x);

            df_dx.at<float>(y, x) = 0.25f * (curr_dx + next_dx);
            df_dy.at<float>(y, x) = 0.25f * (curr_dy + next_dy);
            df_dt.at<float>(y, x) = next_frame.at<float>(y, x) - curr_frame.at<float>(y, x);
        }
    }

    // for all pixels
    int half_window = WINDOW_SIZE / 2;
    for (int y = half_window + 1; y < height - half_window - 1; y += STEP_SIZE) {
        for (int x = half_window + 1; x < width - half_window - 1; x += STEP_SIZE) {
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
            float min_eigenvalue = eigenvalues.at<float>(1);
            float max_eigenvalue = eigenvalues.at<float>(0);
            if (min_eigenvalue < MIN_EIGENVALUE ||
                max_eigenvalue <= 0.0f ||
                min_eigenvalue / max_eigenvalue < MIN_EIGEN_RATIO) {
                continue;
            }

            solve(matrix_A, matrix_B, motion_vectors, DECOMP_SVD);

            float u = motion_vectors.at<float>(0);
            float v = motion_vectors.at<float>(1);
            float motion_squared = u * u + v * v;
            if (motion_squared < MIN_MOTION * MIN_MOTION ||
                motion_squared > MAX_MOTION * MAX_MOTION) {
                continue;
            }

            Point start(x, y);
            Point end(cvRound(x + DISPLAY_SCALE * u), cvRound(y + DISPLAY_SCALE * v));
            arrowedLine(curr_frame_color, start, end, Scalar(0, 0, 255), 1);
        }
    }

    // make window for display output
    namedWindow("image", WINDOW_AUTOSIZE);

    imshow("image", curr_frame_color);

    waitKey(0);
}
