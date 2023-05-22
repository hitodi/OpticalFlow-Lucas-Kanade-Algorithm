// test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>

#define WINDOW_SIZE 5

using namespace cv;

int main()
{
    // read img
    Mat curr_frame = imread("data/img/scene00140.png", IMREAD_GRAYSCALE);
    Mat curr_frame_color = imread("data/img/scene00140.png", IMREAD_COLOR);
    Mat next_frame = imread("data/img/scene00141.png", IMREAD_GRAYSCALE);

    // get img size
    int width = curr_frame.cols;
    int height = curr_frame.rows;

    // calc diff between curr_frame, next_frame
    Mat df_dy = Mat::zeros(height, width, CV_32F);
    Mat df_dx = Mat::zeros(height, width, CV_32F);
    Mat df_dt = Mat::zeros(height, width, CV_32F);

    for (int y = 0; y < height - 1; y++) {
        for (int x = 0; x < width - 1; x++) {
            df_dy.at<float>(y, x) = (float) curr_frame.at<uchar>(y + 1, x) - curr_frame.at<uchar>(y, x);
            df_dx.at<float>(y, x) = (float) curr_frame.at<uchar>(y, x + 1) - curr_frame.at<uchar>(y, x);
            df_dt.at<float>(y, x) = (float) next_frame.at<uchar>(y, x) - curr_frame.at<uchar>(y, x);
        }
    }

    // for all pixels
    for (int y = WINDOW_SIZE / 2; y < width - 3; y += WINDOW_SIZE) {
        for (int x = WINDOW_SIZE / 2; x < height - 3; x += WINDOW_SIZE) {
            int index = 0;
            Mat matrix_A = Mat::zeros(WINDOW_SIZE * WINDOW_SIZE, 2, df_dt.type());
            Mat matrix_B = Mat::zeros(WINDOW_SIZE * WINDOW_SIZE, 1, df_dt.type());
            Mat motion_vectors = Mat::zeros(2, 1, df_dt.type());
            // adapt lucas-kanade algorithm
            for (int i = y - WINDOW_SIZE / 2; i < y + WINDOW_SIZE / 2; i++) {
                for (int j = x - WINDOW_SIZE / 2; j < x + WINDOW_SIZE / 2; j++) {

                    matrix_A.at<float>(index, 0) = df_dy.at<float>(j, i);
                    matrix_A.at<float>(index, 1) = df_dx.at<float>(j, i);
                    matrix_B.at<float>(index, 0) = df_dt.at<float>(j, i);
                    index += 1;
                }
            }

            Mat matrix_A_Transform = matrix_A.t();

            motion_vectors = (matrix_A_Transform * matrix_A).inv() * matrix_A_Transform * matrix_B;

            arrowedLine(curr_frame_color, Point(y, x), Point(cvRound(y + motion_vectors.at<float>(0)), cvRound(x + motion_vectors.at<float>(1))), Scalar(0, 0, 255), 1);
        }
    }

    // make window for display output
    namedWindow("display", WINDOW_AUTOSIZE);

    imshow("image", curr_frame_color);

    waitKey(0);
}

