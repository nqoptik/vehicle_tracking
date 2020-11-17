#include <ctime>
#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// Number of buffer frames
const int number_of_buffer_frames = 200;

// Maximum number of detected objects per frame
const int maximum_number_of_detected_objects_per_frame = 30;

// Maximum number of tracked objects per frame
const int maximum_number_of_tracked_objects_per_frame = maximum_number_of_detected_objects_per_frame * 2;

// Number of trail fields
const int number_of_trail_fields = 6;

// Weight of background
const int weight_of_background = 100;

// Threshold for difference between 2 frames
const int difference_threshold = 70;

// Minimum object area
const int minimum_object_area = 800;

// Maximum moving distance of an object between 2 continuous frames
const int maximum_moving_distance = 35;

// Maximum degree of prediction
const int maximum_degree_of_prediction = 10;

// Maximum changing area of an object between 2 continuous frames
const float maximum_changing_area_rate = 1.35f;

// The resolution of the frame
const cv::Size frame_resolution = cv::Size(640, 480);

void excute(cv::VideoCapture& video, cv::Mat& background);
void setup_trails(int number_of_detected_objects[number_of_buffer_frames],
                  int number_of_predicted_objects[number_of_buffer_frames],
                  int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
                  int kalman_status[maximum_number_of_detected_objects_per_frame]);
cv::Mat get_foreground_from_difference(cv::Mat difference);
void eliminate_external_regions(cv::Mat& foreground);
void get_object_infomation_from_foreground(cv::Mat foreground,
                                           cv::Mat& foreground_area,
                                           std::vector<std::vector<cv::Point>>& contours,
                                           std::vector<cv::Rect>& boundary_rectangles);
void eliminate_small_contours(std::vector<std::vector<cv::Point>>& contours);
void correct_contours(cv::Mat& foreground_area, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i> hierarchy);
void eliminate_border_objects(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundary_rectangles);
void track(int number_of_detected_objects[number_of_buffer_frames],
           int number_of_predicted_objects[number_of_buffer_frames],
           int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
           int current_frame_index,
           std::vector<std::vector<cv::Point>> contours,
           std::vector<cv::Rect> boundary_rectangles,
           cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame],
           int kalman_status[maximum_number_of_detected_objects_per_frame]);
void update_background(cv::Mat frame, cv::Mat foreground_area, cv::Mat& background, int real_frame_index);
void find_paths(int number_of_detected_objects[number_of_buffer_frames],
                int number_of_predicted_objects[number_of_buffer_frames],
                int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
                int current_frame_index,
                std::vector<std::vector<cv::Point>>& paths,
                std::vector<std::vector<bool>>& ticks);
void draw_bounding_box(cv::Mat& image, std::vector<cv::Rect> boundary_rectangles);
void initialise_kalman_filter(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index, float x, float y);
cv::Point kalman_predict(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index);
cv::Point kalman_correct(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index, float x, float y);
int get_euclid_distance(int x1, int y1, int x2, int y2);

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("To run vehicle_tracking, type ./vehicle_tracking <video_file>\n");
        return 1;
    }
    cv::VideoCapture video(argv[1]);

    if (!video.isOpened())
    {
        std::cout << "Video not found." << std::endl;
        return 0;
    }

    // Take the first frame as original background
    cv::Mat background;
    video >> background;
    cv::resize(background, background, frame_resolution, 0, 0, cv::INTER_NEAREST);

    excute(video, background);
    cv::waitKey(0);

    return 0;
}

void excute(cv::VideoCapture& video, cv::Mat& background)
{
    cv::Mat frame, difference, foreground, foreground_area;
    int real_frame_index, current_frame_index;
    int number_of_detected_objects[number_of_buffer_frames];
    int number_of_predicted_objects[number_of_buffer_frames];
    int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields];
    cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame];
    int kalman_status[maximum_number_of_detected_objects_per_frame];

    real_frame_index = 0;
    setup_trails(number_of_detected_objects, number_of_predicted_objects, trails, kalman_status);
    double t1 = clock();
    while (true)
    {
        cv::imshow("background", background);
        current_frame_index = real_frame_index % number_of_buffer_frames;

        video >> frame;

        if (frame.empty())
        {
            break;
            std::cout << "End of video." << std::endl;
        }

        cv::resize(frame, frame, frame_resolution, 0, 0, cv::INTER_NEAREST);

        difference = cv::abs(frame - background);
        foreground = get_foreground_from_difference(difference);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Rect> boundary_rectangles;
        get_object_infomation_from_foreground(foreground, foreground_area, contours, boundary_rectangles);

        track(number_of_detected_objects, number_of_predicted_objects, trails, current_frame_index, contours, boundary_rectangles, kalman_filter, kalman_status);

        update_background(frame, foreground_area, background, real_frame_index);

        draw_bounding_box(frame, boundary_rectangles);
        std::vector<std::vector<cv::Point>> paths;
        std::vector<std::vector<bool>> ticks;
        find_paths(number_of_detected_objects, number_of_predicted_objects, trails, current_frame_index, paths, ticks);

        for (unsigned int i = 0; i < paths.size(); ++i)
        {
            if (paths[i].size() > 1)
            {
                for (unsigned int j = 0; j < paths[i].size() - 1; ++j)
                {
                    if (ticks[i][j])
                    {
                        cv::line(frame, paths[i][j], paths[i][j + 1], cv::Scalar(0, 255 - j * 2, 0), 2);
                    }
                    else
                    {
                        cv::line(frame, paths[i][j], paths[i][j + 1], cv::Scalar(0, 0, 255 - j * 2), 2);
                    }
                }
            }

            if (paths[i].size() > maximum_degree_of_prediction)
            {
                int length = 0;

                for (int j = 0; j < maximum_degree_of_prediction; ++j)
                {
                    if (paths[i][j].y < 240)
                    {
                        length += round(get_euclid_distance(paths[i][j].x, paths[i][j].y, paths[i][j + 1].x, paths[i][j + 1].y) * (2.7 - paths[i][j].y * 2.7 / frame.rows));
                    }
                    else
                    {
                        length += get_euclid_distance(paths[i][j].x, paths[i][j].y, paths[i][j + 1].x, paths[i][j + 1].y);
                    }
                }
                length = round(2 * length / (float)maximum_degree_of_prediction);

                if (ticks[i][0])
                {
                    std::string str = std::to_string(length);
                    str.append("km/h");
                    cv::Point text_position = cv::Point(trails[current_frame_index][i][1] - 40, trails[current_frame_index][i][2] - 5);
                    cv::putText(frame, str, text_position, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 8);
                }
            }

            if (ticks[i][0] && paths[i].size() > maximum_degree_of_prediction)
            {
                std::string str = std::to_string(trails[current_frame_index][i][5]);
                cv::Point text_position = cv::Point(trails[current_frame_index][i][1] - 40, trails[current_frame_index][i][2] + 25);
                cv::putText(frame, str, text_position, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2, 8);
            }
        }
        std::string str = "frame:";
        str.append(std::to_string(real_frame_index));
        cv::Point text_position = cv::Point(5, 25);
        cv::putText(frame, str, text_position, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 125, 255), 2, 8);

        if (real_frame_index > weight_of_background)
            cv::imshow("frame", frame);
        if (real_frame_index < weight_of_background)
            cv::waitKey(1);
        else
            cv::waitKey(1);
        ++real_frame_index;
    }
    double t2 = clock();
    std::cout << t2 - t1;
}

void setup_trails(int number_of_detected_objects[number_of_buffer_frames],
                  int number_of_predicted_objects[number_of_buffer_frames],
                  int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
                  int kalman_status[maximum_number_of_detected_objects_per_frame])
{
    for (int i = 0; i < number_of_buffer_frames; ++i)
    {
        number_of_detected_objects[i] = 0;
        number_of_predicted_objects[i] = 0;

        for (int j = 0; j < maximum_number_of_tracked_objects_per_frame; ++j)
        {
            trails[i][j][0] = 0;
            trails[i][j][1] = 0;
            trails[i][j][2] = 0;
            trails[i][j][3] = -1;
            trails[i][j][4] = -1;
            trails[i][j][5] = -1;
        }
    }

    for (int i = 0; i < maximum_number_of_detected_objects_per_frame; ++i)
    {
        kalman_status[i] = -1;
    }
}

cv::Mat get_foreground_from_difference(cv::Mat difference)
{
    cv::Mat foreground;
    cv::threshold(difference, foreground, difference_threshold, 255, cv::THRESH_BINARY);
    cv::cvtColor(foreground, foreground, cv::COLOR_BGR2GRAY);
    cv::threshold(foreground, foreground, 5, 255, cv::THRESH_BINARY);

    // Eliminate external regions (depends on the video)
    eliminate_external_regions(foreground);

    dilate(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    //erode(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    cv::imshow("foreground", foreground);
    return foreground;
}

void eliminate_external_regions(cv::Mat& foreground)
{
    for (int i = 0; i < frame_resolution.height; ++i)
    {
        for (int j = 0; j < 190 - i / 3; ++j)
        {
            foreground.at<uchar>(i, j) = 0;
        }
    }

    for (int i = 0; i < frame_resolution.height; ++i)
    {
        for (int j = 450 + i / 3; j < frame_resolution.width; ++j)
        {
            foreground.at<uchar>(i, j) = 0;
        }
    }
}

void get_object_infomation_from_foreground(cv::Mat foreground,
                                           cv::Mat& foreground_area,
                                           std::vector<std::vector<cv::Point>>& contours,
                                           std::vector<cv::Rect>& boundary_rectangles)
{
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(foreground, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    eliminate_small_contours(contours);

    correct_contours(foreground_area, contours, hierarchy);

    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        boundary_rectangles.push_back(cv::boundingRect(contours[i]));
    }

    eliminate_border_objects(contours, boundary_rectangles);
}

void eliminate_small_contours(std::vector<std::vector<cv::Point>>& contours)
{
    int i = contours.size() - 1;
    while (i >= 0)
    {
        if (cv::contourArea(contours[i]) < minimum_object_area)
        {
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
        }
        --i;
    }
}

void correct_contours(cv::Mat& foreground_area, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i> hierarchy)
{
    for (unsigned int i = 0; i < contours.size(); ++i)
    {
        if (contours[i].size() > 3)
        {
            cv::convexHull(cv::Mat(contours[i]).clone(), contours[i]);
        }
    }

    foreground_area = cv::Mat::zeros(frame_resolution, CV_8UC3);

    for (size_t i = 0; i < contours.size(); ++i)
    {
        cv::drawContours(foreground_area, contours, i, cv::Scalar(255, 255, 255), cv::FILLED);
    }
    cv::Mat contours_image;
    cv::cvtColor(foreground_area, contours_image, cv::COLOR_BGR2GRAY);

    cv::findContours(contours_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void eliminate_border_objects(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundary_rectangles)
{
    for (unsigned int i = 0; i < boundary_rectangles.size(); ++i)
    {
        if (boundary_rectangles[i].tl().x < 5)
        {
            boundary_rectangles.erase(boundary_rectangles.begin() + i, boundary_rectangles.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundary_rectangles.size(); ++i)
    {
        if (boundary_rectangles[i].tl().y < 5)
        {
            boundary_rectangles.erase(boundary_rectangles.begin() + i, boundary_rectangles.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundary_rectangles.size(); ++i)
    {
        if (boundary_rectangles[i].br().y > frame_resolution.height - 30)
        {
            boundary_rectangles.erase(boundary_rectangles.begin() + i, boundary_rectangles.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundary_rectangles.size(); ++i)
    {
        if (boundary_rectangles[i].br().x > frame_resolution.width - 30)
        {
            boundary_rectangles.erase(boundary_rectangles.begin() + i, boundary_rectangles.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }
}

void track(int number_of_detected_objects[number_of_buffer_frames],
           int number_of_predicted_objects[number_of_buffer_frames],
           int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
           int current_frame_index,
           std::vector<std::vector<cv::Point>> contours,
           std::vector<cv::Rect> boundary_rectangles,
           cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame],
           int kalman_status[maximum_number_of_detected_objects_per_frame])
{
    int previous_frame_index = (current_frame_index + number_of_buffer_frames - 1) % number_of_buffer_frames;
    int next_frame_index = (current_frame_index + 1) % number_of_buffer_frames;

    number_of_detected_objects[current_frame_index] = contours.size();
    number_of_predicted_objects[next_frame_index] = 0;
    int nexJ;
    for (int curI = 0; curI < number_of_detected_objects[current_frame_index]; ++curI)
    {
        nexJ = maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[next_frame_index];

        // Step 1: Get object information
        trails[current_frame_index][curI][0] = round(cv::contourArea(contours[curI]));
        int mx = (boundary_rectangles[curI].x + boundary_rectangles[curI].br().x) / 2;
        int my = (boundary_rectangles[curI].y + boundary_rectangles[curI].br().y) / 2;

        int minimum_moving_distance = maximum_moving_distance + 1;
        int preI = -1;
        for (int pI = 0; pI < number_of_detected_objects[previous_frame_index]; ++pI)
        {
            if (trails[previous_frame_index][pI][4] != -2)
            {
                int x2 = trails[previous_frame_index][pI][1];
                int y2 = trails[previous_frame_index][pI][2];
                int this_distance = get_euclid_distance(mx, my, x2, y2);

                if (this_distance < minimum_moving_distance)
                {
                    minimum_moving_distance = this_distance;
                    preI = pI;
                }
            }
        }

        // Assume that the nearest object is preI
        float areaDiff = trails[current_frame_index][curI][0] / (float)trails[previous_frame_index][preI][0];
        if (minimum_moving_distance < maximum_moving_distance && areaDiff < maximum_changing_area_rate && areaDiff > 1 / maximum_changing_area_rate)
        {
            // We have an object which moves normally
            // trails[previous_frame_index][preI] ==> trails[current_frame_index][curI]
            int kalman_index = trails[previous_frame_index][preI][5];
            cv::Point s = kalman_correct(kalman_filter, kalman_index, (float)mx, (float)my);
            cv::Point p = kalman_predict(kalman_filter, kalman_index);
            trails[previous_frame_index][preI][4] = -2;
            trails[current_frame_index][curI][1] = s.x;
            trails[current_frame_index][curI][2] = s.y;
            trails[current_frame_index][curI][3] = preI;
            trails[current_frame_index][curI][4] = 0;
            trails[current_frame_index][curI][5] = kalman_index;

            // Predict the next frame
            trails[next_frame_index][nexJ][0] = trails[current_frame_index][curI][0];
            trails[next_frame_index][nexJ][1] = p.x;
            trails[next_frame_index][nexJ][2] = p.y;
            trails[next_frame_index][nexJ][3] = curI;
            trails[next_frame_index][nexJ][4] = 1;
            trails[next_frame_index][nexJ][5] = kalman_index;
        }
        else
        {
            // We do not have any object which moves normally
            minimum_moving_distance = maximum_moving_distance + 1;
            int preJ = -1;

            for (int pJ = maximum_number_of_detected_objects_per_frame; pJ < maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[previous_frame_index]; ++pJ)
            {
                if (trails[previous_frame_index][pJ][4] != -2)
                {
                    int x2 = trails[previous_frame_index][pJ][1];
                    int y2 = trails[previous_frame_index][pJ][2];
                    int this_distance = get_euclid_distance(mx, my, x2, y2);

                    if (this_distance < minimum_moving_distance)
                    {
                        minimum_moving_distance = this_distance;
                        preJ = pJ;
                    }
                }
            }

            // Assume that the nearest object is preJ
            areaDiff = trails[current_frame_index][curI][0] / (float)trails[previous_frame_index][preJ][0];
            float curMaxRate = maximum_changing_area_rate + 0.02f * trails[previous_frame_index][preJ][4];
            if (minimum_moving_distance < maximum_moving_distance && areaDiff < curMaxRate && areaDiff > 1 / curMaxRate)
            {
                // We have an object which is predicted correctly. Previously, It was lost track and now get track again
                // trails[previous_frame_index][preJ] ==> trails[current_frame_index][curI]
                int kalman_index = trails[previous_frame_index][preJ][5];
                cv::Point s = kalman_correct(kalman_filter, kalman_index, (float)mx, (float)my);
                cv::Point p = kalman_predict(kalman_filter, kalman_index);
                trails[previous_frame_index][preJ][4] = -2;
                trails[current_frame_index][curI][1] = s.x;
                trails[current_frame_index][curI][2] = s.y;
                trails[current_frame_index][curI][3] = preJ;
                trails[current_frame_index][curI][4] = 0;
                trails[current_frame_index][curI][5] = kalman_index;

                // Predict the next frame
                trails[next_frame_index][nexJ][0] = trails[current_frame_index][curI][0];
                trails[next_frame_index][nexJ][1] = p.x;
                trails[next_frame_index][nexJ][2] = p.y;
                trails[next_frame_index][nexJ][3] = curI;
                trails[next_frame_index][nexJ][4] = 1;
                trails[next_frame_index][nexJ][5] = kalman_index;
            }
            else
            {
                // We have a new object
                // nothing ==> trails[current_frame_index][curI]
                int kalman_index = 0;
                while (kalman_index < maximum_number_of_detected_objects_per_frame && kalman_status[kalman_index] != -1)
                {
                    ++kalman_index;
                }
                kalman_status[kalman_index] = 1;
                initialise_kalman_filter(kalman_filter, kalman_index, (float)mx, (float)my);
                cv::Point p = kalman_predict(kalman_filter, kalman_index);
                trails[current_frame_index][curI][1] = mx;
                trails[current_frame_index][curI][2] = my;
                trails[current_frame_index][curI][3] = -1;
                trails[current_frame_index][curI][4] = 0;
                trails[current_frame_index][curI][5] = kalman_index;

                // Predict the next frame
                trails[next_frame_index][nexJ][0] = trails[current_frame_index][curI][0];
                trails[next_frame_index][nexJ][1] = p.x;
                trails[next_frame_index][nexJ][2] = p.y;
                trails[next_frame_index][nexJ][3] = curI;
                trails[next_frame_index][nexJ][4] = 1;
                trails[next_frame_index][nexJ][5] = kalman_index;
            }
        }

        ++number_of_predicted_objects[next_frame_index];
    }

    // Step 3: In current frame, we have some predicted object (curJ) is the same as detected object or has a too big prediction degree,so we need eliminating them.
    for (int curJ = maximum_number_of_detected_objects_per_frame; curJ < maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[current_frame_index]; ++curJ)
    {
        int kalman_index = trails[current_frame_index][curJ][5];
        if (trails[current_frame_index][curJ][4] > maximum_degree_of_prediction)
        {
            // This object has predicted too many times but has not taken any result, need eliminating trails[current_frame_index][curJ]
            // int kalman_index = trails[current_frame_index][curJ][5];
            kalman_status[kalman_index] = -1;
            trails[current_frame_index][curJ][0] = 0;
            trails[current_frame_index][curJ][1] = 0;
            trails[current_frame_index][curJ][2] = 0;
            trails[current_frame_index][curJ][3] = -1;
            trails[current_frame_index][curJ][4] = -1;
            trails[current_frame_index][curJ][5] = -1;
        }
        else
        {
            // This object has predicted not too many times, may be we should keep track
            int minimum_moving_distance = maximum_moving_distance + 1;
            int curI = -1;
            for (int cI = 0; cI < number_of_detected_objects[current_frame_index]; ++cI)
            {
                int x1 = trails[current_frame_index][curJ][1];
                int y1 = trails[current_frame_index][curJ][2];
                int x2 = trails[current_frame_index][cI][1];
                int y2 = trails[current_frame_index][cI][2];
                int this_distance = get_euclid_distance(x1, y1, x2, y2);

                if (this_distance < minimum_moving_distance)
                {
                    minimum_moving_distance = this_distance;
                    curI = cI;
                }
            }

            // Assume the the nearest object is curI
            float areaDiff = trails[current_frame_index][curJ][0] / (float)trails[current_frame_index][curI][0];
            float curMaxRate = maximum_changing_area_rate + 0.02f * trails[current_frame_index][curJ][4];
            if (minimum_moving_distance < maximum_moving_distance && areaDiff < curMaxRate && areaDiff > 1 / curMaxRate)
            {
                if (trails[current_frame_index][curI][5] == trails[current_frame_index][curJ][5])
                {
                    // Is the same object, need eliminating trails[current_frame_index][curJ]
                    trails[current_frame_index][curJ][0] = 0;
                    trails[current_frame_index][curJ][1] = 0;
                    trails[current_frame_index][curJ][2] = 0;
                    trails[current_frame_index][curJ][3] = -1;
                    trails[current_frame_index][curJ][4] = -1;
                    trails[current_frame_index][curJ][5] = -1;
                }
            }
        }
    }

    // Step 4: Predict the next frame from predicted object in current frame

    for (int curJ = maximum_number_of_detected_objects_per_frame; curJ < maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[current_frame_index]; ++curJ)
    {
        if (trails[current_frame_index][curJ][4] != -1)
        {
            // This object still remains after step 3, so it quite different than detected object and not too many prediction times
            // Predict the next frame
            nexJ = maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[next_frame_index];
            int kalman_index = trails[current_frame_index][curJ][5];
            cv::Point p = kalman_predict(kalman_filter, kalman_index);
            trails[next_frame_index][nexJ][0] = trails[current_frame_index][curJ][0];
            trails[next_frame_index][nexJ][1] = p.x;
            trails[next_frame_index][nexJ][2] = p.y;
            trails[next_frame_index][nexJ][3] = curJ;
            trails[next_frame_index][nexJ][4] = trails[current_frame_index][curJ][4] + 1;
            //std::cout << "trails[current_frame_index][curJ][4]: " << trails[current_frame_index][curJ][4] << "  trails[current_frame_index][curJ][5]: " << trails[current_frame_index][curJ][5] << std::endl;
            trails[next_frame_index][nexJ][5] = kalman_index;
            ++number_of_predicted_objects[next_frame_index];
        }
    }
}

void update_background(cv::Mat frame, cv::Mat foreground_area, cv::Mat& background, int real_frame_index)
{
    cv::Mat combined_area, background_area, joined_area;
    combined_area = (frame + background * 4) / 5;
    cv::bitwise_not(foreground_area, background_area);

    cv::bitwise_and(foreground_area, combined_area, foreground_area);
    cv::bitwise_and(background_area, frame, background_area);
    cv::bitwise_or(foreground_area, background_area, joined_area);
    cv::imshow("joined_area", joined_area);

    if (real_frame_index < weight_of_background)
    {
        background = (frame + background * real_frame_index) / (real_frame_index + 1);
    }
    else
    {
        background = (joined_area * 2 + background * weight_of_background) / (weight_of_background + 2);
    }
}

void find_paths(int number_of_detected_objects[number_of_buffer_frames],
                int number_of_predicted_objects[number_of_buffer_frames],
                int trails[number_of_buffer_frames][maximum_number_of_tracked_objects_per_frame][number_of_trail_fields],
                int current_frame_index,
                std::vector<std::vector<cv::Point>>& paths,
                std::vector<std::vector<bool>>& ticks)
{
    for (int curI = 0; curI < number_of_detected_objects[current_frame_index]; ++curI)
    {
        std::vector<cv::Point> pathI;
        std::vector<bool> tickI;
        int current_frame = current_frame_index;
        int object_index = curI;
        cv::Point current_point = cv::Point(trails[current_frame][object_index][1], trails[current_frame][object_index][2]);
        pathI.push_back(current_point);
        tickI.push_back(true);
        cv::Point previous_point;
        int previous_frame;

        while (true)
        {
            previous_frame = (current_frame + number_of_buffer_frames - 1) % number_of_buffer_frames;
            int previous_object_index = trails[current_frame][object_index][3];
            if (previous_object_index != -1)
            {
                previous_point = cv::Point(trails[previous_frame][previous_object_index][1], trails[previous_frame][previous_object_index][2]);
                if (MAX(previous_object_index, object_index) < maximum_number_of_detected_objects_per_frame)
                {
                    tickI.push_back(true);
                }
                else
                {
                    tickI.push_back(false);
                }
                object_index = previous_object_index;
                current_frame = previous_frame;
                current_point = previous_point;
                pathI.push_back(current_point);
            }
            else
            {
                break;
            }
        }

        paths.push_back(pathI);
        ticks.push_back(tickI);
    }

    for (int curJ = maximum_number_of_detected_objects_per_frame; curJ < maximum_number_of_detected_objects_per_frame + number_of_predicted_objects[current_frame_index]; ++curJ)
    {
        if (trails[current_frame_index][curJ][4] != -1)
        {
            std::vector<cv::Point> pathI;
            std::vector<bool> tickI;
            int current_frame = current_frame_index;
            int object_index = curJ;
            cv::Point current_point = cv::Point(trails[current_frame][object_index][1], trails[current_frame][object_index][2]);
            pathI.push_back(current_point);
            tickI.push_back(false);
            cv::Point previous_point;
            int previous_frame;

            for (;;)
            {
                previous_frame = (current_frame + number_of_buffer_frames - 1) % number_of_buffer_frames;
                int previous_object_index = trails[current_frame][object_index][3];
                if (previous_object_index != -1)
                {
                    previous_point = cv::Point(trails[previous_frame][previous_object_index][1], trails[previous_frame][previous_object_index][2]);
                    if (MAX(previous_object_index, object_index) < maximum_number_of_detected_objects_per_frame)
                    {
                        tickI.push_back(true);
                    }
                    else
                    {
                        tickI.push_back(false);
                    }
                    object_index = previous_object_index;
                    current_frame = previous_frame;
                    current_point = previous_point;
                    pathI.push_back(current_point);
                }
                else
                {
                    break;
                }
            }

            paths.push_back(pathI);
            ticks.push_back(tickI);
        }
    }
}

void draw_bounding_box(cv::Mat& image, std::vector<cv::Rect> boundary_rectangles)
{
    for (unsigned int i = 0; i < boundary_rectangles.size(); ++i)
    {
        cv::rectangle(image, boundary_rectangles[i].tl(), boundary_rectangles[i].br(), cv::Scalar(255, 0, 0), 2, 8, 0);
    }
}

void initialise_kalman_filter(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index, float x, float y)
{
    kalman_filter[kalman_index].init(4, 2, 0);
    cv::Mat_<float> measurement(2, 1);
    measurement = cv::Mat_<float>::zeros(2, 1);
    measurement.at<float>(0, 0) = x;
    measurement.at<float>(0, 0) = y;

    kalman_filter[kalman_index].statePre.setTo(0);
    kalman_filter[kalman_index].statePre.at<float>(0, 0) = x;
    kalman_filter[kalman_index].statePre.at<float>(1, 0) = y;

    kalman_filter[kalman_index].statePost.setTo(0);
    kalman_filter[kalman_index].statePost.at<float>(0, 0) = x;
    kalman_filter[kalman_index].statePost.at<float>(1, 0) = y;

    kalman_filter[kalman_index].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 0.8, 0, 0, 1, 0, 0.8, 0, 0, 1, 0, 0, 0, 0, 1);
    cv::setIdentity(kalman_filter[kalman_index].measurementMatrix);
    cv::setIdentity(kalman_filter[kalman_index].processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(kalman_filter[kalman_index].measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(kalman_filter[kalman_index].errorCovPost, cv::Scalar::all(.1));
}

cv::Point kalman_predict(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index)
{
    cv::Mat prediction = kalman_filter[kalman_index].predict();
    cv::Point predicted_point(round(prediction.at<float>(0)), round(prediction.at<float>(1)));
    return predicted_point;
}

cv::Point kalman_correct(cv::KalmanFilter kalman_filter[maximum_number_of_detected_objects_per_frame], int kalman_index, float x, float y)
{
    cv::Mat_<float> measurement(2, 1);
    measurement(0) = x;
    measurement(1) = y;
    cv::Mat estimated = kalman_filter[kalman_index].correct(measurement);
    cv::Point state_point(round(estimated.at<float>(0)), round(estimated.at<float>(1)));
    return state_point;
}

int get_euclid_distance(int x1, int y1, int x2, int y2)
{
    int dx = x2 - x1;
    int dy = y2 - y1;
    return round(sqrt((float)(dx * dx + dy * dy)));
}
