#include <iostream>
#include <ctime>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

//number of buffer frames
const int numOfBufferFrames = 200;

//maximum number of detected objects per frame
const int maxNumOfDetectedObjectsPerFrame = 30;

//maximum number of tracked objects per frame
const int maxNumOfTrackedObjectsPerFrame = maxNumOfDetectedObjectsPerFrame * 2;

//number of trail fields
const int numOfTrailFields = 6;

//weight of background
const int weightOfBackground = 100;

//threshold for difference between 2 frames
const int thresholdForDifference = 70;

//minimum object area
const int minObjestArea = 800;

//maximum moving distance of an object between 2 continuous frames
const int maxMovingDistance = 35;

//maximum degree of prediction
const int maxDegreeOfPrediction = 10;

//maximum changing area of an object between 2 continuous frames
const float maxChangingAreaRate = 1.35f;

//common size of the frames
const cv::Size commonSize = cv::Size(640, 480);

void excute(cv::VideoCapture& video, cv::Mat& background);
void setupTrails(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int KalmanStatus[maxNumOfDetectedObjectsPerFrame]);
cv::Mat getForegroundFromDifference(cv::Mat difference);
void eliminateExternalRegions(cv::Mat& foreground);
void getObjectInfomationFromForeground(cv::Mat foreground, cv::Mat& foreArea, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundRects);
void eliminateSmallContours(std::vector<std::vector<cv::Point>>& contours);
void correctContours(cv::Mat& foreArea, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i> hierarchy);
void eliminateBorderObjects(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundRects);
void track(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int curF, std::vector<std::vector<cv::Point>> contours, std::vector<cv::Rect> boundRects, cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanStatus[maxNumOfDetectedObjectsPerFrame]);
void updateBackground(cv::Mat frame, cv::Mat foreArea, cv::Mat& background, int realIndexOfFramet);
void findPaths(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int curF, std::vector<std::vector<cv::Point>>& paths, std::vector<std::vector<bool>>& ticks);
void drawBoundingBox(cv::Mat& image, std::vector<cv::Rect> boundRects);
void initKalman(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex, float x, float y);
cv::Point kalmanPredict(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex);
cv::Point kalmanCorrect(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex, float x, float y);
int EuclidDistance(int x1, int y1, int x2, int y2);

int main(int argc, char** argv) {
    cv::VideoCapture video(argv[1]);

    if (!video.isOpened()) {
        std::cout << "Video not found." << std::endl;
        return 0;
    }

    //take first frame as original background
    cv::Mat background;
    video >> background;
    cv::resize(background, background, commonSize, 0, 0, cv::INTER_NEAREST);

    excute(video, background);
    cv::waitKey(0);

    return 0;
}

void excute(cv::VideoCapture& video, cv::Mat& background) {
    cv::Mat frame, difference, foreground, foreArea;
    int realIndexOfFrame, curF;
    int numOfDetectedObjects[numOfBufferFrames];
    int numOfPredictedObjects[numOfBufferFrames];
    int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields];
    cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame];
    int KalmanStatus[maxNumOfDetectedObjectsPerFrame];

    realIndexOfFrame = 0;
    setupTrails(numOfDetectedObjects, numOfPredictedObjects, trails, KalmanStatus);
    double t1 = clock();
    for (;;) {
        cv::imshow("background", background);
        curF = realIndexOfFrame % numOfBufferFrames;

        video >> frame;

        if (frame.empty()) {
            break;
            std::cout << "End of video." << std::endl;
        }

        cv::resize(frame, frame, commonSize, 0, 0, cv::INTER_NEAREST);

        difference = cv::abs(frame - background);
        foreground = getForegroundFromDifference(difference);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Rect> boundRects;
        getObjectInfomationFromForeground(foreground, foreArea, contours, boundRects);

        track(numOfDetectedObjects, numOfPredictedObjects, trails, curF, contours, boundRects, KF, KalmanStatus);

        updateBackground(frame, foreArea, background, realIndexOfFrame);

        drawBoundingBox(frame, boundRects);
        std::vector<std::vector<cv::Point>> paths;
        std::vector<std::vector<bool>> ticks;
        findPaths(numOfDetectedObjects, numOfPredictedObjects, trails, curF, paths, ticks);

        for (unsigned int i = 0; i < paths.size(); i++) {
            if (paths[i].size() > 1) {
                for (unsigned int j = 0; j < paths[i].size() - 1; j++) {
                    if (ticks[i][j]) {
                        cv::line(frame, paths[i][j], paths[i][j + 1], cv::Scalar(0, 255 - j * 2, 0), 2);
                    } else {
                        cv::line(frame, paths[i][j], paths[i][j + 1], cv::Scalar(0, 0, 255 - j * 2), 2);
                    }
                }
            }

            if (paths[i].size() > maxDegreeOfPrediction) {
                int length = 0;

                for (int j = 0; j < maxDegreeOfPrediction; j++) {
                    if (paths[i][j].y < 240) {
                        length += round(EuclidDistance(paths[i][j].x, paths[i][j].y, paths[i][j + 1].x, paths[i][j + 1].y) * (2.7 - paths[i][j].y * 2.7 / frame.rows));
                    } else {
                        length += EuclidDistance(paths[i][j].x, paths[i][j].y, paths[i][j + 1].x, paths[i][j + 1].y);
                    }
                }
                length = round(2 * length / (float)maxDegreeOfPrediction);

                if (ticks[i][0]) {
                    std::string str = std::to_string(length);
                    str.append("km/h");
                    cv::Point textPosition = cv::Point(trails[curF][i][1] - 40, trails[curF][i][2] - 5);
                    cv::putText(frame, str, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, 8);
                }
            }

            if (ticks[i][0] && paths[i].size() > maxDegreeOfPrediction) {
                std::string str = std::to_string(trails[curF][i][5]);
                cv::Point textPosition = cv::Point(trails[curF][i][1] - 40, trails[curF][i][2] + 25);
                cv::putText(frame, str, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2, 8);
            }
        }
        std::string str = "frame:";
        str.append(std::to_string(realIndexOfFrame));
        cv::Point textPosition = cv::Point(5, 25);
        cv::putText(frame, str, textPosition, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 125, 255), 2, 8);

        if (realIndexOfFrame > weightOfBackground)
            cv::imshow("frame", frame);
        if (realIndexOfFrame < weightOfBackground)
            cv::waitKey(1);
        else
            cv::waitKey(1);
        realIndexOfFrame++;
    }
    double t2 = clock();
    std::cout << t2 - t1;
}

void setupTrails(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int KalmanStatus[maxNumOfDetectedObjectsPerFrame]) {
    for (int i = 0; i < numOfBufferFrames; i++) {
        numOfDetectedObjects[i] = 0;
        numOfPredictedObjects[i] = 0;

        for (int j = 0; j < maxNumOfTrackedObjectsPerFrame; j++) {
            trails[i][j][0] = 0;
            trails[i][j][1] = 0;
            trails[i][j][2] = 0;
            trails[i][j][3] = -1;
            trails[i][j][4] = -1;
            trails[i][j][5] = -1;
        }
    }

    for (int i = 0; i < maxNumOfDetectedObjectsPerFrame; i++) {
        KalmanStatus[i] = -1;
    }
}

cv::Mat getForegroundFromDifference(cv::Mat difference) {
    cv::Mat foreground;
    cv::threshold(difference, foreground, thresholdForDifference, 255, CV_THRESH_BINARY);
    cv::cvtColor(foreground, foreground, CV_BGR2GRAY);
    cv::threshold(foreground, foreground, 5, 255, CV_THRESH_BINARY);

    //eliminate external regions (depend the video)
    eliminateExternalRegions(foreground);

    dilate(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    //erode(foreground, foreground, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    cv::imshow("foreground", foreground);
    return foreground;
}

void eliminateExternalRegions(cv::Mat& foreground) {
    for (int i = 0; i < commonSize.height; i++) {
        for (int j = 0; j < 190 - i / 3; j++) {
            foreground.at<uchar>(i, j) = 0;
        }
    }

    for (int i = 0; i < commonSize.height; i++) {
        for (int j = 450 + i / 3; j < commonSize.width; j++) {
            foreground.at<uchar>(i, j) = 0;
        }
    }
}

void getObjectInfomationFromForeground(cv::Mat foreground, cv::Mat& foreArea, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundRects) {
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(foreground, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    eliminateSmallContours(contours);

    correctContours(foreArea, contours, hierarchy);

    for (unsigned int i = 0; i < contours.size(); i++) {
        boundRects.push_back(boundingRect(contours[i]));
    }

    eliminateBorderObjects(contours, boundRects);
}

void eliminateSmallContours(std::vector<std::vector<cv::Point>>& contours) {
    int i = contours.size() - 1;
    while (i >= 0) {
        if (cv::contourArea(contours[i]) < minObjestArea) {
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
        }
        i--;
    }
}

void correctContours(cv::Mat& foreArea, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i> hierarchy) {
    for (unsigned int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 3) {
            cv::convexHull(cv::Mat(contours[i]).clone(), contours[i]);
        }
    }

    foreArea = cv::Mat::zeros(commonSize, CV_8UC3);

    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(foreArea, contours, i, cv::Scalar(255, 255, 255), CV_FILLED);
    }
    cv::Mat contoursImg;
    cv::cvtColor(foreArea, contoursImg, CV_BGR2GRAY);

    cv::findContours(contoursImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
}

void eliminateBorderObjects(std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Rect>& boundRects) {
    for (unsigned int i = 0; i < boundRects.size(); i++) {
        if (boundRects[i].tl().x < 5) {
            boundRects.erase(boundRects.begin() + i, boundRects.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundRects.size(); i++) {
        if (boundRects[i].tl().y < 5) {
            boundRects.erase(boundRects.begin() + i, boundRects.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundRects.size(); i++) {
        if (boundRects[i].br().y > commonSize.height - 30) {
            boundRects.erase(boundRects.begin() + i, boundRects.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }

    for (unsigned int i = 0; i < boundRects.size(); i++) {
        if (boundRects[i].br().x > commonSize.width - 30) {
            boundRects.erase(boundRects.begin() + i, boundRects.begin() + i + 1);
            contours.erase(contours.begin() + i, contours.begin() + i + 1);
            i = MAX(0, i - 1);
        }
    }
}

void track(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int curF, std::vector<std::vector<cv::Point>> contours, std::vector<cv::Rect> boundRects, cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanStatus[maxNumOfDetectedObjectsPerFrame]) {
    int preF = (curF + numOfBufferFrames - 1) % numOfBufferFrames;
    int nexF = (curF + 1) % numOfBufferFrames;

    numOfDetectedObjects[curF] = contours.size();
    numOfPredictedObjects[nexF] = 0;
    int nexJ;
    for (int curI = 0; curI < numOfDetectedObjects[curF]; curI++) {
        nexJ = maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[nexF];

        ///Step 1: get object information
        trails[curF][curI][0] = round(cv::contourArea(contours[curI]));
        int mx = (boundRects[curI].x + boundRects[curI].br().x) / 2;
        int my = (boundRects[curI].y + boundRects[curI].br().y) / 2;

        int minMovingDistance = maxMovingDistance + 1;
        int preI = -1;
        for (int pI = 0; pI < numOfDetectedObjects[preF]; pI++) {
            if (trails[preF][pI][4] != -2) {
                int x2 = trails[preF][pI][1];
                int y2 = trails[preF][pI][2];
                int Ed = EuclidDistance(mx, my, x2, y2);

                if (Ed < minMovingDistance) {
                    minMovingDistance = Ed;
                    preI = pI;
                }
            }
        }

        //assume that the nearest object is preI
        float areaDiff = trails[curF][curI][0] / (float)trails[preF][preI][0];
        if (minMovingDistance < maxMovingDistance && areaDiff < maxChangingAreaRate && areaDiff > 1 / maxChangingAreaRate) {
            //we have an object which moves normally
            //trails[preF][preI] ==> trails[curF][curI]
            int KalmanIndex = trails[preF][preI][5];
            cv::Point s = kalmanCorrect(KF, KalmanIndex, (float)mx, (float)my);
            cv::Point p = kalmanPredict(KF, KalmanIndex);
            trails[preF][preI][4] = -2;
            trails[curF][curI][1] = s.x;
            trails[curF][curI][2] = s.y;
            trails[curF][curI][3] = preI;
            trails[curF][curI][4] = 0;
            trails[curF][curI][5] = KalmanIndex;
            //now we predict next frame

            trails[nexF][nexJ][0] = trails[curF][curI][0];
            trails[nexF][nexJ][1] = p.x;
            trails[nexF][nexJ][2] = p.y;
            trails[nexF][nexJ][3] = curI;
            trails[nexF][nexJ][4] = 1;
            trails[nexF][nexJ][5] = KalmanIndex;
        } else {
            //we haven't an object which moves normally
            minMovingDistance = maxMovingDistance + 1;
            int preJ = -1;

            for (int pJ = maxNumOfDetectedObjectsPerFrame; pJ < maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[preF]; pJ++) {
                if (trails[preF][pJ][4] != -2) {
                    int x2 = trails[preF][pJ][1];
                    int y2 = trails[preF][pJ][2];
                    int Ed = EuclidDistance(mx, my, x2, y2);

                    if (Ed < minMovingDistance) {
                        minMovingDistance = Ed;
                        preJ = pJ;
                    }
                }
            }

            //assume that the nearest object is preJ
            areaDiff = trails[curF][curI][0] / (float)trails[preF][preJ][0];
            float curMaxRate = maxChangingAreaRate + 0.02f * trails[preF][preJ][4];
            if (minMovingDistance < maxMovingDistance && areaDiff < curMaxRate && areaDiff > 1 / curMaxRate) {
                //we have an object which is predicted correctly. Previously, It was lost track and now get track again
                //trails[preF][preJ] ==> trails[curF][curI]
                int KalmanIndex = trails[preF][preJ][5];
                cv::Point s = kalmanCorrect(KF, KalmanIndex, (float)mx, (float)my);
                cv::Point p = kalmanPredict(KF, KalmanIndex);
                trails[preF][preJ][4] = -2;
                trails[curF][curI][1] = s.x;
                trails[curF][curI][2] = s.y;
                trails[curF][curI][3] = preJ;
                trails[curF][curI][4] = 0;
                trails[curF][curI][5] = KalmanIndex;

                //now we predict next frame
                trails[nexF][nexJ][0] = trails[curF][curI][0];
                trails[nexF][nexJ][1] = p.x;
                trails[nexF][nexJ][2] = p.y;
                trails[nexF][nexJ][3] = curI;
                trails[nexF][nexJ][4] = 1;
                trails[nexF][nexJ][5] = KalmanIndex;
            } else {
                //we have a new object
                //nothing ==> trails[curF][curI]
                int KalmanIndex = 0;
                while (KalmanIndex < maxNumOfDetectedObjectsPerFrame && KalmanStatus[KalmanIndex] != -1) {
                    KalmanIndex++;
                }
                KalmanStatus[KalmanIndex] = 1;
                initKalman(KF, KalmanIndex, (float)mx, (float)my);
                cv::Point p = kalmanPredict(KF, KalmanIndex);
                trails[curF][curI][1] = mx;
                trails[curF][curI][2] = my;
                trails[curF][curI][3] = -1;
                trails[curF][curI][4] = 0;
                trails[curF][curI][5] = KalmanIndex;

                //now we predict next frame
                trails[nexF][nexJ][0] = trails[curF][curI][0];
                trails[nexF][nexJ][1] = p.x;
                trails[nexF][nexJ][2] = p.y;
                trails[nexF][nexJ][3] = curI;
                trails[nexF][nexJ][4] = 1;
                trails[nexF][nexJ][5] = KalmanIndex;
            }
        }

        numOfPredictedObjects[nexF]++;
    }

    ///Step 3: In current frame, we have some predicted object (curJ) is the same as detected object or has a too big prediction degree,so we need eliminating them.
    for (int curJ = maxNumOfDetectedObjectsPerFrame; curJ < maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[curF]; curJ++) {
        int KalmanIndex = trails[curF][curJ][5];
        if (trails[curF][curJ][4] > maxDegreeOfPrediction) {
            //This object has predicted too many times but has not taken any result, need eliminating trails[curF][curJ]
            //int KalmanIndex = trails[curF][curJ][5];
            KalmanStatus[KalmanIndex] = -1;
            trails[curF][curJ][0] = 0;
            trails[curF][curJ][1] = 0;
            trails[curF][curJ][2] = 0;
            trails[curF][curJ][3] = -1;
            trails[curF][curJ][4] = -1;
            trails[curF][curJ][5] = -1;
        } else {
            //This object has predicted not too many times, may be we should keep track
            int minMovingDistance = maxMovingDistance + 1;
            int curI = -1;
            for (int cI = 0; cI < numOfDetectedObjects[curF]; cI++) {
                int x1 = trails[curF][curJ][1];
                int y1 = trails[curF][curJ][2];
                int x2 = trails[curF][cI][1];
                int y2 = trails[curF][cI][2];
                int Ed = EuclidDistance(x1, y1, x2, y2);

                if (Ed < minMovingDistance) {
                    minMovingDistance = Ed;
                    curI = cI;
                }
            }

            //assume the the nearest object is curI
            float areaDiff = trails[curF][curJ][0] / (float)trails[curF][curI][0];
            float curMaxRate = maxChangingAreaRate + 0.02f * trails[curF][curJ][4];
            if (minMovingDistance < maxMovingDistance && areaDiff < curMaxRate && areaDiff > 1 / curMaxRate) {
                if (trails[curF][curI][5] == trails[curF][curJ][5]) {
                    //is the same object, need eliminating trails[curF][curJ]
                    trails[curF][curJ][0] = 0;
                    trails[curF][curJ][1] = 0;
                    trails[curF][curJ][2] = 0;
                    trails[curF][curJ][3] = -1;
                    trails[curF][curJ][4] = -1;
                    trails[curF][curJ][5] = -1;
                }
            }
        }
    }

    ///Step 4: predict the next frame from predicted object in current frame

    for (int curJ = maxNumOfDetectedObjectsPerFrame; curJ < maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[curF]; curJ++) {
        if (trails[curF][curJ][4] != -1) {
            //This object still remains after step 3, so it quite different than detected object and not too many prediction times
            //now we predict next frame
            nexJ = maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[nexF];
            int preIJ = trails[curF][curJ][3];
            int KalmanIndex = trails[curF][curJ][5];
            cv::Point p = kalmanPredict(KF, KalmanIndex);
            trails[nexF][nexJ][0] = trails[curF][curJ][0];
            trails[nexF][nexJ][1] = p.x;
            trails[nexF][nexJ][2] = p.y;
            trails[nexF][nexJ][3] = curJ;
            trails[nexF][nexJ][4] = trails[curF][curJ][4] + 1;
            //std::cout << "trails[curF][curJ][4]: " << trails[curF][curJ][4] << "  trails[curF][curJ][5]: " << trails[curF][curJ][5] << std::endl;
            trails[nexF][nexJ][5] = KalmanIndex;
            numOfPredictedObjects[nexF]++;
        }
    }
}

void updateBackground(cv::Mat frame, cv::Mat foreArea, cv::Mat& background, int realIndexOfFrame) {
    cv::Mat combinedArea, backArea, joinedArea;
    combinedArea = (frame + background * 4) / 5;
    cv::bitwise_not(foreArea, backArea);

    cv::bitwise_and(foreArea, combinedArea, foreArea);
    cv::bitwise_and(backArea, frame, backArea);
    cv::bitwise_or(foreArea, backArea, joinedArea);
    cv::imshow("joinedArea", joinedArea);

    if (realIndexOfFrame < weightOfBackground) {
        background = (frame + background * realIndexOfFrame) / (realIndexOfFrame + 1);
    } else {
        background = (joinedArea * 2 + background * weightOfBackground) / (weightOfBackground + 2);
    }
}

void findPaths(int numOfDetectedObjects[numOfBufferFrames], int numOfPredictedObjects[numOfBufferFrames], int trails[numOfBufferFrames][maxNumOfTrackedObjectsPerFrame][numOfTrailFields], int curF, std::vector<std::vector<cv::Point>>& paths, std::vector<std::vector<bool>>& ticks) {
    for (int curI = 0; curI < numOfDetectedObjects[curF]; curI++) {
        std::vector<cv::Point> pathI;
        std::vector<bool> tickI;
        int currentFrame = curF;
        int objectIndex = curI;
        cv::Point currentPoint = cv::Point(trails[currentFrame][objectIndex][1], trails[currentFrame][objectIndex][2]);
        pathI.push_back(currentPoint);
        tickI.push_back(true);
        cv::Point prePoint;
        int preFrame;

        for (;;) {
            preFrame = (currentFrame + numOfBufferFrames - 1) % numOfBufferFrames;
            int preObjectIndex = trails[currentFrame][objectIndex][3];
            if (preObjectIndex != -1) {
                prePoint = cv::Point(trails[preFrame][preObjectIndex][1], trails[preFrame][preObjectIndex][2]);
                if (MAX(preObjectIndex, objectIndex) < maxNumOfDetectedObjectsPerFrame) {
                    tickI.push_back(true);
                } else {
                    tickI.push_back(false);
                }
                objectIndex = preObjectIndex;
                currentFrame = preFrame;
                currentPoint = prePoint;
                pathI.push_back(currentPoint);
            } else {
                break;
            }
        }

        paths.push_back(pathI);
        ticks.push_back(tickI);
    }

    for (int curJ = maxNumOfDetectedObjectsPerFrame; curJ < maxNumOfDetectedObjectsPerFrame + numOfPredictedObjects[curF]; curJ++) {
        if (trails[curF][curJ][4] != -1) {
            std::vector<cv::Point> pathI;
            std::vector<bool> tickI;
            int currentFrame = curF;
            int objectIndex = curJ;
            cv::Point currentPoint = cv::Point(trails[currentFrame][objectIndex][1], trails[currentFrame][objectIndex][2]);
            pathI.push_back(currentPoint);
            tickI.push_back(false);
            cv::Point prePoint;
            int preFrame;

            for (;;) {
                preFrame = (currentFrame + numOfBufferFrames - 1) % numOfBufferFrames;
                int preObjectIndex = trails[currentFrame][objectIndex][3];
                if (preObjectIndex != -1) {
                    prePoint = cv::Point(trails[preFrame][preObjectIndex][1], trails[preFrame][preObjectIndex][2]);
                    if (MAX(preObjectIndex, objectIndex) < maxNumOfDetectedObjectsPerFrame) {
                        tickI.push_back(true);
                    } else {
                        tickI.push_back(false);
                    }
                    objectIndex = preObjectIndex;
                    currentFrame = preFrame;
                    currentPoint = prePoint;
                    pathI.push_back(currentPoint);

                } else {
                    break;
                }
            }

            paths.push_back(pathI);
            ticks.push_back(tickI);
        }
    }
}

void drawBoundingBox(cv::Mat& image, std::vector<cv::Rect> boundRects) {
    for (unsigned int i = 0; i < boundRects.size(); i++) {
        cv::rectangle(image, boundRects[i].tl(), boundRects[i].br(), cv::Scalar(255, 0, 0), 2, 8, 0);
    }
}

void initKalman(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex, float x, float y) {
    KF[KalmanIndex].init(4, 2, 0);
    cv::Mat_<float> measurement(2, 1);
    measurement = cv::Mat_<float>::zeros(2, 1);
    measurement.at<float>(0, 0) = x;
    measurement.at<float>(0, 0) = y;

    KF[KalmanIndex].statePre.setTo(0);
    KF[KalmanIndex].statePre.at<float>(0, 0) = x;
    KF[KalmanIndex].statePre.at<float>(1, 0) = y;

    KF[KalmanIndex].statePost.setTo(0);
    KF[KalmanIndex].statePost.at<float>(0, 0) = x;
    KF[KalmanIndex].statePost.at<float>(1, 0) = y;

    KF[KalmanIndex].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 0.8, 0, 0, 1, 0, 0.8, 0, 0, 1, 0, 0, 0, 0, 1);
    cv::setIdentity(KF[KalmanIndex].measurementMatrix);
    cv::setIdentity(KF[KalmanIndex].processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(KF[KalmanIndex].measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(KF[KalmanIndex].errorCovPost, cv::Scalar::all(.1));
}

cv::Point kalmanPredict(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex) {
    cv::Mat prediction = KF[KalmanIndex].predict();
    cv::Point predictPt(round(prediction.at<float>(0)), round(prediction.at<float>(1)));
    return predictPt;
}

cv::Point kalmanCorrect(cv::KalmanFilter KF[maxNumOfDetectedObjectsPerFrame], int KalmanIndex, float x, float y) {
    cv::Mat_<float> measurement(2, 1);
    measurement(0) = x;
    measurement(1) = y;
    cv::Mat estimated = KF[KalmanIndex].correct(measurement);
    cv::Point statePt(round(estimated.at<float>(0)), round(estimated.at<float>(1)));
    return statePt;
}

int EuclidDistance(int x1, int y1, int x2, int y2) {
    int dx = x2 - x1;
    int dy = y2 - y1;
    return round(sqrt((float)(dx * dx + dy * dy)));
}
