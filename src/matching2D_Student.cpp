#include <numeric>

#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0) {
        int normType = descriptorType.compare("DES_BINARY") == 0
                           ? cv::NORM_HAMMING
                           : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    } else if (matcherType.compare("MAT_FLANN") == 0) {
        if (descSource.type() != CV_32F) {
            descSource.convertTo(descSource, CV_32F);
        }
        if (descRef.type() != CV_32F) {
            descRef.convertTo(descRef, CV_32F);
        }
        matcher =
            cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0) {  // nearest neighbor (best match)
        matcher->match(
            descSource, descRef,
            matches);  // Finds the best match for each descriptor in desc1
    } else if (selectorType.compare("SEL_KNN") ==
               0) {  // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    double t;
    if (descriptorType.compare("BRISK") == 0) {
        int threshold = 30;  // FAST/AGAST detection threshold score.
        int octaves = 3;     // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f;  // apply this scale to the pattern used for
                                    // sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else if (descriptorType.compare("BRIEF") == 0) {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else if (descriptorType.compare("ORB") == 0) {
        cv::Ptr<cv::ORB> extractor = cv::ORB::create();
        extractor->setMaxFeatures(30);
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else if (descriptorType.compare("FREAK") == 0) {
        extractor = cv::xfeatures2d::FREAK::create();
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else if (descriptorType.compare("AKAZE") == 0) {
        extractor = cv::AKAZE::create();
        t = (double)cv::getTickCount();
        extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else if (descriptorType.compare("SIFT") == 0) {
        extractor = cv::xfeatures2d::SIFT::create();
        t = (double)cv::getTickCount();
        extractor->compute(img, keypoints, descriptors);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    } else {
        std::cerr << "Wrong descriptor type: " << descriptorType << std::endl;
        std::cerr << "=> Valid descriptor types: BRISK, BRIEF, ORB, FREAK, "
                     "AKAZE, SIFT"
                  << std::endl;
        exit(1);
    }

    return t;

    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
    //      << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                             bool bVis) {
    // compute detector parameters based on image size
    int blockSize = 4;  //  size of an average block for computing a derivative
                        //  covariation matrix over each pixel neighborhood
    double maxOverlap =
        0.0;  // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners =
        img.rows * img.cols / max(1.0, minDistance);  // max. num. of keypoints

    double qualityLevel = 0.01;  // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
        cv::imwrite("../detection_results/SHITOMASI.jpg", visImage);
    }

    return t;
}

// https://docs.opencv.org/4.5.2/d4/d7d/tutorial_harris_detector.html
double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                          bool bVis) {
    double time;
    int blockSize = 6;
    int apertureSize = 3;
    int minResponse = 50;
    double k = 0.04;
    double maxOverlap = 0.0;
    int response = 0;
    cv::KeyPoint newKeyPoint;
    bool bOverlap;

    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);

    double t = (double)cv::getTickCount();
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            response = (int)dst_norm.at<float>(j, i);

            if (response > minResponse) {
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap) {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response) {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }
                if (!bOverlap) {
                    keypoints.push_back(newKeyPoint);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
        cv::imwrite("../detection_results/HARRIS.jpg", visImage);
    }

    return t;
}

double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                          std::string detectorType, bool bVis) {
    cv::Ptr<cv::Feature2D> detector;
    if (detectorType.compare("FAST") == 0) {
        int threshold = 30;
        int bNMS = true;
        cv::FastFeatureDetector::DetectorType type =
            cv::FastFeatureDetector::TYPE_9_16;
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
    } else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create(30, 3, 1.0);
    } else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
    } else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    } else if (detectorType.compare("SIFT") == 0) {
        detector = cv::xfeatures2d::SIFT::create();
    }

    // Apply corner detection
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Detector Results - " + detectorType;
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
        cv::imwrite(std::string("../detection_results/") + detectorType + ".jpg", visImage);
    }

}