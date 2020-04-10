//
//  util.h
//  Eerie
//
//  Created by Evgeny on 9/17/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#ifndef __Eerie__io__
#define __Eerie__io__

#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/xfeatures2d.hpp>
using namespace cv::xfeatures2d;
namespace cv {
namespace evg {

// TODO: for read - change "cv::Mat& descr" to "cv::Mat descr = cv::noArray()"

bool readVLFeatFormat   (const std::string& filepath,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors,
                         bool binFormat = false);

bool writeVLFeatFormat  (const std::string& filepath,
                         const std::vector<cv::KeyPoint>& keypoints,
                         const cv::Mat& descriptors,
                         bool binFormat = false);


bool readUbcFormat      (const std::string& filepath,
                         std::vector<cv::KeyPoint>& keypoints,
                         cv::Mat& descriptors,
                         bool binFormat = false);

bool writeUbcFormat     (const std::string& filepath,
                         const std::vector<cv::KeyPoint>& keypoints,
                         const cv::Mat& descriptors,
                         bool binFormat = false);
    
bool writeVsfmMatches   (const std::string& filepath,
                         const std::string& imName1, const std::string& imName2,
                         const std::vector<cv::DMatch>& matches);

bool readVsfmMatches    (const std::string& filepath,
                         std::string& imName1, std::string& imName2,
                         std::vector<cv::DMatch>& matches);
    
bool writeSimpleMatches (const std::string& filepath,
                         const std::string& imName1, const std::string& imName2,
                         const std::vector<cv::KeyPoint>& keypoints1,
                         const std::vector<cv::KeyPoint>& keypoints2,
                         const std::vector<cv::DMatch>& matches);
    
bool readSimpleMatches  (const std::string& filepath,
                         std::string& imName1, std::string& imName2,
                         std::vector<cv::KeyPoint>& keypoints1,
                         std::vector<cv::KeyPoint>& keypoints2,
                         std::vector<cv::DMatch>& matches);
    


} // namespace evg
} // namespace cv

#endif /* defined(__Eerie__io__) */
