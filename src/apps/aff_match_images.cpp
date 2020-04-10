//
//  aff_match_images.cpp
//
//  Created by Evgeny on 7/22/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>
#include <fstream>

#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "aff_features2d.hpp"

#include "mediaIO.h"
#include "featuresIO.h"

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace TCLAP;


static cv::FeatureDetector* newFeatureDetector (const std::string& featureType)
{
    FeatureDetector* detector (NULL);
    // if (featureType == "sift")
    //     detector = new cv::SIFT();
    // else if (featureType == "surf")
    //     detector = new cv::SURF();
    // else if (featureType == "orb")
    //     detector = new ORB();
    // else if (featureType == "brisk")
    //     detector = new BRISK();
    // else assert(0);
    detector = new SIFT();
    return detector;
}


static cv::DescriptorExtractor* newDescriptorExtractor (const string& featureType)
{
    DescriptorExtractor* extractor (NULL);

    // if (featureType == "sift")
    //     extractor = new SIFT();
    // else if (featureType == "surf")
    //     extractor = new SURF();
    // else if (featureType == "orb")
    //     extractor = new ORB();
    // else if (featureType == "brisk")
    //     extractor = new BRISK();
    // else assert(0);
    // return extractor;
    return new SIFT();
}


static cv::DescriptorMatcher* newMatcher (const std::string& featureType, int verbose = 0)
{
    cv::DescriptorMatcher* matcher;
    if      (featureType == "sift" || featureType == "surf")
    {
        if (verbose) cout << "using FlannBasedMatcher on sift or surf" << endl;
        matcher = new FlannBasedMatcher();
    }
    //else if (featureType == "orb" || featureType == "brisk")
    //{
    //    if (verbose) cout << "using BFMatcher(NORM_HAMMING) on orb, or brisk" << endl;
    //    matcher = new BFMatcher (NORM_HAMMING);
    //}
    else if (featureType == "orb" || featureType == "brisk")
    {
        if (verbose) cout << "using FlannBasedMatcher(LshIndexParams) on orb or brisk" << endl;
        matcher = new FlannBasedMatcher(new flann::LshIndexParams(8, 15, 2));
    }
    else
        assert (0);
    return matcher;
}


int main(int argc, const char * argv[])
{
    // parse input
    CmdLine cmd ("match pair from user input and write results");
    
    vector<string> featureTypes;
    featureTypes.push_back("sift");
    featureTypes.push_back("surf");
    featureTypes.push_back("orb");
    featureTypes.push_back("brisk");
    ValuesConstraint<string> cmdFeatureTypes( featureTypes );
    ValueArg<string> cmdFeature("f", "feature", "feature type", true, "", &cmdFeatureTypes, cmd);
    
    ValueArg<int>    cmdTilt ("", "max_tilt", "if not set, incremental match", false, -1, "int", cmd);
    ValueArg<float>  cmdThresh ("t", "threshold", "threshold for matcher in interval [0 1]", false, -1, "float", cmd);
    ValueArg<string> cmd1st ("1", "1st", "1st image file path", true, "", "string", cmd);
    ValueArg<string> cmd2nd ("2", "2nd", "2nd image file path", true, "", "string", cmd);
    ValueArg<string> cmdOut  ("o", "output", "file path for matches", false, "/dev/null", "string", cmd);
    ValueArg<int>    cmdScreenWidth ("", "screenwidth", "for display", false, 1350, "int", cmd);
    MultiSwitchArg   cmdVerbose ("v", "", "level of verbosity of output", cmd);
    SwitchArg        cmdDisableImshow ("", "disable_image", "don't show image", cmd);
    
    cmd.parse(argc, argv);
    string           featureType    = cmdFeature.getValue();
    int              maxTilt        = cmdTilt.getValue();
    float            thres          = cmdThresh.getValue();
    string           imageName1     = cmd1st.getValue();
    string           imageName2     = cmd2nd.getValue();
    string           outName       = cmdOut.getValue();
    int              screenWidth    = cmdScreenWidth.getValue();
    bool             disableImshow  = cmdDisableImshow.getValue();
    int              verbose        = cmdVerbose.getValue();
    
    // file for output
    path outPath = absolute(path(outName));
    if (! exists(outPath.parent_path()))
    {
        cerr << "parent path " << outPath.parent_path() << " doesn't exist." << endl;
        return -1;
    }
    if (is_directory(outPath))
    {
        cerr << "need a filename, not a directory: " << outPath << endl;
        return -1;
    }
    
    Mat im1, im2;
    if (!evg::loadImage(imageName1, im1)) return 0;
    if (!evg::loadImage(imageName2, im2)) return 0;
    
    
    Ptr<FeatureDetector> detector = newFeatureDetector (featureType);
    Ptr<DescriptorExtractor> extractor = newDescriptorExtractor (featureType);
    Ptr<DescriptorMatcher> matcher = newMatcher (featureType, verbose);
    
    Ptr<cv::affma::AffMatcherHelper> affMatcherHelper = cv::affma::createAffMatcherHelper (detector, extractor, matcher);
    
    
    vector<KeyPoint> keypoints1, keypoints2;
    vector<DMatch> matches;
        
    if (maxTilt >= 0)
        affMatcherHelper->matchWithMaxTilt (im1, im2, keypoints1, keypoints2, matches, thres, maxTilt);
    else
        affMatcherHelper->matchIncreasingTilt (im1, im2, keypoints1, keypoints2, matches, thres);
    
    if (!disableImshow)
    {
        Mat im1gray, im2gray;
        cvtColor(im1, im1gray, CV_RGB2GRAY);
        cvtColor(im2, im2gray, CV_RGB2GRAY);
        float factor = float(screenWidth) / im1gray.cols / 2;
        vector<KeyPoint> keypoints1im = keypoints1, keypoints2im = keypoints2;
        for (int i = 0; i != keypoints1im.size(); ++i)
        {
            keypoints1im[i].pt.x = keypoints1im[i].pt.x * factor;
            keypoints1im[i].pt.y = keypoints1im[i].pt.y * factor;
        }
        for (int i = 0; i != keypoints2im.size(); ++i)
        {
            keypoints2im[i].pt.x = keypoints2im[i].pt.x * factor;
            keypoints2im[i].pt.y = keypoints2im[i].pt.y * factor;
        }
        
        resize(im1gray, im1gray, Size(), factor, factor);
        resize(im2gray, im2gray, Size(), factor, factor);
        Mat imgMatches;
        drawMatches (im1gray, keypoints1im, im2gray, keypoints2im, matches, imgMatches,
                     Scalar::all(-1), Scalar::all(-1),
                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        imshow( "matches", imgMatches );
        if (waitKey(0) == 27) return 0;
    }


    // write results
    evg::writeSimpleMatches (outPath.string(), imageName1, imageName2, keypoints1, keypoints2, matches);
    
    
    return 0;
}

