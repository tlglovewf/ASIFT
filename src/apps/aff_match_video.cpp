//
//  aff_match_video.cpp
//
//  Created by Evgeny on 7/22/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <set>
#include <map>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tclap/CmdLine.h>

#include "mediaIO.h"
#include "aff_features2d.hpp"
#include "featuresIO.h"


using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace TCLAP;
using namespace cv::affma;


static cv::FeatureDetector* newFeatureDetector (const std::string& featureType)
{
    // FeatureDetector* detector (NULL);
    // if (featureType == "sift")
    //     detector = new SIFT();
    // else if (featureType == "surf")
    //     detector = new SURF();
    // else if (featureType == "orb")
    //     detector = new ORB();
    // else if (featureType == "brisk")
    //     detector = new BRISK();
    // else assert(0);
    // return detector;
    return new SIFT();
}


static cv::DescriptorExtractor* newDescriptorExtractor (const string& featureType)
{
    // DescriptorExtractor* extractor (NULL);
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
    CmdLine cmd ("match video frames between themselves, frames match are given in a file");
    
    vector<string> featureTypes;
    featureTypes.push_back("sift");
    featureTypes.push_back("surf");
    featureTypes.push_back("orb");
    featureTypes.push_back("brisk");
    ValuesConstraint<string> cmdFeatureTypes( featureTypes );
    ValueArg<string> cmdFeature("f", "feature", "feature type", true, "", &cmdFeatureTypes, cmd);

    ValueArg<int>    cmdTilt ("", "max_tilt", "if not set, incremental match", false, -1, "int", cmd);
    ValueArg<float>  cmdThresh ("t", "threshold", "threshold for matcher", false, -1, "float", cmd);
    ValueArg<string> cmdInVideo ("i", "input", "input video", true, "", "string", cmd);
    ValueArg<string> cmdInFramePairs ("", "framepairs", "file with frame pairs", true, "", "string", cmd);
    ValueArg<string> cmdOutDirName ("o", "output", "output dir. for matches", false, "/dev/null", "string", cmd);
    ValueArg<string> cmdTimeFile ("", "time_name", "write a file with times", false, "", "string", cmd);
    SwitchArg        cmdDisableImshow ("", "disable_image", "don't show image", cmd);
    ValueArg<int>    cmdScreenWidth ("", "screenwidth", "for display", false, 1350, "int", cmd);
    MultiSwitchArg   cmdVerbose ("v", "", "level of verbosity of output", cmd);
    
    cmd.parse(argc, argv);
    string           featureType    = cmdFeature.getValue();
    int              maxTilt        = cmdTilt.getValue();
    float            threshold      = cmdThresh.getValue();
    string           inVideoName    = cmdInVideo.getValue();
    string           inPairsName    = cmdInFramePairs.getValue();
    string           outDirName     = cmdOutDirName.getValue();
    string           outTimeName    = cmdTimeFile.getValue();
    int              screenWidth    = cmdScreenWidth.getValue();
    bool             disableImshow  = cmdDisableImshow.getValue();
    int              verbose        = cmdVerbose.getValue();
    
    // dir for output
    path outDirPath (outDirName);
    if (!exists(outDirPath))
    {
        cerr << "directory path " << outDirPath << " doesn't exist." << endl;
        return -1;
    }
    if (!is_directory(outDirPath) && outDirName != "/dev/null")
    {
        cerr << "need a directory, not a filename: " << outDirPath << endl;
        return -1;
    }
    
    // file for time output
    path outTimePath = outDirPath / outTimeName;
    std::ofstream ofsTime (outTimePath.string());
    ofsTime << "im1 im2 time(s)" << endl;
    
    // validate video
    VideoCapture video = evg::openVideo(inVideoName);
    if( !video.isOpened() )
    {
        cerr << "Error when reading video" << endl;
        return -1;
    }
    
    // validate frame pairs
    path inPairsPath (inPairsName);
    if (! exists(inPairsPath))
    {
        cerr << "file with frame pairs to match " << inPairsPath << " doesn't exist." << endl;
        return -1;
    }
    
    // read frame pairs
    Mat pairsMat = evg::dlmread(inPairsName);
    if (pairsMat.cols != 2)
    {
        cerr << "inPairsName should have two columns" << endl;
        return -1;
    }
    assert (pairsMat.depth() == CV_32F); // this is the current behavior of evg::dlmread
    map<int, set<int> > framePairs;
    for (int row = 0; row != pairsMat.rows; ++row)
    {
        int im1 (pairsMat.at<float>(row,0));
        int im2 (pairsMat.at<float>(row,1));
        framePairs[im1].insert(im2);
    }
    
    
    Ptr<FeatureDetector> detector = newFeatureDetector (featureType);
    Ptr<DescriptorExtractor> extractor = newDescriptorExtractor (featureType);
    Ptr<DescriptorMatcher> matcher = newMatcher (featureType, verbose);
    
    Ptr<AffMatcherHelper> affMatcherHelper = createAffMatcherHelper (detector, extractor, matcher);
    affMatcherHelper->setVerbosity(verbose);
    
    Mat frame;
    map<int, Mat> frames;
    
    for (int im2 = 0; ; ++im2)
    {
        if (verbose > 2)
            cout << "frame: " << im2 << endl;
        
        video >> frame;
        if (frame.empty())
        {
            cout << "finished at frame " << im2 << endl;
            break;
        }
        frames[im2] = frame.clone();
        
        set<int> framesToRemove;
        for (auto im1it : frames)
        {
            int im1 = im1it.first;
            if (framePairs.count(im1) && framePairs.at(im1).count(im2))
            {
                if (verbose > 0)
                    cout << "frame pair: " << im1 << " " << im2 << endl;

                vector<KeyPoint> keypoints1, keypoints2;
                vector<DMatch> matches;
                
                if (maxTilt >= 0)
                    affMatcherHelper->matchWithMaxTilt (frames[im1], frames[im2],
                                          keypoints1, keypoints2, matches, threshold, maxTilt);
                else
                    affMatcherHelper->matchIncreasingTilt (frames[im1], frames[im2],
                                          keypoints1, keypoints2, matches, threshold);

                //const float cutoff = 1.f;
                //filterDuplicateMatches (keypoints1, keypoints2, matches, cutoff);
                
                if (!disableImshow)
                {
                    Mat im1gray, im2gray;
                    cvtColor(frames[im1], im1gray, CV_RGB2GRAY);
                    cvtColor(frames[im2], im2gray, CV_RGB2GRAY);
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
                if (outDirName != "/dev/null")
                {
                    ostringstream im1str, im2str;
                    im1str << im1;
                    im2str << im2;
                    string outMatchesName = "matches-" + im1str.str() + "-" + im2str.str() + ".txt";
                    path outMatchesPath = outDirPath / outMatchesName;
                    evg::writeSimpleMatches (outMatchesPath.string(), im1str.str(), im2str.str(),
                                             keypoints1, keypoints2, matches);
                }
                
                // remove processed pair
                assert (framePairs.at(im1).count(im2));
                framePairs.at(im1).erase(im2);
                if (framePairs.at(im1).empty())
                    framesToRemove.insert(im1);
            }
        }
        
        for (int im1 : framesToRemove)
        {
            frames.erase(im1);
            if (verbose > 1)
                cout << "removed frame: " << im1 << endl;
        }

    }
    ofsTime.close();
    
    return 0;
}

