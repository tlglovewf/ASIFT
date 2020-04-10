/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  An OpenCV Implementation of affine-covariant matching (matching with different viewpoints)
//  Further Information Refer to:
//  Author: Evgeny Toropov
//  etoropov@andrew.cmu.edu
//
// IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
// 
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
// 
// 
//                           License Agreement
//                For Open Source Computer Vision Library
// 
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2008-2013, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
// 
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
// 
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
// 
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
// 
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include <iostream>
#include <map>
#include <algorithm>

//#include "precomp.hpp"

#include "aff_features2d.hpp"


using namespace std;

namespace cv { namespace affma {



/****************************************************************************************\
*                                  Matchers                                              *
\****************************************************************************************/


class AffDescriptorMatcherImpl : public AffDescriptorMatcher {
private:

    class View {
        vector<int>       _bookkeeping;
        vector<KeyPoint>  _keys;
        Mat               _descriptors;
    public:
        void              add (const KeyPoint& keypoint, const Mat& descr, const int globalIdx);
        vector<KeyPoint>  getKeypoints() const   { return _keys; }
        Mat               getDescriptors() const { return _descriptors; }
        int               getGlobalIdx(int localIdx) const;
        void              reserve (unsigned long n);
    };
    
    typedef pair<int, int> ViewIdPair;
    typedef vector< vector<DMatch> > DMatchesVector;
    
    void               splitByViews (const vector<KeyPoint>& queryKeys,
                                     const vector<KeyPoint>& trainKeys,
                                     const Mat& queryDescriptors, const Mat& trainDescriptors,
                                     vector<View>& queryViews, vector<View>& trainViews) const;
    
    void               combineFromViews
                                    (const map<ViewIdPair, DMatchesVector>& matchesByView,
                                     const vector<View>& queryViews, const vector<View>& trainViews,
                                     CV_OUT vector<vector<DMatch> >& matches) const;

private:

    // underlying matcher
    Ptr<DescriptorMatcher> _matcher;

    // view pairs that are actually matched
    std::set<ViewIdPair>   _viewPairsPool;
    
    
public:
    AffDescriptorMatcherImpl (const Ptr<DescriptorMatcher>& matcher_)
       : _matcher(matcher_) { CV_Assert(_matcher != NULL); }
    
    virtual ~AffDescriptorMatcherImpl() { }
    
    bool    isMaskSupported() const      { return _matcher->isMaskSupported(); }

    void    setViewPairsPool( std::set< ViewIdPair > viewPairsPool );

    void    match(       const vector<KeyPoint>& queryKeypoints,
                         const vector<KeyPoint>& trainKeypoints,
                         const Mat& queryDescriptors, const Mat& trainDescriptors,
                         CV_OUT vector<DMatch>& matches, const Mat& mask=Mat() ) const;

    void    knnMatch(    const vector<KeyPoint>& queryKeypoints,
                         const vector<KeyPoint>& trainKeypoints,
                         const Mat& queryDescriptors, const Mat& trainDescriptors,
                         CV_OUT vector<vector<DMatch> >& matches, int k,
                         const Mat& mask=Mat(), bool compactResult=false ) const;

    void    radiusMatch( const vector<KeyPoint>& queryKeypoints,
                         const vector<KeyPoint>& trainKeypoints,
                         const Mat& queryDescriptors, const Mat& trainDescriptors,
                         vector<vector<DMatch> >& matches, float maxDistance,
                         const Mat& mask=Mat(), bool compactResult=false ) const;
};


Ptr<AffDescriptorMatcher> createAffDescriptorMatcher (const Ptr<DescriptorMatcher>& matcher_)
{
    return new AffDescriptorMatcherImpl (matcher_);
}


void AffDescriptorMatcherImpl::View::add (const KeyPoint& key, const Mat& descr, const int globalIdx)
{
    _bookkeeping.push_back(globalIdx);
    _keys.push_back(key);
    _descriptors.push_back(descr);
}

int AffDescriptorMatcherImpl::View::getGlobalIdx(int localIdx) const
{
    CV_Assert (localIdx < _bookkeeping.size());
    return _bookkeeping.at(localIdx);
}

void AffDescriptorMatcherImpl::View::reserve (unsigned long n)
{
    _bookkeeping.reserve(n);
    _keys.reserve(n);
    _descriptors.reserve(n);
}


void AffDescriptorMatcherImpl::splitByViews (const vector<KeyPoint>& queryKeypoints,
                                             const vector<KeyPoint>& trainKeypoints,
                                             const Mat& queryDescriptors,const Mat& trainDescriptors,
                                             vector<View>& queryViews,vector<View>& trainViews) const
{
    CV_Assert (queryDescriptors.rows == queryKeypoints.size());
    CV_Assert (trainDescriptors.rows == trainKeypoints.size());
    CV_Assert (queryDescriptors.cols == trainDescriptors.cols);

    // find the number of viewpoints (numViews) used for KeyPoint detection
    int queryNumViews = 0, trainNumViews = 0;
    vector<KeyPoint>::const_iterator it;
    
    for (it = queryKeypoints.begin(); it != queryKeypoints.end(); ++it)
        queryNumViews = cv::max (it->class_id, queryNumViews);
    queryNumViews++;
    
    for (it = trainKeypoints.begin(); it != trainKeypoints.end(); ++it)
        trainNumViews = cv::max (it->class_id, trainNumViews);
    trainNumViews++;
    
    // make sure numView is resonable and KeyPoint::class_id is not used for something else
    if (queryNumViews < 0 || queryNumViews > AffAngles::MaxPossibleNumViews ||
        trainNumViews < 0 || trainNumViews > AffAngles::MaxPossibleNumViews)
    {
        cerr << "AffDescriptorMatcherImpl::splitByViews: queryNumViews == " << queryNumViews
             << ", trainNumViews == " << trainNumViews
             << ". The KeyPoint::class_id is probably used by some other tool" << endl;
    }
    CV_Assert (queryNumViews <= AffAngles::MaxPossibleNumViews && queryNumViews >= 0);
    CV_Assert (trainNumViews <= AffAngles::MaxPossibleNumViews && trainNumViews >= 0);
    
    queryViews = vector<View>(queryNumViews);
    trainViews = vector<View>(trainNumViews);
    
    // reserve some space for speed
    const double ReserveCoef = 1.5;
    for (int iView = 0; iView != queryNumViews; ++iView)
        queryViews[iView].reserve( queryKeypoints.size() / queryNumViews * ReserveCoef );
    for (int iView = 0; iView != trainNumViews; ++iView)
        trainViews[iView].reserve( trainKeypoints.size() / trainNumViews * ReserveCoef );
    
    // split query and train keypoints and descriptors into views
    for (int i = 0; i != queryKeypoints.size(); ++i)
        queryViews[ queryKeypoints[i].class_id ].add(queryKeypoints[i], queryDescriptors.row(i), i);
    for (int i = 0; i != trainKeypoints.size(); ++i)
        trainViews[ trainKeypoints[i].class_id ].add(trainKeypoints[i], trainDescriptors.row(i), i);
}


void AffDescriptorMatcherImpl::combineFromViews
                               (const map<ViewIdPair, DMatchesVector>& matchesByView,
                                const vector<View>& queryViews, const vector<View>& trainViews,
                                CV_OUT vector<vector<DMatch> >& matches_) const
{
    unsigned long numMatches = 0;
    for (int iView1 = 0; iView1 != queryViews.size(); ++iView1)
        for (int iView2 = 0; iView2 != trainViews.size(); ++iView2)
            if ( matchesByView.count( make_pair(iView1, iView2) ) )
                numMatches += matchesByView.at( make_pair(iView1, iView2) ).size();
    
    matches_.clear();
    matches_.reserve(numMatches);
    
    for (int iView1 = 0; iView1 != queryViews.size(); ++iView1)
        for (int iView2 = 0; iView2 != trainViews.size(); ++iView2)
            if ( matchesByView.count( make_pair(iView1, iView2) ) )
            {
                const DMatchesVector& viewPairMatches = matchesByView.at( make_pair(iView1, iView2) );
                for (int i = 0; i != viewPairMatches.size(); ++i)
                {
                    vector<DMatch> matchRow = viewPairMatches[i];
                    vector<DMatch> newRow (matchRow);  // .distance is copied and not changed after
                    for (unsigned long j = 0; j != matchRow.size(); ++j)
                    {
                        newRow[j].queryIdx = queryViews.at(iView1).getGlobalIdx(matchRow[j].queryIdx);
                        newRow[j].trainIdx = trainViews.at(iView2).getGlobalIdx(matchRow[j].trainIdx);
                    }
                    matches_.push_back(newRow);
                }
            }
}


void AffDescriptorMatcherImpl::setViewPairsPool( std::set< ViewIdPair > viewPairsPool )
{
    _viewPairsPool = viewPairsPool;
}


void AffDescriptorMatcherImpl::match( const std::vector<KeyPoint>& queryKeypoints,
                                      const std::vector<KeyPoint>& trainKeypoints,
                                      const Mat& queryDescriptors, const Mat& trainDescriptors,
                                      CV_OUT std::vector<DMatch>& matches_, const Mat& mask ) const
{
    vector<vector<DMatch> > matchesKnn;
    knnMatch( queryKeypoints, trainKeypoints, queryDescriptors, trainDescriptors,
              matchesKnn, 1, mask, true );
    
    // rewrite matches in vector<DMatch> format
    matches_.clear();
    matches_.reserve(matchesKnn.size());
    for (int i = 0; i != matchesKnn.size(); ++i)
        matches_.push_back (matchesKnn[i][0]);
}


void AffDescriptorMatcherImpl::knnMatch( const vector<KeyPoint>& queryKeypoints_,
                                         const vector<KeyPoint>& trainKeypoints_,
                                         const Mat& queryDescriptors_, const Mat& trainDescriptors_,
                                         CV_OUT vector<vector<DMatch> >& matches_, int k_,
                                         const Mat& mask_, bool compactResult_) const
{
    // split by view pairs
    vector<View> queryViews, trainViews;
    splitByViews (queryKeypoints_, trainKeypoints_, queryDescriptors_, trainDescriptors_,
                  queryViews, trainViews);
        
    // match
    map<ViewIdPair, DMatchesVector> matchesByView;
    
    // if _viewPairsPool is empty viewpairs have not been set. Then match all pairs
    for (int iView1 = 0; iView1 != queryViews.size(); ++iView1)
        for (int iView2 = 0; iView2 != trainViews.size(); ++iView2)
            if ( _viewPairsPool.empty() || _viewPairsPool.count(make_pair(iView1, iView2)) )
                _matcher->knnMatch (queryViews[iView1].getDescriptors(),
                                    trainViews[iView2].getDescriptors(),
                                    matchesByView[ make_pair(iView1, iView2) ], k_);
    
    // combine view pairs
    combineFromViews (matchesByView, queryViews, trainViews, matches_);
}


void AffDescriptorMatcherImpl::radiusMatch
         ( const std::vector<KeyPoint>& queryKeypoints_, const std::vector<KeyPoint>& trainKeypoints_,
           const Mat& queryDescriptors_, const Mat& trainDescriptors_,
           std::vector<std::vector<DMatch> >& matches_,
           float maxDistance_, const Mat& mask_, bool compactResult_ ) const
{
    // split by view pairs
    vector<View> queryViews, trainViews;
    splitByViews (queryKeypoints_, trainKeypoints_, queryDescriptors_, trainDescriptors_,
                  queryViews, trainViews);
        
    // match
    map<ViewIdPair, DMatchesVector> matchesByView;
    
    // if _viewPairsPool is empty viewpairs have not been set. Then match all pairs
    for (int iView1 = 0; iView1 != queryViews.size(); ++iView1)
        for (int iView2 = 0; iView2 != trainViews.size(); ++iView2)
            if ( _viewPairsPool.empty() || _viewPairsPool.count(make_pair(iView1, iView2)) )
                _matcher->radiusMatch (queryViews[iView1].getDescriptors(),
                                       trainViews[iView2].getDescriptors(),
                                       matchesByView[ make_pair(iView1, iView2) ], maxDistance_);
    
    // combine view pairs
    combineFromViews (matchesByView, queryViews, trainViews, matches_);
}




}} // namespaces

