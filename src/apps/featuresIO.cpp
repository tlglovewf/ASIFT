//
//  Created by Evgeny on 9/17/13.
//  Copyright (c) 2013 Evgeny Toropov. All rights reserved.
//

#include <fstream>
#include <sstream>
#include <stdexcept>

#include <boost/filesystem.hpp>

#include "featuresIO.h"

namespace cv {
namespace evg {

using namespace std;
using namespace boost::filesystem;


bool is_little_endian()
{
    int n = 1;
    bool little = (*(char *)&n == 1);
    return little;
}
bool is_big_endian() { return !is_little_endian(); }

void swap_host_big_endianness_8 (void *dst, void* src)
{
  char *dst_ = (char*) dst ;
  char *src_ = (char*) src ;
  if (is_big_endian())
  {
    dst_ [0] = src_ [0] ;
    dst_ [1] = src_ [1] ;
    dst_ [2] = src_ [2] ;
    dst_ [3] = src_ [3] ;
    dst_ [4] = src_ [4] ;
    dst_ [5] = src_ [5] ;
    dst_ [6] = src_ [6] ;
    dst_ [7] = src_ [7] ;
  } else {
    dst_ [0] = src_ [7] ;
    dst_ [1] = src_ [6] ;
    dst_ [2] = src_ [5] ;
    dst_ [3] = src_ [4] ;
    dst_ [4] = src_ [3] ;
    dst_ [5] = src_ [2] ;
    dst_ [6] = src_ [1] ;
    dst_ [7] = src_ [0] ;
  }
}

double correct_endiannes (double src)
{
    double dst;
    swap_host_big_endianness_8 (&dst, &src);
    return dst;
}





/// ================================   VLFeat  ==================================



bool readVLFeatFileAscii (const std::string& filepath,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::Mat& descriptors)
{
    try {
        std::ifstream ifs (filepath.c_str());
        if (!ifs) throw runtime_error("evg::readVLFeatFile: cannot open file of vlfeat format");
        
        // clear
        keypoints = vector<KeyPoint> ();
        descriptors = Mat ();
        
        // read line by line
        while (true)
        {
            string line;
            getline (ifs, line);
            
            // condition to exit the loop
            if (ifs.eof()) break;
            
            if (!ifs) throw runtime_error("evg::readVLFeatFile: failed reading a line '" + line + "'");
            
            istringstream iss(line);
        
            // read header (4 floats in format "x y size angle")
            float x, y, size, angle;
            iss >> x >> y >> size >> angle;
            keypoints.push_back(KeyPoint (x, y, size, angle));
            if (!ifs) throw runtime_error("evg::readVLFeatFile: cannot read keypoint data");
            
            // read descriptor (N uchar-s)
            vector<uchar> descrVector;
            unsigned int value;
            while (iss >> value)
            {
                descrVector.push_back(value);
                if (value > 255)
                    throw runtime_error("evg::readVLFeatFile: descritptors are not in range [0 255]");
            }
            
            if (descrVector.size() == 0)
                throw runtime_error("evg::readVLFeatFile: descriptor size was 0");
            
            if (descriptors.rows > 0 && descriptors.cols != descrVector.size())
                throw runtime_error("evg::readVLFeatFile: inconsistent descriptor size across lines");
            
            Mat descriptor = Mat(descrVector).t();
            descriptors.push_back(descriptor);
        }
        descriptors.convertTo (descriptors, CV_8U);
                    
        ifs.close();
        return true;
    } catch (exception& e) {
        cerr << e.what() << endl;
        return false;
    }
}


bool readVLFeatFileBin (const std::string& filepath,
                        std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors)
{
    const int DescrSize = 128;
    
    try {
        std::ifstream ifs (filepath.c_str(), ios::binary);
        if (!ifs) throw runtime_error("evg::readVLFeatFile: cannot open file of vlfeat format");
        
        // clear
        keypoints = vector<KeyPoint> ();
        descriptors = Mat ();
        
        // read line by line
        while (true)
        {
            // read header (4 doubles in format "x y size angle")
            const int HeaderSize = 4;
            double header[HeaderSize];
            ifs.read ((char*)&header, sizeof(header));
            
            // condition to exit the cycle
            if (ifs.eof())
                break;
            
            if (!ifs)
                throw runtime_error("evg::readVLFeatFile: cannot read keypoint data");

            for (int i = 0; i != HeaderSize; ++i)
                header[i] = correct_endiannes(header[i]);

            float x, y, size, angle;
            x = float(header[0]);
            y = float(header[1]);
            size = float(header[2]);
            angle = float(header[3]);
            keypoints.push_back(KeyPoint (x, y, size, angle));
            
            // read descriptor (128 uchar-s)
            uchar body[DescrSize];
            ifs.read ((char*)&body, sizeof(body));
            if (!ifs)
                throw runtime_error("evg::readVLFeatFile: cannot read descriptor data");

            vector<uchar> descrVector (body, body + DescrSize);
            Mat descriptor = Mat(descrVector).t();

            if (descriptor.cols != DescrSize)
                throw runtime_error("evg::readVLFeatFile: descriptor size should be 128");
            
            descriptors.push_back(descriptor);
        }
        descriptors.convertTo (descriptors, CV_8U);
                    
        ifs.close();
        return true;
    } catch (exception& e) {
        cerr << e.what() << endl;
        return false;
    }
}


bool writeVLFeatFileAscii (const std::string& filepath,
                           const std::vector<cv::KeyPoint>& keypoints,
                           const cv::Mat& descriptors)
{
    try {
        if (keypoints.size() != descriptors.rows)
            throw runtime_error("evg::writeVLFeatFileAscii: keypoints number != descriptors number");
        
        if (descriptors.type() != CV_8U)
            throw runtime_error("evg::writeVLFeatFileAscii: descriptors Mat type is not CV_8U");
    
        std::ofstream ofs (filepath.c_str());
        if (!ofs) throw runtime_error("evg::writeVLFeatFileAscii: cannot open file for ascii writing");
        
        for (int iKey = 0; iKey < descriptors.rows; ++iKey)
        {
            // write keypoint: y x size angle
            const KeyPoint& key = keypoints[iKey];
            ofs << key.pt.x << ' ' << key.pt.y << ' ' << key.size << ' ' << key.angle << ' ';
            
            // write descriptor in the same line: descr[0] descr[1] ... descr[N]
            const uchar* p = descriptors.ptr (iKey);
            for (int iBin = 0; iBin < descriptors.cols; ++iBin)
                ofs << int(*p++) << ((iBin == descriptors.cols - 1) ? "" : " ");
            ofs << endl;
        }
        
        ofs.flush();
        ofs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


bool writeVLFeatFileBin (const std::string& filepath,
                         const std::vector<cv::KeyPoint>& keypoints,
                         const cv::Mat& descriptors)
{
    try {
        if (keypoints.size() != descriptors.rows)
            throw runtime_error("writeVLFeatFileBin: keypoints number != descriptors number");
    
        std::ofstream ofs (filepath.c_str(), ios::binary);
        if (!ofs) throw runtime_error("evg::writeVLFeatFileBin: cannot open file for bin writing");
        
        for (int row = 0; row < descriptors.rows; ++row)
        {
            // write keypoint: x y size angle
            const KeyPoint& key = keypoints[row];
            const int HeaderSize = 4;
            double header[HeaderSize];
            header[0] = correct_endiannes(double(key.pt.x));;
            header[1] = correct_endiannes(double(key.pt.y));;
            header[2] = correct_endiannes(double(key.size));;
            header[3] = correct_endiannes(double(key.angle));;
            ofs.write((char*)&header, sizeof(header));
            
            // write descriptor in the same line: descr[0] descr[1] ... descr[N]
            const uchar* p = descriptors.ptr(row);
            ofs.write((char*)p, descriptors.cols);
        }
        
        ofs.flush();
        ofs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


/// ---------------------------------------------------------------------- ///


bool readVLFeatFormat (const std::string& filepath,
                       std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                       bool binFormat)
{
    if (binFormat)
        return readVLFeatFileBin (filepath, keypoints, descriptors);
    else
        return readVLFeatFileAscii (filepath, keypoints, descriptors);
}


bool writeVLFeatFormat (const std::string& filepath,
                        const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
                        bool binFormat)
{
    if (binFormat)
        return writeVLFeatFileBin (filepath, keypoints, descriptors);
    else
        return writeVLFeatFileAscii (filepath, keypoints, descriptors);
}




/// ================================    Ubc   ==================================



bool readUbcFileAscii (const std::string& filepath,
                       std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    try {
        std::ifstream ifs (filepath.c_str());
        if (!ifs) throw runtime_error("evg::readUbcFileAscii: cannot open file of ubc format");
        
        // clear
        keypoints = vector<KeyPoint> ();
        descriptors = Mat ();
        
        // read global header
        int numPoints, descrSize;
        string line;
        getline (ifs, line);
        istringstream iss(line);
        iss >> numPoints >> descrSize;

        if (!iss) throw runtime_error("evg::readUbcFileAscii: failed reading header from line '" + line + "'");
        
        // read line by line
        while (true)
        {
            string line;
            getline (ifs, line);
            
            // condition to exit the loop
            if (ifs.eof()) break;
        
            // read header (4 floats in format "y x size angle")
            float x, y, size, angle;
            istringstream iss(line);
            iss >> y >> x >> size >> angle;
            keypoints.push_back(KeyPoint (x, y, size, angle));
            if (!iss) throw runtime_error("evg::readUbcFileAscii: cannot read keypoint from line '" + line + "'");
            
            // read descriptor (N uchar-s)
            line = string();
            getline (ifs, line);
            if (!ifs) throw runtime_error("evg::readUbcFileAscii: cannot read descriptor line");

            istringstream iss2(line);
            vector<uchar> descrVector;
            unsigned int value;
            while (iss2 >> value)
            {
                descrVector.push_back(value);
                if (value > 255)
                    throw runtime_error("evg::readUbcFileAscii: descritptors are not in range [0 255]");
            }
            
            if (descrVector.size() == 0)
                throw runtime_error("evg::readUbcFileAscii: descriptor size was 0 from line '" + line + "'");
            
            if (descriptors.rows > 0 && descriptors.cols != descrVector.size())
                throw runtime_error("evg::readUbcFileAscii: inconsistent descriptor size across lines");
            Mat descriptor = Mat(descrVector).t();
            descriptors.push_back(descriptor);
        }
        descriptors.convertTo (descriptors, CV_8U);
                    
        if (descriptors.rows != numPoints)
            throw runtime_error("evg::readUbcFileAscii: read different number of points from declared");
            
        if (descriptors.cols != descrSize)
            throw runtime_error("evg::readUbcFileAscii: descriptor size differs from the declared");

        ifs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


bool writeUbcFileAscii  (const std::string& filepath,
                         const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
{
    try {
        if (keypoints.size() != descriptors.rows)
            throw runtime_error("evg::writeUbcFileAscii: keypoints number != descriptors number");
    
        if (descriptors.type() != CV_8U)
            throw runtime_error("evg::writeUbcFileAscii: descriptors Mat type is not CV_8U");
    
        std::ofstream ofs (filepath.c_str());
        if (!ofs) throw runtime_error("evg::writeUbcFileAscii: cannot open file for ascii writing");
        
        // write number of keypoints and descriptor size
        ofs << descriptors.rows << ' ' << descriptors.cols << endl;
        
        for (int iKey = 0; iKey != keypoints.size(); ++iKey)
        {
            // write keypoint header
            const KeyPoint& key = keypoints[iKey];
            ofs << ' ' << key.pt.y << ' ' << key.pt.x << ' ' << key.size << ' ' << key.angle << endl;
            
            // write descriptor: descr[0] descr[1] ... descr[N]
            const uchar* p = descriptors.ptr (iKey);
            for (int iBin = 0; iBin < descriptors.cols; ++iBin)
                ofs << ' ' << int(*p++);
            ofs << endl;
        }
        
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


/// ---------------------------------------------------------------------- ///


bool readUbcFileBin (const std::string& filepath,
                     std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    const int DescrSize = 128;
    
    try {
        std::ifstream ifs (filepath.c_str(), ios::binary);
        if (!ifs) throw runtime_error("evg::readUbcFile: cannot open file of ubc format");
        
        // clear
        keypoints = vector<KeyPoint> ();
        descriptors = Mat ();
        
        int header1[2];
        ifs.read ((char*)&header1, sizeof(header1));
        int header2[3];
        ifs.read ((char*)&header2, sizeof(header2));
        
        // ground truth
        int name = ('S'+ ('I'<<8)+('F'<<16)+('T'<<24));
        int version1 = ('V'+('4'<<8)+('.'<<16)+('0'<<24));
        int version2 = ('V'+('5'<<8)+('.'<<16)+('0'<<24));
        
        // the header's 1-8 bytes are either SIFTV4.0 or SIFTV5.0
        if (header1[0] != name || (header1[1] != version1 && header1[1] != version2))
            throw runtime_error("evg::readUbcFile: binary file in ubc (Lowe's) format must "
                              "start with SIFTV4.0 or SIFTV5.0");
        
        // the header's 9-11 bytes are [NPoints 5 128]
        int numPoints = header2[0];
        if (header2[1] != 5 || header2[2] != DescrSize)
            throw runtime_error("evg::readUbcFile: bytes 9-12 in binary file in ubc (Lowe's) format "
                              "are [numPoints 5 128]");
        
        // read line by line
        for (int i = 0; i != numPoints; ++i)
        {
            // read header (5 floats in format "x y color size angle")
            const int KeypointSize = 5;
            float header[KeypointSize];
            ifs.read ((char*)&header, sizeof(header));
            
            if (!ifs)
                throw runtime_error("evg::readUbcFile: cannot read keypoint data");

            //for (int i = 0; i != KeypointSize; ++i)
            //    header[i] = correct_endiannes(header[i]);

            float x, y, color, size, angle;
            x = header[0];
            y = header[1];
            color = header[2]; // is never used
            size = header[3];
            angle = header[4];
            keypoints.push_back(KeyPoint (x, y, size, angle));
            
//            cout << "keypoints: " << x << " " << y << " " << color << " " << size << " " << angle << endl;
            
            // read descriptor (128 uchar-s)
            uchar descr[DescrSize];
            ifs.read ((char*)&descr, sizeof(descr));
            if (!ifs)
                throw runtime_error("evg::readUbcFile: cannot read descriptor data");

            vector<uchar> descrVector (descr, descr + DescrSize);
            Mat descriptor = Mat(descrVector).t();
            
//            for (int j = 0; j != 100; ++j)
//                cout << (int) descr[j] << ", " << flush;
//            cout << endl;

            if (descriptor.cols != DescrSize)
                throw runtime_error("evg::readUbcFile: descriptor size should be 128");
            
            descriptors.push_back(descriptor);
        }
        descriptors.convertTo (descriptors, CV_8U);
                    
        // read footer (4 doubles in format "x y size angle")
        int footer;
        ifs.read ((char*)&footer, sizeof(footer));
        
        // try to read again. If didn't yet reach the eof, return success but print a warning
        char dummy;
        ifs.read ((char*)&dummy, sizeof(dummy));
        if (!ifs.eof())
            cerr << "evg::readUbcFile: warning: number of points in the file seems "
                    "acceed the number declared in the header" << endl;
        
        ifs.close();
        return true;
    } catch (exception& e) {
        cerr << e.what() << endl;
        return false;
    }
}


bool writeUbcFileBin (const std::string& filepath,
                      const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors)
{
    try {
        if (keypoints.size() != descriptors.rows)
            throw runtime_error("writeUbcFileBin: keypoints number != descriptors number");
    
        std::ofstream ofs (filepath.c_str(), ios::binary);
        if (!ofs) throw runtime_error("evg::writeUbcFileBin: cannot open file for bin writing");
        
        // header
        char name[4]    = {'S', 'I', 'F', 'T'};
        char version[4] = {'V', '4', '.', '0'};
        ofs.write((char*)&name, sizeof(name));
        ofs.write((char*)&version, sizeof(version));
        
        int numPoints = descriptors.cols;
        int KeypointSize = 5;
        int DescrSize = 128;
        ofs.write((char*)&numPoints, sizeof(numPoints));
        ofs.write((char*)&KeypointSize, sizeof(KeypointSize));
        ofs.write((char*)&DescrSize, sizeof(DescrSize));
        if (!ofs) throw runtime_error("evg::writeUbcFileBin: cannot write the header");
        
        for (int row = 0; row < descriptors.rows; ++row)
        {
            // write keypoint: x y color size angle
            const KeyPoint& key = keypoints[row];
            float header[KeypointSize];
            // FIXME: set endianness to Win format
            header[0] = key.pt.x;
            header[1] = key.pt.y;
            header[2] = 0;
            header[3] = key.size;
            header[4] = key.angle;
            ofs.write((char*)&header, sizeof(header));
            if (!ofs) throw runtime_error("evg::writeUbcFileBin: cannot write the keypoint");
            
            // write descriptor in the same line: descr[0] descr[1] ... descr[N]
            const uchar* p = descriptors.ptr(row);
            ofs.write((char*)p, descriptors.cols);
            if (!ofs) throw runtime_error("evg::writeUbcFileBin: cannot write the descriptor");
        }
        
        char eof_marker[4] = {static_cast<char>(0xff), 'E', 'O', 'F'};
        ofs.write((char*)&eof_marker, sizeof(eof_marker));
        if (!ofs) throw runtime_error("evg::writeUbcFileBin: cannot write the footer");
 
        ofs.flush();
        ofs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}




bool readUbcFormat  (const std::string& filepath,
                     std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
                     bool binFormat)
{
    if (binFormat)
        return readUbcFileBin (filepath, keypoints, descriptors);
    else
        return readUbcFileAscii (filepath, keypoints, descriptors);
}


bool writeUbcFormat (const std::string& filepath,
                     const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
                     bool binFormat)
{
    if (binFormat)
        return writeUbcFileBin (filepath, keypoints, descriptors);
    else
        return writeUbcFileAscii (filepath, keypoints, descriptors);
}




/// ============================ Matches =============================


bool writeVsfmMatches (const string& filepath, const string& imName1, const string& imName2,
                       const vector<DMatch>& matches)
{
    try {
        std::ofstream ofs (filepath.c_str(), ios::binary);
        if (!ofs) throw runtime_error("evg::writeVsfmMatches: cannot open file for writing");
        
        // header
        ofs << imName1 << ' ' << imName2 << ' ' << matches.size() << '\n';
        
        // matches
        for (int i = 0; i != matches.size(); ++i)
            ofs << matches[i].queryIdx << ' ';
        ofs << '\n';
        for (int i = 0; i != matches.size(); ++i)
            ofs << matches[i].trainIdx << ' ';
        ofs << '\n';
 
        ofs.flush();
        ofs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


bool readVsfmMatches    (const string& filepath, string& imName1, string& imName2,
                         vector<DMatch>& matches)
{
    try {
        std::ifstream ifs (filepath.c_str());
        if (!ifs) throw runtime_error("evg::readVsfmMatches: cannot open file: " + filepath);
        
        // header
        int numMatches;
        ifs >> imName1 >> imName2 >> numMatches;
        if (!ifs) throw runtime_error("evg::readVsfmMatches: failed reading the header");
        
        matches = vector<DMatch> (numMatches);
        
        // two lines
        for (int iMatch = 0; iMatch != numMatches; ++iMatch)
        {
            if (!ifs) throw runtime_error("evg::readVsfmMatches: failed reading 1st line");
            ifs >> matches[iMatch].queryIdx;
        }
        for (int iMatch = 0; iMatch != numMatches; ++iMatch)
        {
            if (!ifs) throw runtime_error("evg::readVsfmMatches: failed reading 2nd line");
            ifs >> matches[iMatch].trainIdx;
        }

        // make sure there is nothing else there
        string dummy;
        ifs >> dummy;
        if (!ifs.eof())
            cerr << "evg::readVsfmMatches: warning: number of matches in the file seems to "
                    "acceed the number declared in the header" << endl;
        
        ifs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}


bool writeSimpleMatches (const std::string& outFileName,
                         const std::string& imName1, const std::string& imName2,
                         const std::vector<cv::KeyPoint>& keypoints1,
                         const std::vector<cv::KeyPoint>& keypoints2,
                         const std::vector<cv::DMatch>& matches)
{
    try {
        path outFilePath (outFileName);
        if (is_directory(outFilePath))
        {
            cerr << "writeSimpleMatches: Need a filename, not a directory: " << outFilePath << endl;
            return 0;
        }
        if (!exists(outFilePath.parent_path()))
        {
            cerr << "writeSimpleMatches: Parent path does not exist for: " << outFilePath << endl;
            return 0;
        }
        
        std::ofstream ofs (outFilePath.string().c_str(), ios::binary);
        if (!ofs) throw runtime_error("evg::writeSimpleMatches: cannot open file for writing");
        
        // header
        ofs << imName1 << endl;
        ofs << imName2 << endl;
        ofs << matches.size() << endl;
        ofs << "x1 y1 x2 y2" << endl;
        
        // matches
        for (int i = 0; i != matches.size(); ++i)
        {
            if (keypoints1.size() < matches[i].queryIdx || keypoints2.size() < matches[i].trainIdx)
            {
                cerr << "match " << i << " refers to an out-of-range keypoint." << endl;
                return 0;
            }
            const Point2f& pt1 = keypoints1[matches[i].queryIdx].pt;
            const Point2f& pt2 = keypoints2[matches[i].trainIdx].pt;
            ofs << pt1.x << " " << pt1.y << " " << pt2.x << " " << pt2.y << "\n";
        }

        ofs.flush();
        ofs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}



bool readSimpleMatches  (const std::string& filepath,
                         std::string& imName1, std::string& imName2,
                         std::vector<cv::KeyPoint>& keypoints1,
                         std::vector<cv::KeyPoint>& keypoints2,
                         std::vector<cv::DMatch>& matches)
{
    try {
        std::ifstream ifs (filepath.c_str());
        if (!ifs) throw runtime_error("evg::readSimpleMatches: cannot open file: " + filepath);
        
        string line;

        // header
        
        // image names
        getline(ifs, imName1);
        getline(ifs, imName2);
        
        // number of matches
        getline(ifs, line);
        istringstream iss(line);
        int n;
        iss >> n;
        keypoints1.reserve(n);
        keypoints2.reserve(n);
        matches.reserve(n);

        // dummy line of format: "x1 y1 x2 y2"
        getline(ifs, line);
    
        if (!ifs) throw runtime_error("evg::readSimpleMatches: failed to read header");
        
        // data
        for (int i = 0; getline(ifs, line); ++i)
        {
            float x1, y1, x2, y2;
            istringstream iss (line);
            iss >> x1 >> y1 >> x2 >> y2;
            keypoints1.push_back( KeyPoint(Point2f(x1, y1), 1, 1) );
            keypoints2.push_back( KeyPoint(Point2f(x2, y2), 1, 1) );
            matches.push_back( DMatch(i, i, 1) );
        }
                
        ifs.close();
        return true;
    } catch(exception& e) {
        cerr << e.what() << endl;
        return false;
    }        
}




} // namespace evg
} // namespace cv


