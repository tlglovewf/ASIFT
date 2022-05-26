#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include "mediaIO.h"

namespace cv {
namespace evg {

using namespace std;
using namespace boost::filesystem;


Mat loadImage (const std::string& imagePath)
{
    path p = absolute(path(imagePath));
    if (! exists(p) )
    {
        cerr << "evg::loadImage(): Path " << p << " does not exist." << endl;
        throw exception();
    }
    cv::Mat image = cv::imread(p.string(), CV_LOAD_IMAGE_COLOR);
    if (! image.data )
    {
        cerr << "Image " << p << " failed to open." << endl;
        throw exception();
    }
    return image;
}


bool loadImage (const std::string& imagePath, Mat& image)
{
    try {
        image = loadImage(imagePath);
        return 1;
    } catch(...) {
        cerr << "evg::loadImage(): failed." << endl;
        return 0;
    }
}


VideoCapture openVideo(const string& videoPath)
{
    path p = absolute(path(videoPath));
    if (! exists(p) )
    {
        cerr << "evg::openVideo(): Video path " << p << " does not exist." << endl;
        throw exception();
    }
    
    VideoCapture video;
    if (! video.open(p.string()) )
    {
        cerr << "evg::openVideo(): Video " << p << " failed to open." << endl;
        throw exception();
    }
    
    return video;
}


bool openVideo(const string& videoPath, VideoCapture& _video)
{
    try {
        _video = openVideo(videoPath);
        return 1;
    } catch (...) {
        cerr << "evg::openVideo(): failed." << endl;
        return 0; }
}
    
    
VideoWriter newVideo (const string& _videoOutPath, const VideoCapture& _videoIn)
{
    // get parameters from the input video
    cv::VideoCapture videoIn = _videoIn;
    int codec = (int)(videoIn.get(CV_CAP_PROP_FOURCC));
    double fps = videoIn.get(CV_CAP_PROP_FPS);
    const Size frameSize ( (int)(videoIn.get(CV_CAP_PROP_FRAME_WIDTH)),
                           (int)(videoIn.get(CV_CAP_PROP_FRAME_HEIGHT)) );
    
    // in case of camera input, set default frame rate
    const int DefaultFps = 30;
    if (fps == 0) fps = DefaultFps;
    
    cout << "evg::newVideo(video): codec code = " << codec << endl;
    cout << "                      frame rate = " << fps << endl;
    cout << "                      frame size = [" << frameSize.width << " x "
         << frameSize.height << "]" << endl;

    // check the directory path for output video
    path p = absolute(path(_videoOutPath));
    if (! exists(p.parent_path()) )
    {
        cerr << "evg::newVideo(video): output video directory path "
             << p.parent_path() << " does not exist." << endl;
        throw exception();
    }

    // open video for output
    VideoWriter videoOut;
    if (! videoOut.open(p.string(), codec, fps, frameSize) )
    {
        cerr << "evg::newVideo(video): output video " << p << " failed to open." << endl;
        throw exception();
    }
    
    return videoOut;
}


bool newVideo (const string& _videoOutPath, const VideoCapture& _videoIn,
               VideoWriter& videoOut)
{
    try {
        videoOut = evg::newVideo(_videoOutPath, _videoIn);
        return 1;
    } catch (...) {
        cerr << "evg::newVideo(video): failed." << endl;
        return 0;
    }
}


VideoWriter newVideo (const std::string& _videoOutPath, const Mat& _image,
                      const double fps, const int codec)
{
    // check if the _image can be written as a frame
    /// TODO: anything else?
    if (_image.channels() != 1 && _image.channels() != 3)
    {
        cerr << "evg::newVideo(image): Image must be of 1 or 3 channels." << endl;
        throw exception();
    }
    if (_image.depth() != CV_8U)
        cerr << "warning evg::newVideo(image): the image is not CV_8U. "
                "You will have to provide CV_8U frames for writing video." << endl;

    // set parameters
    const Size frameSize ( (int)(_image.size().width), (int)(_image.size().height) );
    bool isColor = ( _image.channels() == 3 );
    
    cout << "evg::newVideo(image): codec code = " << codec << endl;
    cout << "                      frame rate = " << fps << endl;
    cout << "                      frame size = [" << frameSize.width << " x "
         << frameSize.height << "]" << endl;
    
    // check the parent path for output video
    path p = absolute(path(_videoOutPath));
    if (! exists(p.parent_path()) )
    {
        cerr << "evg::newVideo(image): output video directory path "
             << p.parent_path() << " does not exist." << endl;
        throw exception();
    }
    
    // open video for output
    VideoWriter videoOut;
    if (! videoOut.open(p.string(), codec, fps, frameSize, isColor) )
    {
        cerr << "evg::newVideo(image): output video " << p << " failed to open." << endl;
        throw exception();
    }
    
    return videoOut;
}


bool newVideo (const string& _videoOutPath, const Mat& _image,
               const double fps, const int codec, VideoWriter& videoOut)
{
    try {
        videoOut = evg::newVideo(_videoOutPath, _image, fps, codec);
        return 1;
    } catch (...) {
        cerr << "evg::newVideo(image): failed." << endl;
        return 0;
    }
}


Mat undistortImage(const std::string& calibrationPath, const Mat& _image)
{
    // open calibration file
    path p = absolute(path(calibrationPath));
    if (! exists(p) )
    {
        cerr << "evg::calibImage(): Path " << p << " does not exist." << endl;
        throw exception();
    }
    FileStorage fs(p.string().c_str(), FileStorage::READ);
    if (! fs.isOpened() )
    {
        cerr << "evg::calibImage(): Calibration file " << p << " failed to open." << endl;
        throw exception();
    }
    
    // read calibration file
    Mat cameraMatrix, distCoeffs;
    fs["Camera_Matrix"] >> cameraMatrix;
    fs["Distortion_Coefficients"] >> distCoeffs;
    fs.release();
    
    // undistort image
    Mat resultImage;
    undistort (_image, resultImage, cameraMatrix, distCoeffs);
    
    return resultImage;
}


bool undistortImageBool (const std::string& calibrationPath, cv::Mat& image)
{
    try {
        image = undistortImage(calibrationPath, image);
        return 1;
    } catch (...) {
        cerr << "undistortImageBool(): failed." << endl;
        return 0;
    }
}


void undistortVideo (const string& _calibrationPath, VideoCapture& _videoIn,
                     const string& _videoOutPath)
{
    // open calibration file
    path p = absolute(path(_calibrationPath));
    if (! exists(p) )
    {
        cerr << "evg::undistortVideo(): Path " << p << " does not exist." << endl;
        throw exception();
    }
    FileStorage fs (p.string().c_str(), FileStorage::READ);
    if (! fs.isOpened() )
    {
        cerr << "evg::undistortVideo(): Calibration file " << p
             << " failed to open." << endl;
        throw exception();
    }
    
    // read calibration file
    Mat cameraMatrix, distCoeffs;
    fs["Camera_Matrix"] >> cameraMatrix;
    fs["Distortion_Coefficients"] >> distCoeffs;
    fs.release();

    // start output video
    VideoWriter videoOut = evg::newVideo (_videoOutPath, _videoIn);
    Mat frameIn, frameOut;
    
    // undistort video
    while (_videoIn.read(frameIn))
    {
        undistort (frameIn, frameOut, cameraMatrix, distCoeffs);
        videoOut << frameOut;
    }
}


bool undistortVideoBool (const string& _calibrationPath, VideoCapture& _videoIn,
                         const string& _videoOutPath)
{
    try {
        undistortVideo (_calibrationPath, _videoIn, _videoOutPath);
        return 1;
    } catch (...) {
        cerr << "evg::undistortVideoBool(): failed." << endl;
        return 0;
    }
}


void testImage (const cv::Mat& image)
{
    cout << "testing image..." << endl;
    namedWindow("evg_test", CV_WINDOW_AUTOSIZE);
    imshow("evg_test, press any key", image);
    waitKey();
}


bool testImageBool (const cv::Mat& image)
{
    try {
        testImage(image);
        return 1;
    } catch(...) {
        cerr << "evg::testImageBool(): failed." << endl;
        return 0;
    }
}



template<typename Tp>
Mat dlmread (const std::string& dlmfilePath, cv::Mat matrix, int row1)
{
    // open file
    path p = absolute(path(dlmfilePath));
    if (! exists(p) )
    {
        cerr << "evg::dlmread(): Path " << p << " does not exist." << endl;
        throw exception();
    }
    std::ifstream fileStream (p.string().c_str());
    if (!fileStream.good())
    {
        cerr << "evg::dlmread(): File " << p << " failed to open." << endl;
        throw exception();
    }
    
    // read file to compute the matrix size [numRows, numCols]
    string line;
    istringstream iss;
    unsigned int row = 0, col = 0, numRows = 0, numCols = 0;
    for (row = 0; getline(fileStream, line); ++row)
    {
        // process the empty line in the end of file
        if (line == "") { --row; break; }
        iss.clear();
        iss.str(line);
        double dummy;
        for (col = 0; (iss >> dummy); ++col)
            ;
        if (col > numCols) numCols = col;
    }
    numRows = row - row1;
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file " << p << std::endl;
        throw exception();
    }
    
    // rewind the file
    fileStream.clear();
    fileStream.seekg(ios_base::beg);
    
    // create the matrix of result size and of given type (8U by default)
    matrix = Mat::zeros(numRows, numCols, DataType<Tp>::channel_type);
    
    // skip header
    for (int row = 0; row != row1; ++row)
        getline(fileStream, line);
    // read file and put values into Mat
    float num;
    for (int row = 0; getline(fileStream, line) && row != matrix.rows; ++row)
    {
        iss.clear();
        iss.str(line);
        for (int col = 0; col != matrix.cols && (iss >> num); ++col)
            matrix.at<Tp>(row, col) = num;
    }
    
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file." << std::endl;
        throw exception();
    }
    
    return matrix;
}




// delimiter is always space or tab
// matrix will put zeros for missing values
Mat dlmread (const std::string& dlmfilePath, cv::Mat matrix, int row1)
{
    // open file
    path p = absolute(path(dlmfilePath));
    if (! exists(p) )
    {
        cerr << "evg::dlmread(): Path " << p << " does not exist." << endl;
        throw exception();
    }
    std::ifstream fileStream (p.string().c_str());
    if (!fileStream.good())
    {
        cerr << "evg::dlmread(): File " << p << " failed to open." << endl;
        throw exception();
    }
    
    // read file to compute the matrix size [numRows, numCols]
    string line;
    istringstream iss;
    unsigned int row = 0, col = 0, numRows = 0, numCols = 0;
    for (row = 0; getline(fileStream, line); ++row)
    {
        // process the empty line in the end of file
        if (line == "") { --row; break; }
        iss.clear();
        iss.str(line);
        double dummy;
        for (col = 0; (iss >> dummy); ++col)
            ;
        if (col > numCols) numCols = col;
    }
    numRows = row - row1;
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file " << p << std::endl;
        throw exception();
    }
    
    // rewind the file
    fileStream.clear();
    fileStream.seekg(ios_base::beg);
    
    // create the matrix of result size and of given type (8U by default)
    matrix = Mat::zeros(numRows, numCols, CV_32F);
    
    // skip header
    for (int row = 0; row != row1; ++row)
        getline(fileStream, line);
    // read file and put values into Mat
    float num;
    for (int row = 0; getline(fileStream, line) && row != matrix.rows; ++row)
    {
        iss.clear();
        iss.str(line);
        for (int col = 0; col != matrix.cols && (iss >> num); ++col)
            matrix.at<float>(row, col) = num;
    }
    
    if(fileStream.bad() || iss.bad())
    {
        std::cerr << "evg::dlmread(): error reading the file." << std::endl;
        throw exception();
    }
    
    return matrix;
}


bool dlmreadBool (const std::string& dlmfilePath, Mat matrix, int row1)
{
    try {
        matrix = dlmread (dlmfilePath, matrix, row1);
        return 1;
    } catch(...) {
        cerr << "evg::dlmread(): failed." << endl;
        return 0;
    }
}


// delimiter is currently space only
void dlmwrite (const std::string& dlmfilePath, const Mat& _matrix)
{
    // check the directory path for output video
    path p = absolute(path(dlmfilePath));
    if (! exists(p.parent_path()) )
    {
        cerr << "evg::dlmwrite(): Directory path " << p.parent_path()
             << " does not exist." << endl;
        throw exception();
    }

    // open file
    std::ofstream ofs (p.string().c_str());
    if (!ofs.good())
    {
        cerr << "evg::dlmwrite(): File " << p << " failed to open." << endl;
        throw exception();
    }
    
    // write stuff
    Mat matrix = _matrix;
    if (matrix.type() != CV_32F)
        matrix.convertTo(matrix, CV_32F);
    for (int i = 0; i != matrix.rows; ++i)
        for (int j = 0; j != matrix.cols; ++j)
            ofs << matrix.at<float>(i, j) << (j == matrix.cols-1 ? '\n' : ' ');
    ofs << flush;
}


bool dlmwriteBool (const std::string& dlmfilePath, const Mat& matrix)
{
    try {
        dlmwrite(dlmfilePath, matrix);
        return 1;
    } catch(...) {
        cerr << "evg::dlmwriteBool(): failed." << endl;
        return 0;
    }
}


// from http://stackoverflow.com/questions/3190378/opencv-store-to-database
void saveMat ( const string& filename, const Mat& M)
{
    try {
        // only one channel
        assert (M.channels() == 1);
    
        // check the parent path for output video
        path p = absolute(path(filename));
        if (! exists(p.parent_path()) )
        {
            cerr << "evg::saveMat(): Directory path " << p.parent_path()
                 << " does not exist" << endl;
            throw exception();
        }
        if (M.empty())
        {
            cerr << "evg::saveMat(): matrix is empty" << endl;
            throw exception();
        }
        std::ofstream out(p.string().c_str(), ios::out|ios::binary);
        if (!out)
        {
            cerr << "evg::saveMat(): cannot open file for writing" << endl;
            throw exception();
        }
        int cols = M.cols;
        int rows = M.rows;
        int chan = M.channels();
        int eSiz = int((M.dataend-M.datastart)/(cols*rows*chan));
        
        // Write header
        out.write((char*)&cols,sizeof(cols));
        out.write((char*)&rows,sizeof(rows));
        out.write((char*)&chan,sizeof(chan));
        out.write((char*)&eSiz,sizeof(eSiz));
        
        // Write data.
        if (M.isContinuous())
            out.write((char *)M.data,cols*rows*chan*eSiz);
        else
        {
            cerr << "evg::saveMat(): matrix must be continious." << endl;
            throw exception();
        }
        out.close();
    } catch(...) {
        cerr << "evg::saveMat(): exception caught." << endl;
        throw exception();
    }
}


bool saveMatBool ( const string& filename, const Mat& M)
{
    try {
        saveMat (filename, M);
        return 1;
    } catch(...) {
        cerr << "evg::saveMatBool(): failed." << endl;
        return 0;
    }
}


// from http://stackoverflow.com/questions/3190378/opencv-store-to-database
Mat readMat( const string& filename)
{
    try {
        Mat M;
    
        // open file
        path p = absolute(path(filename));
        if (! exists(p) )
        {
            cerr << "evg::dlmread(): Path " << p << " does not exist." << endl;
            throw exception();
        }
        std::ifstream in (p.string().c_str(), ios::in|ios::binary);
        if (!in)
        {
            cerr << "evg::readMat(): cannot open file " << p << " for reading" << endl;
            throw exception();
        }
        int cols;
        int rows;
        int chan;
        int eSiz;
        
        // Read header
        in.read((char*)&cols,sizeof(cols));
        in.read((char*)&rows,sizeof(rows));
        in.read((char*)&chan,sizeof(chan));
        in.read((char*)&eSiz,sizeof(eSiz));
        
        // Determine type of the matrix
        int type = 0;
        switch (eSiz){
            case sizeof(char):
                type = CV_8UC(chan);
                break;
            case sizeof(float):
                type = CV_32FC(chan);
                break;
            case sizeof(double):
                type = CV_64FC(chan);
                break;
            default:
                cerr << "evg::readMat(): bad matrix header" << endl;
                throw exception();
        }
        
        // Alocate Matrix.
        M = Mat(rows,cols,type,Scalar(1));
        
        // Read data.
        assert(M.isContinuous());
        in.read((char *)M.data,cols*rows*chan*eSiz);

        in.close();
        return M;
    } catch(...) {
        cerr << "evg::readMat(): exception caught." << endl;
        throw exception();
    }
}


bool readMat( const string& filename, Mat& M )
{
    try {
        M = readMat(filename);
        return 1;
    } catch(...) { return 0; }
}



SrcVideo::SrcVideo (const Type _type, const std::string _videoPath)
  : type (_type),
    videoPath(_videoPath),
    CameraWidth(640),
    CameraHeight(480)
{
    if (type == FILE && videoPath == "")
        cerr << "warning: evg::SrcVideo::SrcVideo(): input is set as file, "
                "but file path is not specified." << endl;
}


SrcVideo::SrcVideo (const SrcVideo& other)
  : type (other.type),
    videoPath(other.videoPath),
    CameraWidth(other.CameraWidth),
    CameraHeight(other.CameraHeight)
{
}


SrcVideo SrcVideo::operator= (const SrcVideo& other)
{
    type = other.type;
    videoPath = other.videoPath;
    CameraWidth = other.CameraWidth;
    CameraHeight = other.CameraHeight;
    return *this;
}



bool SrcVideo::openResource(VideoCapture& video)
{
    try {
        if (video.isOpened()) return 1;
        
        if (type == evg::SrcVideo::CAMERA)
        {
            if (! video.open(0))
            {
                cerr << "evg::SrcVideo::openResource(): Camera failed to open." << endl;
                return 0;
            }
            video.set(CV_CAP_PROP_FRAME_WIDTH, CameraWidth);
            video.set(CV_CAP_PROP_FRAME_HEIGHT, CameraHeight);
            cout << "note: evg::SrcVideo::openResource(): width x height = "
                 << video.get(CV_CAP_PROP_FRAME_WIDTH) << " x "
                 << video.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;
            cout << "note: evg::SrcVideo::openResource(): Camera successfully opened."
                 << endl;
        }
        else if (type == evg::SrcVideo::FILE)
        {
            if (! evg::openVideo(videoPath, video)) return 0;
        }
        else
        {
            cout << "evg::SrcVideo::openResource(): bad input type." << endl;
            return 0;
        }
        return 1;
    } catch (...) {
        cerr << "evg::SrcVideo::openResource(): excepton caught." << endl;
        return 0;
    }
}

bool SrcVideo::closeResource(VideoCapture& video)
{
    try {
        if (! video.isOpened()) return 1;
        video.release();
        return 1;
    } catch(...) {
        cerr << "evg::SrcVideo::closeResource(): failed." << endl;
        return 0;
    }
}


} // namespace evg
} // namespace cv

