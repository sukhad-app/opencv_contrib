/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2000-2015, Intel Corporation, all rights reserved.
Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
Copyright (C) 2009-2015, NVIDIA Corporation, all rights reserved.
Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
Copyright (C) 2015, OpenCV Foundation, all rights reserved.
Copyright (C) 2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


/*
This file contains implementation of the bio-inspired features (BIF) approach
for computing image descriptors, applicable for human age estimation. For more
details we refer to [1,2].

REFERENCES
  [1] Guo, Guodong, et al. "Human age estimation using bio-inspired features."
      Computer Vision and Pattern Recognition, 2009. CVPR 2009.
  [2] Spizhevoi, A. S., and A. V. Bovyrin. "Estimating human age using
      bio-inspired features and the ranking method." Pattern Recognition and
      Image Analysis 25.3 (2015): 547-552.
*/
#include "opencv2/face/face_alignment.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/core.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;
namespace cv{ 
namespace face{

const std::string face_cascade_name = "lbpcascade_frontalface_improved.xml";
const std::string meanshapefile ="/home/sukhad/Desktop/opencv_contrib/modules/face/data/mean_shape.txt";

class ShapePredictor : public cv::face::FaceAlignment{
public: 
    virtual int getNumLandmarks() const {return numlandmarks_;}

    virtual int getNumFaces() const {return numfaces_;}

    virtual vector<cv::Point2f> getInitialShape() const { return initialshape_ ;}


    virtual std::vector<Point2f> setLandmarks(
        cv::Mat img,
        const std::vector<cv::Rect> face,
        const std::vector< std::vector<cv::Point2f> > landmarks
    ) const;


    ShapePredictor(){
        ifstream f(meanshapefile);
        string s;
        while(getline(f,s)){
            int i=0;
            char x1[100],y1[100];
            while(s[i]!=','){
                x1[i]=s[i];
                i++;
            }
            string x,y;
            x=string(x1);
            i++;
            int j=0;
            while(i<s.length()){
                y1[j++]=s[i];
                i++;
            }
            y=string(y1);
            Point2f pt;
            pt.x=std::stof(x);
            pt.y=std::stof(y);
            initialshape_.push_back(pt);
        }
        landmarks.clear();
        faces.clear();
    }
    ~ShapePredictor(){}
private:
    int numlandmarks_;
    int numfaces_;
    std::vector<cv::Point2f>  initialshape_;
    std::vector<cv::Rect> faces;
    std::vector<vector<cv::Point2f> > landmarks;
    //@param stores landmarks corresponding to a each image 
    
     
    
    //@param stores names of the training images
    

    /*Returns all the bounding rectangles enclosing all the faces in an image*/
    std::vector<cv::Rect> getBoundingRect(cv::Mat src);
    
    /*Reads the landmarks from the txt file.
      Each file's first line should give the path of the image whose 
      landmarks are being described in the file.Then in the subsequent 
      lines there should be coordinates of the landmarks in the image
      i.e each line should be of the form "x,y"(ignoring the double quotes)
      where x represents thee x coordinate of the landmark and y represents 
      the y coordinate of thee landmark.*/
    void getData(std::vector<string> filename,std::map<string,std::vector<cv::Point2f> > &_trainlandmarks
                ,std::vector<string> &_trainimages);
    

    /*This function gets tthe relative shape of the face scaled and centred 
      according to the bounding rectangleof the detected face*/
    void getMeanShapeRelative(cv::Rect face);
    /*@param faces - stores the bounding boxes of the face whose initial shape has
            to be found out*/

    /*This function calculates mean shape while training.
    Only called when new training data is supplied by the train function.*/
    void calcMeanShape(std::map<string,std::vector<cv::Point2f> > &_trainlandmarks,std::vector<string> &_trainimages);

    /*This functions scales the annotations according to a common size which is considered for all images*/
    void scaleData(std::map<string,std::vector<cv::Point2f> > & _trainlandmarks,
                                std::vector<string> & _trainimages ,Size s=Size(640,480) );
};

std::vector<cv::Rect> ShapePredictor :: getBoundingRect(cv::Mat src){
    std::vector<cv::Rect> faces;
    cv::CascadeClassifier face_cascade;  
  
    if(!face_cascade.load(face_cascade_name)){
        cout<<"Error"<<"\n";
        return faces;
    }
  
    Mat frame_gray;

    cvtColor( src, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    face_cascade.detectMultiScale( frame_gray, faces, 1.05, 3, 0, Size(30,30) );

    return faces;

}

void ShapePredictor :: getData(std::vector<string> filename,std::map<string,std::vector<cv::Point2f> > 
                              &_trainlandmarks,std::vector<string> &_trainimages)
{     
    for(int j=0;j<filename.size();j++){
        ifstream f(filename[j]);
        if(!f.is_open()){
            CV_Error(Error::StsError, "File can't be opened for reading!");
        }
        string img;
        //get the path of the image whose landmarks have to be detected
        getline(f,img);
        //push the image paths in the vector
       _trainimages.push_back(img);
        string s;
        while(getline(f,s)){
            int i=0;
            char x1[100],y1[100];
            while(s[i]!=','){
                x1[i]=s[i];
                i++;
            }
            string x,y;
            x=string(x1);
            i++;
            int j=0;
            while(i<s.length()){
                y1[j++]=s[i];
                i++;
            }
            y=string(y1);
            cv::Point2f pt;
            pt.x=std::stof(x);
            pt.y=std::stof(y);
           _trainlandmarks[img].push_back(pt);

        }
        f.close();
    }
}

void ShapePredictor :: getMeanShapeRelative(cv::Rect face){
    Point2f srcTri[3];
    Point2f dstTri[3];
    dstTri[0] = Point2f(face.x , face.y );
    dstTri[1] = Point2f( face.x + face.width, face.y );
    dstTri[2] = Point2f( face.x, face.y+face.height);
    float minx=8000.0,maxx=0.0,miny=8000.0,maxy=0.0;
    for(int i=0;i<initialshape_.size();i++){
        Point2f pt1;
        pt1.x=initialshape_[i].x;
        if(pt1.x<minx)
            minx=pt1.x;
        if(pt1.x>maxx)
            maxx=pt1.x;

        pt1.y=initialshape_[i].y;
        if(pt1.y<miny)
            miny=pt1.y;
        if(pt1.y>maxx)
            maxy=pt1.y;
    }
    srcTri[0] = Point2f(minx , miny );
    srcTri[1] = Point2f( maxx, miny );
    srcTri[2] = Point2f( minx, maxy );
    Mat warp_mat( 2, 3, CV_32FC1 );
    warp_mat = getAffineTransform( srcTri, dstTri );

    

    for(int i=0;i<initialshape_.size();i++){
        Point2f pt1=initialshape_[i];
        Mat C = (Mat_<double>(3,1) << pt1.x, pt1.y, 1);
        Mat D =warp_mat*C;   
        pt1.x=abs(D.at<double>(0,0));
        pt1.y=abs(D.at<double>(1,0));
        initialshape_[i]=pt1;
    }
}
  

void ShapePredictor :: calcMeanShape(std::map<string,std::vector<cv::Point2f> > &_trainlandmarks,std::vector<string> &_trainimages){

    if(_trainimages.empty()) {
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly by train function. Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    Point2f srcTri[3];
    Point2f dstTri[3];
    cv::Mat dst=cv::imread(_trainimages[0]);
    vector<Rect> faces1=getBoundingRect(dst);
    dstTri[0] = Point2f(faces1[0].x , faces1[0].y );
    dstTri[1] = Point2f( faces1[0].x + faces1[0].width, faces1[0].y );
    dstTri[2] = Point2f( faces1[0].x, faces1[0].y+faces1[0].height);
    
    long double xmean[200]={0.0};
    long double ymean[200]={0.0};
    int k;

    for(int i=0;i<_trainimages.size();i++){
        string img =_trainimages[i];
        cv::Mat src = cv::imread(img);
        vector<Rect> faces=getBoundingRect(src);
        
        srcTri[0] = Point2f( faces[0].x , faces[0].y );
        srcTri[1] = Point2f( faces[0].x+faces[0].width-1.f, faces[0].y );
        srcTri[2] = Point2f( faces[0].x, faces[0].y+faces[0].height-1.f );
    
        Mat warp_mat( 2, 3, CV_32FC1 );
        warp_mat = getAffineTransform( srcTri, dstTri );
    
        for(k=0;k<_trainlandmarks[img].size();k++){
           Point2f pt=_trainlandmarks[img][k];
           cv::Mat C = (Mat_<double>(3,1) << pt.x, pt.y, 1);
           cv::Mat D =warp_mat*C;   
           pt.x=abs(D.at<double>(0,0));
           pt.y=abs(D.at<double>(1,0));
           xmean[k]=xmean[k]+pt.x;
           ymean[k]=ymean[k]+pt.y;
        }

    }
    for(int i=0;i<k;i++){
        xmean[i]=xmean[i]/_trainimages.size();
        ymean[i]=ymean[i]/_trainimages.size();
        initialshape_[i]=Point2f(xmean[i],ymean[i]);
    }

}

void ShapePredictor :: scaleData( std::map<string,std::vector<cv::Point2f> > &_trainlandmarks,
                                std::vector<string> &_trainimages ,Size s)
{
    if(_trainimages.empty()) {
        // throw error if no data (or simply return -1?)
        String error_message = "The data is not loaded properly by train function. Aborting...";
        CV_Error(Error::StsBadArg, error_message);
        return ;
    }
    float scalex,scaley;

    for(int i=0;i<_trainimages.size();i++){
        string img =_trainimages[i];
        cv::Mat src = cv::imread(img);
        scalex=s.width/src.cols;
        scaley=s.height/src.rows;
        for(int k=0;k<_trainlandmarks[img].size();k++){
            Point2f pt=_trainlandmarks[img][k];
            pt.x=pt.x*scalex;
            pt.y=pt.y*scaley;
            _trainlandmarks[img][k]=pt;
        }

    }
}


}//face
}//cv


