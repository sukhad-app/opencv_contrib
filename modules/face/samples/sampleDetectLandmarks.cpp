#include "opencv2/face.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>
using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier face_cascade;
bool myDetector( InputArray image, OutputArray ROIs );

bool myDetector( InputArray image, OutputArray ROIs ){
    Mat gray;
    std::vector<Rect> faces;

    if(image.channels()>1){
        cvtColor(image.getMat(),gray,COLOR_BGR2GRAY);
    }else{
        gray = image.getMat().clone();
    }
    equalizeHist( gray, gray );

    face_cascade.detectMultiScale( gray, faces, 1.05, 3,0, Size(30, 30) );
    Mat(faces).copyTo(ROIs);
    return true;
}

int main(int argc,char** argv){
    //Give the path to the directory containing all the files containing data
    CommandLineParser parser(argc, argv,
        "{ help h usage ?    |      | give the following arguments in following format }"
        "{ model_filename f  |      | (required) path to binary file storing the trained model which is to be loaded [example - /data/file.dat]}"
        "{ image i           |      | (required) path to image in which face landmarks have to be detected.[example - /data/image.jpg] }"
        "{ face_cascade c    |      | Path to the face cascade xml file which you want to use as a detector}"
    );
    // Read in the input arguments
    if (parser.has("help")){
        parser.printMessage();
        cerr << "TIP: Use absolute paths to avoid any problems with the software!" << endl;
        return 0;
    }
    string filename(parser.get<string>("model_filename"));
    if (filename.empty()){
        parser.printMessage();
        cerr << "The name  of  the model file to be loaded for detecting landmarks is not found" << endl;
        return -1;
    }
    string image(parser.get<string>("image"));
    if (image.empty()){
        parser.printMessage();
        cerr << "The name  of  the image file in which landmarks have to be detected is not found" << endl;
        return -1;
    }
    string cascade_name(parser.get<string>("face_cascade"));
    if (cascade_name.empty()){
        parser.printMessage();
        cerr << "The name of the cascade classifier to be loaded to detect faces is not found" << endl;
        return -1;
    }

    //pass the face cascade xml file which you want to pass as a detector
    Mat img = imread(image);
    face_cascade.load(cascade_name);
    FacemarkKazemi::Params params;
    Ptr<Facemark> facemark = FacemarkKazemi::create(params);
    facemark->setFaceDetector(myDetector);
    facemark->loadModel(filename);
    cout<<"Loaded model"<<endl;
    vector<Rect> faces;
    resize(img,img,Size(460,460));
    facemark->getFaces(img,faces);
    vector< vector<Point2f> > shapes;
    if(facemark->fit(img,faces,shapes))
    {
        for( size_t i = 0; i < faces.size(); i++ )
        {
            cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
        }
        for(unsigned long i=0;i<faces.size();i++){
            for(unsigned long k=0;k<shapes[i].size();k++)
                cv::circle(img,shapes[i][k],5,cv::Scalar(0,0,255),FILLED);
        }
        namedWindow("Detected_shape");
        imshow("Detected_shape",img);
        waitKey(0);
    }
    return 0;
}