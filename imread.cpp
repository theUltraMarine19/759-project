#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/core.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/imgcodecs.hpp"
#include "/home/arijit/installation/OpenCV-3.4.4/include/opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    Mat image;
    image = imread("Swimming-club.jpg",  IMREAD_COLOR ); // Read the file
    if(image.empty())                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    double *img = image.ptr<double>(0);
    cout << image.rows << endl;
    return 0;
}
