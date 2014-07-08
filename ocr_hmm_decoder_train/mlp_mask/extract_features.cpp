/**
 * @function moments_demo.cpp
 * @brief Demo code to calculate moments
 * @author OpenCV team
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;


/**
 * @function main
 */
int main( int, char** argv )
{
  Mat src; Mat src_gray;
  /// Load source image and convert it to gray
  src = imread( argv[1], 1 );

  /// Convert image to gray
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  /// Find contours
  findContours( src_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  int idx = 0;
  if (contours.size()==2)
  {
    // "i" and "j" have two contours, take the larger one
    Rect bbox0 = boundingRect(contours[0]);
    Rect bbox1 = boundingRect(contours[1]);
    if (bbox1.area() > bbox0.area())
      idx = 1;
  }
  else if (contours.size()>2)
  {
    fprintf(stderr,"Error: inconsistent number of contours\n");
    return(-1);
  }
  Rect bbox = boundingRect(contours[idx]);
  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  Mat tmp;
  //Crop to fit the exact rect of the contour and resize to 16x16
  src_gray(bbox).copyTo(tmp);
  resize(tmp,tmp,Size(16,16));
  cout << argv[2];
  for (int i=0; i<16; i++)
  {
    for (int j=0; j<16; j++)
      cout << " " << ((int)tmp.at<uchar>(i,j)==0)?0:1;
    //cout << endl;
  }

  return(0);
}

