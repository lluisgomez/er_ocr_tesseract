#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main (int argc, char* argv[])
{

  int image_width = 35;
  int image_height = 35;

    Mat img = imread(argv[1]);
    if(img.channels() != 3)
      return(0);
    cvtColor(img,img,COLOR_RGB2GRAY);

    Mat tmp;
    img.copyTo(tmp);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    /// Find contours
    findContours( tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
      //cout << "Error: inconsistent number of contours " << contours.size() << endl;
      return(0);
    }
    Rect bbox = boundingRect(contours[idx]);

    //Crop to fit the exact rect of the contour and resize to a fixed-sized matrix of 35 x 35 pixel, while retaining the centroid of the region and aspect ratio.
    Mat mask = Mat::zeros(image_height,image_width,CV_8UC1);
    img(bbox).copyTo(tmp);


    /*
    //This tries to do non-liear mapping but doesn't work for the moment
    Moments mu = moments( tmp, true );
    Point centroid;
    centroid.x = mu.m10 / mu.m00;
    centroid.y = mu.m01 / mu.m00;
    cout << " centroid " << centroid << endl;

    Mat map_x, map_y;
    map_x.create( tmp.size(), CV_32FC1 );
    map_y.create( tmp.size(), CV_32FC1 );
    for( int j = 0; j < tmp.rows; j++ )
    { 
      for( int i = 0; i < tmp.cols; i++ )
        {
          map_x.at<float>(j,i) = 2*(i-centroid.x)+tmp.cols/2;
          map_y.at<float>(j,i) = 2*(j-centroid.y)+tmp.rows/2;
        }
    }

    imshow("befo",tmp);
    remap( tmp, tmp, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0) );
    imshow("afte",tmp);
    waitKey(0);*/


    if (tmp.cols>tmp.rows)
    {
      int height = image_width*tmp.rows/tmp.cols;
      resize(tmp,tmp,Size(image_width,height));
      tmp.copyTo(mask(Rect(0,(image_height-height)/2,image_width,height)));
    }
    else
    {
      int width = image_height*tmp.cols/tmp.rows;
      resize(tmp,tmp,Size(width,image_height));
      tmp.copyTo(mask(Rect((image_width-width)/2,0,width,image_height)));
    }

    //find contours again (now resized)
    mask.copyTo(tmp);
    findContours( tmp, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(0, 0) );

    vector<Mat> maps;
    for (int i=0; i<8; i++)
    {
      Mat map = Mat::zeros(image_height,image_width,CV_8UC1);
      maps.push_back(map);
    }
    for (int c=0; c<contours.size(); c++)
      for (int i=0; i<contours[c].size(); i++)
      {
        //cout << contours[c][i] << " -- " << contours[c][(i+1)%contours[c].size()] << endl;
        double dy = contours[c][i].y - contours[c][(i+1)%contours[c].size()].y;
        double dx = contours[c][i].x - contours[c][(i+1)%contours[c].size()].x;
        double angle = atan2 (dy,dx) * 180 / 3.14159265;
        //cout << " angle = " << angle << endl;
        int idx = 0;
        if ((angle>=157.5)||(angle<=-157.5))
          idx = 0;
        else if ((angle>=-157.5)&&(angle<=-112.5))
          idx = 1;
        else if ((angle>=-112.5)&&(angle<=-67.5))
          idx = 2;
        else if ((angle>=-67.5)&&(angle<=-22.5))
          idx = 3;
        else if ((angle>=-22.5)&&(angle<=22.5))
          idx = 4;
        else if ((angle>=22.5)&&(angle<=67.5))
          idx = 5;
        else if ((angle>=67.5)&&(angle<=112.5))
          idx = 6;
        else if ((angle>=112.5)&&(angle<=157.5))
          idx = 7;

        line(maps[idx],contours[c][i],contours[c][(i+1)%contours[c].size()],Scalar(255));
      }

    //On each bitmap a regular 7x7 Gaussian masks are evenly placed
    for (int i=0; i<maps.size(); i++)
    {
      copyMakeBorder(maps[i],maps[i],7,7,7,7,BORDER_CONSTANT,Scalar(0));
      GaussianBlur(maps[i], maps[i], Size(7,7), 7, 7);
      normalize(maps[i],maps[i],0,255,NORM_MINMAX);
      resize(maps[i],maps[i],Size(image_width,image_height));
    }

    //Generate features for each bitmap
    vector<double> feature_vector(200,0);
    Mat patch;
    for (int i=0; i<maps.size(); i++)
    {
      for(int y=0; y<image_height; y=y+7)
      {
        for(int x=0; x<image_width; x=x+7)
        {
          maps[i](Rect(x,y,7,7)).copyTo(patch);
          Scalar mean,std;
          meanStdDev(patch,mean,std);
          feature_vector[i*25+((int)x/7)+((int)y/7)*5] = mean[0];
          //cout << " avg " << mean[0] << " in patch " << x << "," << y << " channel " << i << " idx = " << i*25+((int)x/7)+((int)y/7)*5<< endl;
        }
      }
    }

    //cout << Mat(feature_vector) << endl;
    /*imshow("1",img);
    imshow("2",mask);
    Mat all_maps = Mat::zeros(image_height,image_width*maps.size(),CV_8UC1);
    for (int i=0; i<maps.size(); i++)
    {
      maps[i].copyTo(all_maps(Rect(i*maps[0].cols,0,maps[0].cols,maps[0].rows)));
    }
    imshow("3",all_maps);
    imwrite("out.jpg",mask);
    waitKey(0);*/

  cout << argv[2];
  for (int i=0; i<200; i++)
  {
      cout << " " << feature_vector[i];
  }

  return(0);

}
