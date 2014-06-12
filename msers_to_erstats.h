#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void MSERsToERStats(InputArray image, vector<vector<Point> > &contours, vector<vector<ERStat> > &mser_regions)
{

  CV_Assert(!contours.empty());
  Mat grey = image.getMat();
  // assert correct image type
  CV_Assert( grey.type() == CV_8UC1 );
  if (!mser_regions.empty())
    mser_regions.clear();

  //MSER output contains both MSER+ and MSER- regions in a single vector bu we want them separated
  mser_regions.resize(2);

  //Append "fake" root region to simulate a tree structure (needed for grouping)
  ERStat fake_root;
  mser_regions[0].push_back(fake_root);
  mser_regions[1].push_back(fake_root);

  Mat mask = Mat::zeros(grey.rows, grey.cols, CV_8UC1);
  Mat mtmp = Mat::zeros(grey.rows, grey.cols, CV_8UC1);
  for (int i=0; i<contours.size(); i++)
  {

    ERStat cser;
    cser.area = contours[i].size();
    cser.rect = boundingRect(contours[i]);

    float avg_intensity = 0;
    const vector<Point>& r = contours[i];
    for ( int j = 0; j < (int)r.size(); j++ )
    {
      Point pt = r[j];
      mask.at<unsigned char>(pt) = 255;
      avg_intensity += (float)grey.at<unsigned char>(pt)/(int)r.size();
    }

    double min, max;
    Point min_loc, max_loc;
    minMaxLoc(grey(cser.rect), &min, &max, &min_loc, &max_loc, mask(cser.rect));

    Mat element = getStructuringElement( MORPH_RECT, Size(5,5), Point(2,2) );
    dilate( mask(cser.rect), mtmp(cser.rect), element );
    absdiff( mtmp(cser.rect), mask(cser.rect), mtmp(cser.rect) );

    Scalar mean,std;
    meanStdDev(grey(cser.rect), mean, std, mtmp(cser.rect) );

    if (avg_intensity < mean[0])
    {
      cser.level  = (int)max;
      cser.pixel  = (max_loc.y+cser.rect.y)*grey.cols+max_loc.x+cser.rect.x;
      cser.parent = &(mser_regions[0][0]);
      mser_regions[0].push_back(cser);
    }
    else
    {
      cser.level  = 255-(int)min;
      cser.pixel  = (min_loc.y+cser.rect.y)*grey.cols+min_loc.x+cser.rect.x;
      cser.parent = &(mser_regions[1][0]);
      mser_regions[1].push_back(cser);
    }

    mask(cser.rect) = 0;
    mtmp(cser.rect) = 0;
  }
}
