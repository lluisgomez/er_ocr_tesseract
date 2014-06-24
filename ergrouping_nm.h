#include  <vector>
#include  <iostream>
#include  <iomanip>

using  namespace std;
using  namespace cv;

//these threshold values are learned from training dataset
#define PAIR_MIN_HEIGHT_RATIO     0.4
#define PAIR_MIN_CENTROID_ANGLE - 0.85
#define PAIR_MAX_CENTROID_ANGLE   0.85
#define PAIR_MIN_REGION_DIST    - 0.4 
#define PAIR_MAX_REGION_DIST      2.2
#define PAIR_MAX_INTENSITY_DIST   111
#define PAIR_MAX_AB_DIST          54

#define TRIPLET_MAX_DIST          0.9
#define TRIPLET_MAX_SLOPE         0.3

#define SEQUENCE_MAX_TRIPLET_DIST 0.45
#define SEQUENCE_MIN_LENGHT       4

// struct line_estimates
// Represents a line estimate (as above) for an ER's group
// i.e.: slope and intercept of 2 top and 2 bottom lines
struct line_estimates
{
    float top1_a0;
    float top1_a1;
    float top2_a0;
    float top2_a1;
    float bottom1_a0;
    float bottom1_a1;
    float bottom2_a0;
    float bottom2_a1;
    int x_min;
    int x_max;
    int h_max;
    bool operator==(const line_estimates& e) const
    {
        return ( (top1_a0 == e.top1_a0) && (top1_a1 == e.top1_a1) && (top2_a0 == e.top2_a0) &&
        (top2_a1 == e.top2_a1) && (bottom1_a0 == e.bottom1_a0) && (bottom1_a1 == e.bottom1_a1) &&
        (bottom2_a0 == e.bottom2_a0) && (bottom2_a1 == e.bottom2_a1) && (x_min == e.x_min) &&
        (x_max == e.x_max) && (h_max == e.h_max) );
    }
};

// distanceLinesEstimates
// Calculates the distance between two line estimates deﬁned as the largest
// normalized vertical difference of their top/bottom lines at their boundary points
// out float distance
float distanceLinesEstimates(line_estimates &a, line_estimates &b);

float distanceLinesEstimates(line_estimates &a, line_estimates &b)
{
    CV_Assert( (a.h_max != 0) && ( b.h_max != 0));

    if (a == b)
        return 0.0f;

    int x_min = min(a.x_min, b.x_min);
    int x_max = max(a.x_max, b.x_max);
    int h_max = max(a.h_max, b.h_max);

    float dist_top = INT_MAX, dist_bottom = INT_MAX;
    for (int i=0; i<2; i++)
    {
        float top_a0, top_a1, bottom_a0, bottom_a1;
        if (i == 0)
        {
            top_a0 = a.top1_a0;
            top_a1 = a.top1_a1;
            bottom_a0 = a.bottom1_a0;
            bottom_a1 = a.bottom1_a1;
        } else {
            top_a0 = a.top2_a0;
            top_a1 = a.top2_a1;
            bottom_a0 = a.bottom2_a0;
            bottom_a1 = a.bottom2_a1;
        }
        for (int j=0; j<2; j++)
        {
            float top_b0, top_b1, bottom_b0, bottom_b1;
            if (j==0)
            {
                top_b0 = b.top1_a0;
                top_b1 = b.top1_a1;
                bottom_b0 = b.bottom1_a0;
                bottom_b1 = b.bottom1_a1;
            } else {
                top_b0 = b.top2_a0;
                top_b1 = b.top2_a1;
                bottom_b0 = b.bottom2_a0;
                bottom_b1 = b.bottom2_a1;
            }

            float x_min_dist = abs((top_a0+x_min*top_a1) - (top_b0+x_min*top_b1));
            float x_max_dist = abs((top_a0+x_max*top_a1) - (top_b0+x_max*top_b1));
            dist_top    = min(dist_top, max(x_min_dist,x_max_dist)/h_max);

            x_min_dist  = abs((bottom_a0+x_min*bottom_a1) - (bottom_b0+x_min*bottom_b1));
            x_max_dist  = abs((bottom_a0+x_max*bottom_a1) - (bottom_b0+x_max*bottom_b1));
            dist_bottom = min(dist_bottom, max(x_min_dist,x_max_dist)/h_max);
        }
    }
    return max(dist_top, dist_bottom);
}

// struct region_pair
// Represents a pair of ER's
struct region_pair
{
    Vec2i a;
    Vec2i b;
    region_pair (Vec2i _a, Vec2i _b) : a(_a), b(_b) {}
    bool operator==(const region_pair& p1) const
    {
        return ( (p1.a == a) && (p1.b == b) );
    }
};

// struct region_triplet
// Represents a triplet of ER's
struct region_triplet
{
    Vec2i a;
    Vec2i b;
    Vec2i c;
    line_estimates estimates;
    region_triplet (Vec2i _a, Vec2i _b, Vec2i _c) : a(_a), b(_b), c(_c) {}
    bool operator==(const region_triplet& t1) const
    {
        return ( (t1.a == a) && (t1.b == b) && (t1.c == c) );
    }
};

// struct region_sequence
// Represents a sequence of more than three ER's
struct region_sequence
{
    vector<region_triplet> triplets;
    region_sequence (region_triplet t)
    {
        triplets.push_back(t);
    }
    region_sequence () {}
};

// Evaluates if a pair of regions is valid or not
// using thresholds learned on training (defined above)
bool isValidPair(Mat &grey, Mat& lab, Mat& mask, vector<Mat> &channels, std::vector< std::vector<ERStat> >& regions, cv::Vec2i idx1, cv::Vec2i idx2);

// Evaluates if a set of 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidTriplet(std::vector< std::vector<ERStat> >& regions, region_pair pair1, region_pair pair2, region_triplet &triplet);

// Evaluates if a set of more than 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidSequence(region_sequence &sequence1, region_sequence &sequence2);

// Check if two sequences share a region in common
bool haveCommonRegion(region_sequence &sequence1, region_sequence &sequence2);
// Check if two triplets share a region in common
bool haveCommonRegion(region_triplet &t1, region_triplet &t2);

// Takes as input the set of ER's extracted by ERFilter
// then finds for all valid pairs and triplets.
// in regions the set of ER's extracted by ERFilter
// in _src the channels from which the ER's were extracted
// out sets of regions, each one represents a possible text line
void erGroupingNM(cv::Mat &img, cv::InputArrayOfArrays _src, std::vector< std::vector<ERStat> >& regions,  std::vector< std::vector<Vec2i> >& groups, std::vector<Rect> &boxes, bool do_feedback_loop);

// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float &a0, float &a1);

// Fit line from three points using Ordinary Least Squares
// out a0 is the intercept
// out a1 is the slope
void fitLineOLS(Point p1, Point p2, Point p3, float &a0, float &a1);

// Fit line from three points using (heuristic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
// returns the error of the single point that doesn't fit the line
float fitLineLMS(Point p1, Point p2, Point p3, float &a0, float &a1);

// Fit a line_estimate to a group of 3 regions
// out triplet.estimates is updated with the new line estimates
bool fitLineEstimates(vector< vector<ERStat> > &regions, region_triplet &triplet);

// Fit line from two points
// out a0 is the intercept
// out a1 is the slope
void fitLine(Point p1, Point p2, float &a0, float &a1)
{
    CV_Assert ( p1.x != p2.x );

    a1 = (float)(p2.y - p1.y) / (p2.x - p1.x);
    a0 = a1 * -1 * p1.x + p1.y;
}

// Fit line from three points using Ordinary Least Squares
// out a0 is the intercept
// out a1 is the slope
void fitLineOLS(Point p1, Point p2, Point p3, float &a0, float &a1)
{
    float sumx  = p1.x + p2.x + p3.x;
    float sumy  = p1.y + p2.y + p3.y;
    float sumxy = p1.x*p1.y + p2.x*p2.y + p3.x*p3.y;
    float sumx2 = p1.x*p1.x + p2.x*p2.x + p3.x*p3.x;

    // line coefficients
    a0=(float)(sumy*sumx2-sumx*sumxy) / (3*sumx2-sumx*sumx);
    a1=(float)(3*sumxy-sumx*sumy) / (3*sumx2-sumx*sumx);
}

// Fit line from three points using (heutistic) Least-Median of Squares
// out a0 is the intercept
// out a1 is the slope
// returns the error of the single point that doesn't fit the line
float fitLineLMS(Point p1, Point p2, Point p3, float &a0, float &a1)
{
    //if this is not changed the line is not valid
    a0 = -1;
    a1 = 0;

    //Least-Median of Squares does not make sense with only three points
    //becuse any line passing by two of them has median_error = 0
    //So we'll take the one with smaller slope
    float l_a0, l_a1, best_slope=INT_MAX, err=0;

    if (p1.x != p2.x)
    {
        fitLine(p1,p2,l_a0,l_a1);;
        if (abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p3.y - (a0+a1*p3.x));
        }
    }


    if (p1.x != p3.x)
    {
        fitLine(p1,p3,l_a0,l_a1);
        if (abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p2.y - (a0+a1*p2.x));
        }
    }


    if (p2.x != p3.x)
    {
        fitLine(p2,p3,l_a0,l_a1);
        if (abs(l_a1) < best_slope)
        {
            best_slope = abs(l_a1);
            a0 = l_a0;
            a1 = l_a1;
            err = (p1.y - (a0+a1*p1.x));
        }
    }
    return err;

}

// Fit a line_estimate to a group of 3 regions
// out triplet.estimates is updated with the new line estimates
bool fitLineEstimates(vector< vector<ERStat> > &regions, region_triplet &triplet)
{
    vector<Rect> char_boxes;
    char_boxes.push_back(regions[triplet.a[0]][triplet.a[1]].rect);
    char_boxes.push_back(regions[triplet.b[0]][triplet.b[1]].rect);
    char_boxes.push_back(regions[triplet.c[0]][triplet.c[1]].rect);

    triplet.estimates.x_min = min(min(char_boxes[0].tl().x,char_boxes[1].tl().x), char_boxes[2].tl().x);
    triplet.estimates.x_max = max(max(char_boxes[0].br().x,char_boxes[1].br().x), char_boxes[2].br().x);
    triplet.estimates.h_max = max(max(char_boxes[0].height,char_boxes[1].height), char_boxes[2].height);

    // Fit one bottom line
    float err = fitLineLMS(char_boxes[0].br(), char_boxes[1].br(), char_boxes[2].br(),
                           triplet.estimates.bottom1_a0, triplet.estimates.bottom1_a1);

    if ((triplet.estimates.bottom1_a0 == -1) && (triplet.estimates.bottom1_a1 == 0))
        return false;

    // Slope for all lines must be the same
    triplet.estimates.bottom2_a1 = triplet.estimates.bottom1_a1;
    triplet.estimates.top1_a1    = triplet.estimates.bottom1_a1;
    triplet.estimates.top2_a1    = triplet.estimates.bottom1_a1;

    if (abs(err) > (float)triplet.estimates.h_max/6)
    {
        // We need two different bottom lines
        triplet.estimates.bottom2_a0 = triplet.estimates.bottom1_a0 + err;
    }
    else
    {
        // Second bottom line is the same
        triplet.estimates.bottom2_a0 = triplet.estimates.bottom1_a0;
    }

    // Fit one top line within the two (Y)-closer coordinates
    int d_12 = abs(char_boxes[0].tl().y - char_boxes[1].tl().y);
    int d_13 = abs(char_boxes[0].tl().y - char_boxes[2].tl().y);
    int d_23 = abs(char_boxes[1].tl().y - char_boxes[2].tl().y);
    if ((d_12<d_13) && (d_12<d_23))
    {
        Point p = Point((char_boxes[0].tl().x + char_boxes[1].tl().x)/2,
                        (char_boxes[0].tl().y + char_boxes[1].tl().y)/2);
        triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 +
                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
        p = char_boxes[2].tl();
        err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
    }
    else if (d_13<d_23)
    {
        Point p = Point((char_boxes[0].tl().x + char_boxes[2].tl().x)/2,
                        (char_boxes[0].tl().y + char_boxes[2].tl().y)/2);
        triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 +
                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
        p = char_boxes[1].tl();
        err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
    }
    else
    {
        Point p = Point((char_boxes[1].tl().x + char_boxes[2].tl().x)/2,
                        (char_boxes[1].tl().y + char_boxes[2].tl().y)/2);
        triplet.estimates.top1_a0 = triplet.estimates.bottom1_a0 +
                (p.y - (triplet.estimates.bottom1_a0+p.x*triplet.estimates.bottom1_a1));
        p = char_boxes[0].tl();
        err = (p.y - (triplet.estimates.top1_a0+p.x*triplet.estimates.top1_a1));
    }

    if (abs(err) > (float)triplet.estimates.h_max/6)
    {
        // We need two different top lines
        triplet.estimates.top2_a0 = triplet.estimates.top1_a0 + err;
    }
    else
    {
        // Second top line is the same
        triplet.estimates.top2_a0 = triplet.estimates.top1_a0;
    }

    return true;
}


// Evaluates if a pair of regions is valid or not
// using thresholds learned on training (defined above)
bool isValidPair(Mat &grey, Mat &lab, Mat &mask, vector<Mat> &channels, std::vector< std::vector<ERStat> >& regions, cv::Vec2i idx1, cv::Vec2i idx2)
{
    Rect minarearect  = regions[idx1[0]][idx1[1]].rect | regions[idx2[0]][idx2[1]].rect;

    // Overlapping regions are not valid pair in any case
    if ( (minarearect == regions[idx1[0]][idx1[1]].rect) ||
         (minarearect == regions[idx2[0]][idx2[1]].rect) )
        return false;

    ERStat *i, *j;
    if (regions[idx1[0]][idx1[1]].rect.x < regions[idx2[0]][idx2[1]].rect.x)
    {
        i = &regions[idx1[0]][idx1[1]];
        j = &regions[idx2[0]][idx2[1]];
    } else {
        i = &regions[idx2[0]][idx2[1]];
        j = &regions[idx1[0]][idx1[1]];
    }

    if (j->rect.x == i->rect.x)
        return false;
    
    float height_ratio = (float)min(i->rect.height,j->rect.height) /
                                max(i->rect.height,j->rect.height);

    Point center_i(i->rect.x+i->rect.width/2, i->rect.y+i->rect.height/2);
    Point center_j(j->rect.x+j->rect.width/2, j->rect.y+j->rect.height/2);
    float centroid_angle = atan2(center_j.y-center_i.y, center_j.x-center_i.x);

    int avg_width = (i->rect.width + j->rect.width) / 2;
    float norm_distance = (float)(j->rect.x-(i->rect.x+i->rect.width))/avg_width;

    if (( height_ratio   < PAIR_MIN_HEIGHT_RATIO) ||
        ( centroid_angle < PAIR_MIN_CENTROID_ANGLE) ||
        ( centroid_angle > PAIR_MAX_CENTROID_ANGLE) ||
        ( norm_distance  < PAIR_MIN_REGION_DIST) ||
        ( norm_distance  > PAIR_MAX_REGION_DIST))
        return false;

    if ((i->parent == NULL)||(j->parent == NULL)) // deprecate the root region
      return false;

    i = &regions[idx1[0]][idx1[1]];
    j = &regions[idx2[0]][idx2[1]];

    Mat region = mask(Rect(Point(i->rect.x,i->rect.y),
                           Point(i->rect.br().x+2,i->rect.br().y+2)));
    region = Scalar(0);

    int newMaskVal = 255;
    int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
    Rect rect;

    floodFill( channels[idx1[0]](Rect(Point(i->rect.x,i->rect.y),Point(i->rect.br().x,i->rect.br().y))),
               region, Point(i->pixel%grey.cols - i->rect.x, i->pixel/grey.cols - i->rect.y),
               Scalar(255), &rect, Scalar(i->level), Scalar(0), flags);
    rect.width += 2;
    rect.height += 2;
    Mat rect_mask = mask(Rect(i->rect.x+1,i->rect.y+1,i->rect.width,i->rect.height));

    Scalar mean,std;
    meanStdDev(grey(i->rect),mean,std,rect_mask);
    int grey_mean1 = mean[0];
    meanStdDev(lab(i->rect),mean,std,rect_mask);
    float a_mean1 = mean[1];
    float b_mean1 = mean[2];

    region = mask(Rect(Point(j->rect.x,j->rect.y),
                           Point(j->rect.br().x+2,j->rect.br().y+2)));
    region = Scalar(0);

    floodFill( channels[idx2[0]](Rect(Point(j->rect.x,j->rect.y),Point(j->rect.br().x,j->rect.br().y))),
               region, Point(j->pixel%grey.cols - j->rect.x, j->pixel/grey.cols - j->rect.y),
               Scalar(255), &rect, Scalar(j->level), Scalar(0), flags);
    rect.width += 2;
    rect.height += 2;
    rect_mask = mask(Rect(j->rect.x+1,j->rect.y+1,j->rect.width,j->rect.height));

    meanStdDev(grey(j->rect),mean,std,rect_mask);
    int grey_mean2 = mean[0];
    meanStdDev(lab(j->rect),mean,std,rect_mask);
    float a_mean2 = mean[1];
    float b_mean2 = mean[2];

    if (abs(grey_mean1-grey_mean2) > PAIR_MAX_INTENSITY_DIST)
      return false;

    if (sqrt(pow(a_mean1-a_mean2,2)+pow(b_mean1-b_mean2,2)) > PAIR_MAX_AB_DIST)
      return false;



    return true;
}

// Evaluates if a set of 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidTriplet(std::vector< std::vector<ERStat> >& regions, region_pair pair1, region_pair pair2, region_triplet &triplet)
{

    if (pair1 == pair2)
        return false;

    // At least one region in common is needed
    if ( (pair1.a == pair2.a)||(pair1.a == pair2.b)||(pair1.b == pair2.a)||(pair1.b == pair2.b) )
    {

        //fill the indexes in the output tripled (sorted)
        if (pair1.a == pair2.a)
        {
            if ((regions[pair1.b[0]][pair1.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
                    (regions[pair2.b[0]][pair2.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x))
                return false;
            if ((regions[pair1.b[0]][pair1.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
                    (regions[pair2.b[0]][pair2.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x))
                return false;

            triplet.a = (regions[pair1.b[0]][pair1.b[1]].rect.x <
                         regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.b : pair2.b;
            triplet.b = pair1.a;
            triplet.c = (regions[pair1.b[0]][pair1.b[1]].rect.x >
                         regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.b : pair2.b;

        } else if (pair1.a == pair2.b) {
            if ((regions[pair1.b[0]][pair1.b[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
                    (regions[pair2.a[0]][pair2.a[1]].rect.x <= regions[pair1.a[0]][pair1.a[1]].rect.x))
                return false;
            if ((regions[pair1.b[0]][pair1.b[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x) &&
                    (regions[pair2.a[0]][pair2.a[1]].rect.x >= regions[pair1.a[0]][pair1.a[1]].rect.x))
                return false;

            triplet.a = (regions[pair1.b[0]][pair1.b[1]].rect.x <
                         regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.b : pair2.a;
            triplet.b = pair1.a;
            triplet.c = (regions[pair1.b[0]][pair1.b[1]].rect.x >
                         regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.b : pair2.a;

        } else if (pair1.b == pair2.a) {
            if ((regions[pair1.a[0]][pair1.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
                    (regions[pair2.b[0]][pair2.b[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x))
                return false;
            if ((regions[pair1.a[0]][pair1.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
                    (regions[pair2.b[0]][pair2.b[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x))
                return false;

            triplet.a = (regions[pair1.a[0]][pair1.a[1]].rect.x <
                         regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.a : pair2.b;
            triplet.b = pair1.b;
            triplet.c = (regions[pair1.a[0]][pair1.a[1]].rect.x >
                         regions[pair2.b[0]][pair2.b[1]].rect.x)? pair1.a : pair2.b;

        } else if (pair1.b == pair2.b) {
            if ((regions[pair1.a[0]][pair1.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
                    (regions[pair2.a[0]][pair2.a[1]].rect.x <= regions[pair1.b[0]][pair1.b[1]].rect.x))
                return false;
            if ((regions[pair1.a[0]][pair1.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x) &&
                    (regions[pair2.a[0]][pair2.a[1]].rect.x >= regions[pair1.b[0]][pair1.b[1]].rect.x))
                return false;

            triplet.a = (regions[pair1.a[0]][pair1.a[1]].rect.x <
                         regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.a : pair2.a;
            triplet.b = pair1.b;
            triplet.c = (regions[pair1.a[0]][pair1.a[1]].rect.x >
                         regions[pair2.a[0]][pair2.a[1]].rect.x)? pair1.a : pair2.a;

        }



        if ( (regions[triplet.a[0]][triplet.a[1]].rect.x == regions[triplet.b[0]][triplet.b[1]].rect.x) &&
             (regions[triplet.a[0]][triplet.a[1]].rect.x == regions[triplet.c[0]][triplet.c[1]].rect.x) )
            return false;

        if ( (regions[triplet.a[0]][triplet.a[1]].rect.br().x == regions[triplet.b[0]][triplet.b[1]].rect.br().x) &&
             (regions[triplet.a[0]][triplet.a[1]].rect.br().x == regions[triplet.c[0]][triplet.c[1]].rect.br().x) )
            return false;


        if (!fitLineEstimates(regions, triplet))
            return false;

        if ( (triplet.estimates.bottom1_a0 < triplet.estimates.top1_a0) ||
             (triplet.estimates.bottom1_a0 < triplet.estimates.top2_a0) ||
             (triplet.estimates.bottom2_a0 < triplet.estimates.top1_a0) ||
             (triplet.estimates.bottom2_a0 < triplet.estimates.top2_a0) )
            return false;

        int central_height = min(triplet.estimates.bottom1_a0, triplet.estimates.bottom2_a0) -
                             max(triplet.estimates.top1_a0,triplet.estimates.top2_a0);
        int top_height     = abs(triplet.estimates.top1_a0 - triplet.estimates.top2_a0);
        int bottom_height  = abs(triplet.estimates.bottom1_a0 - triplet.estimates.bottom2_a0);

        if (central_height == 0)
            return false;

        float top_height_ratio    = (float)top_height/central_height;
        float bottom_height_ratio = (float)bottom_height/central_height;

        if ( (top_height_ratio > TRIPLET_MAX_DIST) || (bottom_height_ratio > TRIPLET_MAX_DIST) )
            return false;

        if (abs(triplet.estimates.bottom1_a1) > TRIPLET_MAX_SLOPE)
            return false;

        return true;
    }

    return false;
}

// Evaluates if a set of more than 3 regions is valid or not
// using thresholds learned on training (defined above)
bool isValidSequence(region_sequence &sequence1, region_sequence &sequence2)
{
    for (size_t i=0; i<sequence2.triplets.size(); i++)
    {
        for (size_t j=0; j<sequence1.triplets.size(); j++)
        {
            if (distanceLinesEstimates(sequence2.triplets[i].estimates,
                                       sequence1.triplets[j].estimates) > SEQUENCE_MAX_TRIPLET_DIST)
                return false;
        }
    }

    return true;
}

// Check if two triplets share a region in common
bool haveCommonRegion(region_triplet &t1, region_triplet &t2)
{
    if ((t1.a==t2.a) || (t1.a==t2.b) || (t1.a==t2.c) || 
        (t1.b==t2.a) || (t1.b==t2.b) || (t1.b==t2.c) || 
        (t1.c==t2.a) || (t1.c==t2.b) || (t1.c==t2.c)) 
      return true;

    return false;
}

// Check if two sequences share a region in common
bool haveCommonRegion(region_sequence &sequence1, region_sequence &sequence2)
{
    for (size_t i=0; i<sequence2.triplets.size(); i++)
    {
        for (size_t j=0; j<sequence1.triplets.size(); j++)
        {
            if (haveCommonRegion(sequence2.triplets[i], sequence1.triplets[j]))
                return true;
        }
    }

    return false;
}

bool sort_couples (Vec3i i,Vec3i j) { return (i[0]<j[0]); }


// Takes as input the set of ER's extracted by ERFilter
// then finds for all valid pairs and triplets.
// in regions the set of ER's extracted by ERFilter
// in _src the channels from which the ER's were extracted
// out sets of regions, each one represents a possible text line
void erGroupingNM(cv::Mat &img, cv::InputArrayOfArrays _src, std::vector< std::vector<ERStat> >& regions,
                  std::vector< std::vector<Vec2i> >& out_groups, std::vector<Rect>& out_boxes, bool do_feedback_loop)
{

    std::vector<Mat> src;
    _src.getMatVector(src);

    CV_Assert ( !src.empty() );
    CV_Assert ( src.size() == regions.size() );

    size_t num_channels = src.size();

    //process each channel independently
    for(size_t c=0; c<num_channels; c++)
    {
        //store indices to regions in a single vector
        std::vector< cv::Vec2i > all_regions;
        for(size_t r=0; r<regions[c].size(); r++)
        {
            all_regions.push_back(Vec2i(c,r));
        }

        std::vector< region_pair > valid_pairs;
        Mat mask = Mat::zeros(img.rows+2, img.cols+2, CV_8UC1);
        Mat grey,lab;
        cvtColor(img, lab, COLOR_RGB2Lab);
        cvtColor(img, grey, COLOR_RGB2GRAY);
    
        //check every possible pair of regions
        for (size_t i=0; i<all_regions.size(); i++)
        {
            vector<int> i_siblings;
            int first_i_sibling_idx = valid_pairs.size();
            for (size_t j=i+1; j<all_regions.size(); j++)
            {
                // check height ratio, centroid angle and region distance normalized by region width
                // fall within a given interval
                if (isValidPair(grey, lab, mask, src, regions, all_regions[i],all_regions[j]))
                {
                    bool isCycle = false;
                    for (size_t k=0; k<i_siblings.size(); k++)
                    {
                      if (isValidPair(grey, lab, mask, src, regions, all_regions[j],all_regions[i_siblings[k]]))
                      {
                        // choose as sibling the closer and not the first that was "paired" with i
                        Point i_center = Point( regions[all_regions[i][0]][all_regions[i][1]].rect.x +
                                                regions[all_regions[i][0]][all_regions[i][1]].rect.width/2,
                                                regions[all_regions[i][0]][all_regions[i][1]].rect.y +
                                                regions[all_regions[i][0]][all_regions[i][1]].rect.height/2 );
                        Point j_center = Point( regions[all_regions[j][0]][all_regions[j][1]].rect.x +
                                                regions[all_regions[j][0]][all_regions[j][1]].rect.width/2,
                                                regions[all_regions[j][0]][all_regions[j][1]].rect.y +
                                                regions[all_regions[j][0]][all_regions[j][1]].rect.height/2 );
                        Point k_center = Point( regions[all_regions[i_siblings[k]][0]][all_regions[i_siblings[k]][1]].rect.x +
                                                regions[all_regions[i_siblings[k]][0]][all_regions[i_siblings[k]][1]].rect.width/2,
                                                regions[all_regions[i_siblings[k]][0]][all_regions[i_siblings[k]][1]].rect.y +
                                                regions[all_regions[i_siblings[k]][0]][all_regions[i_siblings[k]][1]].rect.height/2 );
    
                        if ( norm(i_center - j_center) < norm(i_center - k_center) )
                        {
                          valid_pairs[first_i_sibling_idx+k] = region_pair(all_regions[i],all_regions[j]);
                          i_siblings[k] = j;
                        }
                        isCycle = true;
                        break;
                      }
                    }
                    if (!isCycle)
                    {
                      valid_pairs.push_back(region_pair(all_regions[i],all_regions[j]));
                      i_siblings.push_back(j);
                      //cout << "Valid pair (" << all_regions[i][0] << ","  << all_regions[i][1] << ") (" << all_regions[j][0] << ","  << all_regions[j][1] << ")" << endl;
                    }
                }
            }
        }
    
        //cout << "GroupingNM : detected " << valid_pairs.size() << " valid pairs" << endl;
    
        std::vector< region_triplet > valid_triplets;
    
        //check every possible triplet of regions
        for (size_t i=0; i<valid_pairs.size(); i++)
        {
            for (size_t j=i+1; j<valid_pairs.size(); j++)
            {
                // check colinearity rules
                region_triplet valid_triplet(Vec2i(0,0),Vec2i(0,0),Vec2i(0,0));
                if (isValidTriplet(regions, valid_pairs[i],valid_pairs[j], valid_triplet))
                {
                    valid_triplets.push_back(valid_triplet);
                    //cout << "Valid triplet (" << valid_triplet.a[1] << "," <<  valid_triplet.b[1] << "," <<  valid_triplet.c[1] << ")" << endl;
                }
            }
        }
    
        //cout << "GroupingNM : detected " << valid_triplets.size() << " valid triplets" << endl;
    
        vector<region_sequence> valid_sequences;
        vector<region_sequence> pending_sequences;
    
        for (size_t i=0; i<valid_triplets.size(); i++)
        {
            pending_sequences.push_back(region_sequence(valid_triplets[i]));
        }
    
    
        for (size_t i=0; i<pending_sequences.size(); i++)
        {
            bool expanded = false;
            for (size_t j=i+1; j<pending_sequences.size(); j++)
            {
                if (isValidSequence(pending_sequences[i], pending_sequences[j]))
                {
                    expanded = true;
                    pending_sequences[i].triplets.insert(pending_sequences[i].triplets.begin(), pending_sequences[j].triplets.begin(), pending_sequences[j].triplets.end());
                    pending_sequences.erase(pending_sequences.begin()+j);
                    j--;
                }
            }
            if (expanded)
            {
                valid_sequences.push_back(pending_sequences[i]);
            }
        }
    
        // remove a sequence if one its regions is already grouped within a longer seq
        for (size_t i=0; i<valid_sequences.size(); i++)
        {
            for (size_t j=i+1; j<valid_sequences.size(); j++)
            {
              if (haveCommonRegion(valid_sequences[i],valid_sequences[j]))
              {
                if (valid_sequences[i].triplets.size() < valid_sequences[j].triplets.size())
                {
                  valid_sequences.erase(valid_sequences.begin()+i);
                  i--;
                  break;
                }
                else
                {
                  valid_sequences.erase(valid_sequences.begin()+j);
                  j--;
                }
              }
            }
        }
    
    
        //cout << "GroupingNM : detected " << valid_sequences.size() << " sequences." << endl;

        if (do_feedback_loop)
        {

            //Feedback loop of detected lines to region extraction ... tries to recover missmatches in the region decomposition step by extracting regions in the neighbourhood of a valid sequence and checking if they are consistent with its line estimates
            Ptr<ERFilter> er_filter = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"),1,0.005,0.3,0.,true,0.1);
            for (int i=0; i<valid_sequences.size(); i++)
            {
                vector<Point> bbox_points;

                for (size_t j=0; j<valid_sequences[i].triplets.size(); j++)
                {
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.tl());
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.br());
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.tl());
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.br());
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.tl());
                    bbox_points.push_back(regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.br());
                }

                Rect rect = boundingRect(bbox_points);
                rect.x = max(rect.x,0);
                rect.y = max(rect.y,0);
                rect.width = min(rect.width,src[c].cols-rect.x);
                rect.height = min(rect.height,src[c].rows-rect.y);

                vector<ERStat> aux_regions;
                Mat tmp;
                src[c](rect).copyTo(tmp);
                //                    imshow("tmp",tmp);
                //                    waitKey(0);
                er_filter->run(tmp, aux_regions);
                //cout << aux_regions.size() << " possible regions detected" << endl;

                for(size_t r=0; r<aux_regions.size(); r++)
                {
                    aux_regions[r].rect   = aux_regions[r].rect + Point(rect.x,rect.y);
                    aux_regions[r].pixel  = ((aux_regions[r].pixel/tmp.cols)+rect.y)*src[c].cols + (aux_regions[r].pixel%tmp.cols) + rect.x;
                    bool overlaps = false;
                    for (size_t j=0; j<valid_sequences[i].triplets.size(); j++)
                    {
                        Rect minarearect_a  = regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect | aux_regions[r].rect;
                        Rect minarearect_b  = regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect | aux_regions[r].rect;
                        Rect minarearect_c  = regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect | aux_regions[r].rect;
                        //                    cout << "    (" << aux_regions[r].rect.x << ","  << aux_regions[r].rect.y << "," << aux_regions[r].rect.width << "," << aux_regions[r].rect.height <<  ")" << endl;
                        //                    cout << "    (" << regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.x << ","  << regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.y << "," << regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.width << "," << regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.height <<  ")" << endl;
                        //                    cout << "    (" << regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.x << ","  << regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.y << "," << regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.width << "," << regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.height <<  ")" << endl;
                        //                    cout << "    (" << regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.x << ","  << regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.y << "," << regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.width << "," << regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.height <<  ")" << endl << endl;

                        // Overlapping regions are not valid pair in any case
                        if ( (minarearect_a == aux_regions[r].rect) ||
                             (minarearect_b == aux_regions[r].rect) ||
                             (minarearect_c == aux_regions[r].rect) ||
                             (minarearect_a == regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect) ||
                             (minarearect_b == regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect) ||
                             (minarearect_c == regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect) )

                        {
                            //cout << "its overlap!" << endl;
                            overlaps = true;
                            break;
                        }
                    }
                    if (!overlaps)
                    {
                        //cout << "NO overlap" << endl;
                        //now check if it has at least one valid pair
                        vector<Vec3i> left_couples, right_couples;
                        regions[c].push_back(aux_regions[r]);
                        for (size_t j=0; j<valid_sequences[i].triplets.size(); j++)
                        {
                            if (isValidPair(grey, lab, mask, src, regions, valid_sequences[i].triplets[j].a, Vec2i(c,regions[c].size()-1)))
                            {
                                //cout << "has a pair !" << endl;
                                if (regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.x > aux_regions[r].rect.x)
                                    right_couples.push_back(Vec3i(regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.x - aux_regions[r].rect.x, valid_sequences[i].triplets[j].a[0],valid_sequences[i].triplets[j].a[1]));
                                else
                                    left_couples.push_back(Vec3i(aux_regions[r].rect.x - regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]].rect.x, valid_sequences[i].triplets[j].a[0],valid_sequences[i].triplets[j].a[1]));
                            }
                            if (isValidPair(grey, lab, mask, src, regions, valid_sequences[i].triplets[j].b, Vec2i(c,regions[c].size()-1)))
                            {
                                //cout << "has a pair !" << endl;
                                if (regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.x > aux_regions[r].rect.x)
                                    right_couples.push_back(Vec3i(regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.x - aux_regions[r].rect.x, valid_sequences[i].triplets[j].b[0],valid_sequences[i].triplets[j].b[1]));
                                else
                                    left_couples.push_back(Vec3i(aux_regions[r].rect.x - regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]].rect.x, valid_sequences[i].triplets[j].b[0],valid_sequences[i].triplets[j].b[1]));
                            }
                            if (isValidPair(grey, lab, mask, src, regions, valid_sequences[i].triplets[j].c, Vec2i(c,regions[c].size()-1)))
                            {
                                //cout << "has a pair !" << endl;
                                if (regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.x > aux_regions[r].rect.x)
                                    right_couples.push_back(Vec3i(regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.x - aux_regions[r].rect.x, valid_sequences[i].triplets[j].c[0],valid_sequences[i].triplets[j].c[1]));
                                else
                                    left_couples.push_back(Vec3i(aux_regions[r].rect.x - regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]].rect.x, valid_sequences[i].triplets[j].c[0],valid_sequences[i].triplets[j].c[1]));
                            }
                        }

                        //make it part of a triplet and check if line estimates is consistent with the sequence
                        vector<region_triplet> valid_triplets;
                        if(!left_couples.empty() && !right_couples.empty())
                        {
                            sort(left_couples.begin(), left_couples.end(), sort_couples);
                            sort(right_couples.begin(), right_couples.end(), sort_couples);
                            region_pair pair1(Vec2i(left_couples[0][1],left_couples[0][2]),Vec2i(c,regions[c].size()-1));
                            region_pair pair2(Vec2i(c,regions[c].size()-1), Vec2i(right_couples[0][1],right_couples[0][2]));
                            region_triplet triplet(Vec2i(0,0),Vec2i(0,0),Vec2i(0,0));
                            if (isValidTriplet(regions, pair1, pair2, triplet))
                            {
                                //cout << "Valid triplet here !!" << endl;
                                valid_triplets.push_back(triplet);
                            }
                        }
                        else if (right_couples.size() >= 2)
                        {
                            sort(right_couples.begin(), right_couples.end(), sort_couples);
                            region_pair pair1(Vec2i(c,regions[c].size()-1), Vec2i(right_couples[0][1],right_couples[0][2]));
                            region_pair pair2(Vec2i(right_couples[0][1],right_couples[0][2]), Vec2i(right_couples[1][1],right_couples[1][2]));
                            region_triplet triplet(Vec2i(0,0),Vec2i(0,0),Vec2i(0,0));
                            if (isValidTriplet(regions, pair1, pair2, triplet))
                            {
                                //cout << "Valid triplet here !!" << endl;
                                valid_triplets.push_back(triplet);
                            }
                        }
                        else if (left_couples.size() >=2)
                        {
                            sort(left_couples.begin(), left_couples.end(), sort_couples);
                            region_pair pair1(Vec2i(right_couples[1][1],right_couples[1][2]), Vec2i(right_couples[0][1],right_couples[0][2]));
                            region_pair pair2(Vec2i(right_couples[0][1],right_couples[0][2]),Vec2i(c,regions[c].size()-1));
                            region_triplet triplet(Vec2i(0,0),Vec2i(0,0),Vec2i(0,0));
                            if (isValidTriplet(regions, pair1, pair2, triplet))
                            {
                                //cout << "Valid triplet here !!" << endl;
                                valid_triplets.push_back(triplet);
                            }
                        }
                        else
                        {
                            //cout << "no possible triplet found !" << endl;
                            continue;
                        }

                        //check if line estimates is consistent with the sequence
                        for (size_t t=0; t<valid_triplets.size(); t++)
                        {
                            region_sequence sequence(valid_triplets[t]);
                            if (isValidSequence(valid_sequences[i],sequence))
                            {
                                //cout << "Valid sequence!" << endl;
                                valid_sequences[i].triplets.push_back(valid_triplets[t]);
                            }

                        }
                    }
                    //                    rectangle(img, aux_regions[r].rect.tl(), aux_regions[r].rect.br(), Scalar(255,0,0));
                    //                    imshow("candidate",img(rect));
                    //                    waitKey(0);
                }
            }

        }

    
        // Prepare the sequences for output
        for (size_t i=0; i<valid_sequences.size(); i++)
        {
            vector<Point> bbox_points;
            vector<Vec2i> group_regions;
    
            for (size_t j=0; j<valid_sequences[i].triplets.size(); j++)
            {
                size_t prev_size = group_regions.size();
                if(find(group_regions.begin(), group_regions.end(), valid_sequences[i].triplets[j].a) == group_regions.end())
                  group_regions.push_back(valid_sequences[i].triplets[j].a);
                if(find(group_regions.begin(), group_regions.end(), valid_sequences[i].triplets[j].b) == group_regions.end())
                  group_regions.push_back(valid_sequences[i].triplets[j].b);
                if(find(group_regions.begin(), group_regions.end(), valid_sequences[i].triplets[j].c) == group_regions.end())
                  group_regions.push_back(valid_sequences[i].triplets[j].c);
    
                for (size_t k=prev_size; k<group_regions.size(); k++)
                {
                    bbox_points.push_back(regions[group_regions[k][0]][group_regions[k][1]].rect.tl());
                    bbox_points.push_back(regions[group_regions[k][0]][group_regions[k][1]].rect.br());
                }
            }
    
            out_groups.push_back(group_regions);
            out_boxes.push_back(boundingRect(bbox_points));
            
        }
    }


    //TODO remove this, it is only to visualize for debug

    /*Mat lines = Mat::zeros(src[0].rows+2,src[0].cols+2,CV_8UC3);
    for (size_t c=0; c<regions.size(); c++)
    for (size_t i=1; i<regions[c].size(); i++)
    {
        rectangle(lines, regions[c][i].rect.tl(), regions[c][i].rect.br(), Scalar(255,0,0));
        char buff[5]; char *buff_ptr = buff;
        sprintf(buff, "%d", i);
        putText(lines,string(buff_ptr),regions[c][i].rect.tl(),FONT_HERSHEY_SIMPLEX,1,Scalar(255,0,0));
    }
    for (size_t i=0; i<valid_sequences.size(); i++)
    {
      lines = 0;
        cout << " Sequence with " << valid_sequences[i].triplets.size() << " triplets." << endl;
        for (size_t j=0; j<valid_sequences[i].triplets.size(); j++)
        {
          cout << " (" << valid_sequences[i].triplets[j].a[1] << "," << valid_sequences[i].triplets[j].b[1] << "," << valid_sequences[i].triplets[j].c[1] << ") " ; 


            ERStat *a,*b,*c;
            a = &regions[valid_sequences[i].triplets[j].a[0]][valid_sequences[i].triplets[j].a[1]];
            b = &regions[valid_sequences[i].triplets[j].b[0]][valid_sequences[i].triplets[j].b[1]];
            c = &regions[valid_sequences[i].triplets[j].c[0]][valid_sequences[i].triplets[j].c[1]];

            Point center_a(a->rect.x+a->rect.width/2, a->rect.y+a->rect.height/2);
            Point center_b(b->rect.x+b->rect.width/2, b->rect.y+b->rect.height/2);
            Point center_c(c->rect.x+c->rect.width/2, c->rect.y+c->rect.height/2);

            line(lines,center_a,center_b, Scalar(0,0,255),2);
            line(lines,center_b,center_c, Scalar(0,0,255),2);
        }
        cout << endl;

    imshow("lines",lines);
    waitKey(0);

    }*/

}
