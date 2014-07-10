#include "ocr_hmm_decoder.h"

//Default constructor
OCRHMMDecoder::OCRHMMDecoder( Ptr<OCRHMMDecoder::ClassifierCallback> _classifier,
                   string& _vocabulary,
                   InputArray transition_probabilities_table,
                   InputArray emission_probabilities_table,
                   decoder_mode _mode)
{
  classifier = _classifier;
  transition_p = transition_probabilities_table.getMat();
  emission_p = emission_probabilities_table.getMat();
  vocabulary = _vocabulary;
  mode = _mode;
}

OCRHMMDecoder::~OCRHMMDecoder()
{
}

bool sort_rect_horiz (Rect a,Rect b) { return (a.x<b.x); }

double OCRHMMDecoder::run( InputArray src,
              InputArray mask,
              string& out_sequence,
              component_level level)
{

  vector< vector<int> > observations;
  vector< vector<double> > confidences;
  vector<int> obs;
  // First find contours and sort by x coordinate of bbox
  Mat tmp;
  mask.getMat().copyTo(tmp);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  /// Find contours
  findContours( tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  vector<Rect> contours_rect;
  for (int i=0; i<contours.size(); i++)
  {
    contours_rect.push_back(boundingRect(contours[i]));
  }

  sort(contours_rect.begin(), contours_rect.end(), sort_rect_horiz);
  
  // Do character recognition foreach contour 
  for (int i=0; i<contours.size(); i++)
  {
    Mat tmp_src;
    Mat tmp_mask;
    src.getMat()(contours_rect.at(i)).copyTo(tmp_src);
    mask.getMat()(contours_rect.at(i)).copyTo(tmp_mask);
    vector<int> out_class;
    vector<double> out_conf;
    classifier->eval(tmp_src,tmp_mask,out_class,out_conf);
    obs.push_back(out_class[0]);
    observations.push_back(out_class);
    confidences.push_back(out_conf);
  }
  /*obs.push_back(7);
  vector<int> tmp;
  tmp.push_back(7);
  observations.push_back(tmp);
  vector<double> tmp2;
  tmp2.push_back(1.);
  confidences.push_back(tmp2);

  obs.push_back(4);
  tmp[0] = 4;
  observations.push_back(tmp);
  confidences.push_back(tmp2);

  obs.push_back(11);
  tmp[0] = 11;
  observations.push_back(tmp);
  confidences.push_back(tmp2);

  obs.push_back(7);
  tmp[0] = 7;
  tmp.push_back(11);
  observations.push_back(tmp);
  tmp2[0] = 0.7;
  tmp2.push_back(0.3);
  confidences.push_back(tmp2);

  obs.push_back(14);
  tmp.clear();
  tmp.push_back(14);
  observations.push_back(tmp);
  tmp2.clear();
  tmp2.push_back(1.);
  confidences.push_back(tmp2);*/






  //This must be extracted from dictionary, or just assumed to be equal for all characters
  vector<double> start_p(vocabulary.size());
  for (int i=0; i<vocabulary.size(); i++)
    start_p[i] = 1.0/vocabulary.size();


  Mat V = Mat::zeros(observations.size(),vocabulary.size(),CV_64FC1);
  vector<string> path(vocabulary.size());
    
  // Initialize base cases (t == 0)
  for (int i=0; i<vocabulary.size(); i++)
  {
    for (int j=0; j<observations[0].size(); j++)
    {
      emission_p.at<double>(observations[0][j],obs[0]) = confidences[0][j];
    }
    V.at<double>(0,i) = start_p[i] * emission_p.at<double>(i,obs[0]);
    path[i] = vocabulary.at(i);
  }

    
  // Run Viterbi for t > 0
  for (int t=1; t<obs.size(); t++)
  {

    //Dude this has to be done each time!!
    Mat emission_p = Mat::eye(62,62,CV_64FC1);
    for (int e=0; e<observations[t].size(); e++)
    {
      emission_p.at<double>(observations[t][e],obs[t]) = confidences[t][e];
    }

    vector<string> newpath(vocabulary.size());

    for (int i=0; i<vocabulary.size(); i++)
    {
      double max_prob = 0;
      int best_idx = 0; 
      for (int j=0; j<vocabulary.size(); j++)
      {
        double prob = V.at<double>(t-1,j) * transition_p.at<double>(j,i) * emission_p.at<double>(i,obs[t]);
        if ( prob > max_prob)
        {
          max_prob = prob;
          best_idx = j;
        }
      }

      V.at<double>(t,i) = max_prob;
      newpath[i] = path[best_idx] + vocabulary.at(i);
    }
 
    // Don't need to remember the old paths
    path.swap(newpath);
  }
 
   double max_prob = 0;
   int best_idx = 0; 
   for (int i=0; i<vocabulary.size(); i++)
   {
        double prob = V.at<double>(obs.size()-1,i);
        if ( prob > max_prob)
        {
          max_prob = prob;
          best_idx = i;
        }
   }

   cout << path[best_idx] << endl;
   out_sequence = path[best_idx];
   return max_prob;


}

class CV_EXPORTS OCRHMMClassifierMLP : public OCRHMMDecoder::ClassifierCallback
{
  public:
    //constructor
    OCRHMMClassifierMLP(const std::string& filename);
    // Destructor
    ~OCRHMMClassifierMLP() {}

    void eval( InputArray src, InputArray mask, vector<int>& out_class, vector<double>& out_confidence );
  private:
    CvANN_MLP mlp;
};

OCRHMMClassifierMLP::OCRHMMClassifierMLP (const string& filename)
{
  if (ifstream(filename.c_str()))
    mlp.load( filename.c_str(), "mlp" );
  else
    CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

void OCRHMMClassifierMLP::eval( InputArray _src, InputArray _mask, vector<int>& out_class, vector<double>& out_confidence )
{

  out_class.clear();
  out_confidence.clear();

  int image_height = 35;
  int image_width = 35;
  int num_features = 200;

  Mat img = _mask.getMat();
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
    cout << "Error: inconsistent number of contours" << endl;
    return;
  }
  Rect bbox = boundingRect(contours[idx]);

  //Crop to fit the exact rect of the contour and resize to a fixed-sized matrix of 35 x 35 pixel, while retaining the centroid of the region and aspect ratio.
  Mat mask = Mat::zeros(image_height,image_width,CV_8UC1);
  img(bbox).copyTo(tmp);


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
  Mat sample = Mat(1,num_features,CV_64FC1);
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
        sample.at<double>(0,i*25+((int)x/7)+((int)y/7)*5) = mean[0]/255;
        //cout << " avg " << mean[0] << " in patch " << x << "," << y << " channel " << i << " idx = " << i*25+((int)x/7)+((int)y/7)*5<< endl;
      }
    }
  }

  Mat predictions;
  mlp.predict( sample, predictions);


  static const char* ascii[62] = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"};
  double minVal; 
  double maxVal; 

  minMaxLoc( predictions, &minVal, &maxVal);
  predictions = (predictions - minVal) / (maxVal-minVal);

  printf("\n The char sample may be one of: ");
  for (int j=0; j<predictions.cols; j++)
  {
      cout << ascii[j] << "(" << predictions.at<double>(0,j) << ") ";
      if (predictions.at<double>(0,j) > 0.99)
      {
        out_class.insert(out_class.begin(),j);
        out_confidence.insert(out_confidence.begin(),predictions.at<double>(0,j));
      }
      else 
      {
        out_class.push_back(j);
        out_confidence.push_back(predictions.at<double>(0,j));
      }
  }


  printf("\n !! The char sample is predicted as: %s \n\n", ascii[out_class[0]]);

  cout << sample << endl;
  imshow("1",img);
  imshow("2",mask);
  Mat all_maps = Mat::zeros(image_height,image_width*maps.size(),CV_8UC1);
  for (int i=0; i<maps.size(); i++)
  {
    maps[i].copyTo(all_maps(Rect(i*maps[0].cols,0,maps[0].cols,maps[0].rows)));
  }
  imshow("3",all_maps);
  imwrite("out.jpg",mask);
  waitKey(0);

}


Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierMLP(const std::string& filename)

{
      return makePtr<OCRHMMClassifierMLP>(filename);
}


class CV_EXPORTS OCRHMMClassifierKNN : public OCRHMMDecoder::ClassifierCallback
{
  public:
    //constructor
    OCRHMMClassifierKNN(const std::string& filename);
    // Destructor
    ~OCRHMMClassifierKNN() {}

    void eval( InputArray src, InputArray mask, vector<int>& out_class, vector<double>& out_confidence );
  private:
    CvKNearest knn;
};

OCRHMMClassifierKNN::OCRHMMClassifierKNN (const string& filename)
{
  if (ifstream(filename.c_str()))
  {
    Mat hus, labels;
    cv::FileStorage storage(filename.c_str(), cv::FileStorage::READ);
    storage["hus"] >> hus;
    storage["labels"] >> labels;
    storage.release();
	  knn.train(hus, labels, Mat(), false, 32);
  }
  else
    CV_Error(CV_StsBadArg, "Default classifier data file not found!");
}

void OCRHMMClassifierKNN::eval( InputArray _src, InputArray _mask, vector<int>& out_class, vector<double>& out_confidence )
{

  out_class.clear();
  out_confidence.clear();

  int image_height = 35;
  int image_width = 35;
  int num_features = 200;

  Mat img = _mask.getMat();
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
    cout << "Error: inconsistent number of contours" << endl;
    return;
  }
  Rect bbox = boundingRect(contours[idx]);

  //Crop to fit the exact rect of the contour and resize to a fixed-sized matrix of 35 x 35 pixel, while retaining the centroid of the region and aspect ratio.
  Mat mask = Mat::zeros(image_height,image_width,CV_8UC1);
  img(bbox).copyTo(tmp);


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
  Mat sample = Mat(1,num_features,CV_32FC1);
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
        sample.at<float>(0,i*25+((int)x/7)+((int)y/7)*5) = mean[0]/255;
        //cout << " avg " << mean[0] << " in patch " << x << "," << y << " channel " << i << " idx = " << i*25+((int)x/7)+((int)y/7)*5<< endl;
      }
    }
  }

  Mat responses,dists,predictions;
  knn.find_nearest( sample, 11, &predictions, 0, &responses, &dists);
  
  Scalar dist_sum = sum(dists);
  Mat class_predictions = Mat::zeros(1,62,CV_64FC1);

  static const char* ascii[62] = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"};
  
  printf("\n K nearest responses: ");
  for (int j=0; j<responses.cols; j++)
  {
      cout << ascii[(int)responses.at<float>(0,j)] << "(" << dists.at<float>(0,j) << ")  ";
      class_predictions.at<double>(0,(int)responses.at<float>(0,j)) += dists.at<float>(0,j);
  }
  
  class_predictions = class_predictions/dist_sum[0];

  out_class.push_back((int)predictions.at<float>(0,0));
  out_confidence.push_back(class_predictions.at<double>(0,(int)predictions.at<float>(0,0)));

  for (int i=0; i<class_predictions.cols; i++)
  {
    if ((class_predictions.at<double>(0,i) > 0) && (i != out_class[0]))
    {
      out_class.push_back(i);
      out_confidence.push_back(class_predictions.at<double>(0,i));
    }
  }
  
  printf("\n !! The char sample is predicted as: %s \n\n", ascii[(int)predictions.at<float>(0,0)]);


  //cout << sample << endl;
  imshow("1",img);
  imshow("2",mask);
  Mat all_maps = Mat::zeros(image_height,image_width*maps.size(),CV_8UC1);
  for (int i=0; i<maps.size(); i++)
  {
    maps[i].copyTo(all_maps(Rect(i*maps[0].cols,0,maps[0].cols,maps[0].rows)));
  }
  imshow("3",all_maps);
  imwrite("out.jpg",mask);
  waitKey(0);

}


Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierKNN(const std::string& filename)

{
      return makePtr<OCRHMMClassifierKNN>(filename);
}
