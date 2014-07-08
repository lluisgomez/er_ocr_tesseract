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

class CV_EXPORTS OCRHMMClassifier : public OCRHMMDecoder::ClassifierCallback
{
  public:
    //constructor
    OCRHMMClassifier(const std::string& filename);
    // Destructor
    ~OCRHMMClassifier() {}

    void eval( InputArray src, InputArray mask, vector<int>& out_class, vector<double>& out_confidence );
  private:
    CvANN_MLP mlp;
};

OCRHMMClassifier::OCRHMMClassifier (const string& filename)
{
  if (ifstream(filename.c_str()))
    mlp.load( filename.c_str(), "mlp" );
  else
    CV_Error(CV_StsBadArg, "Default classifier file not found!");
}

void OCRHMMClassifier::eval( InputArray src, InputArray mask, vector<int>& out_class, vector<double>& out_confidence )
{

  out_class.clear();
  out_confidence.clear();

  Mat tmp;
  mask.getMat().copyTo(tmp);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  /// Find contours
  findContours( tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  if (contours.size()>2)
  {
    fprintf(stderr,"Error: inconsistent number of contours\n");
    return;
  }
  Rect bbox = boundingRect(contours[0]);
  //Crop to fit the exact rect of the contour and resize to 16x16
  mask.getMat()(bbox).copyTo(tmp);
  resize(tmp,tmp,Size(16,16));

  int num_features = 256;
  Mat sample = Mat(1,num_features,CV_64FC1);
  for (int i=0; i<16; i++)
  {
    for (int j=0; j<16; j++)
    {
      cout << " " << ((int)tmp.at<uchar>(i,j)==0)?0:1;
      sample.at<double>(0,i*16+j) = ((int)tmp.at<uchar>(i,j)==0)?1:0;
    }
    cout << endl;
  }
  cout << Mat(sample) << endl;

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

}


Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifier(const std::string& filename)

{
      return makePtr<OCRHMMClassifier>(filename);
}
