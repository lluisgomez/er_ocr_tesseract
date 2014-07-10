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
	            vector<Rect>* component_rects, 
              vector<string>* component_texts, 
              vector<float>* component_confidences,
              int component_level)
{

  out_sequence.clear();
  component_rects->clear();
  component_texts->clear();
  component_confidences->clear();

  // First we split a line into words (TODO this must be optional)
  vector<Mat> words_mask;
  vector<Mat> words_src;
  vector<Rect> words_rect;

  /// Find contours
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  Mat tmp;
  mask.getMat().copyTo(tmp);
  findContours( tmp, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0) );
  if (contours.size() < 6)
  {
    //do not split lines with less than 6 characters
    words_mask.push_back(mask.getMat());
    words_src.push_back(src.getMat());
    words_rect.push_back(Rect(0,0,mask.getMat().cols,mask.getMat().rows));
  }
  else
  {


        Mat_<float> vector_w((int)mask.getMat().cols,1);
        reduce(mask.getMat(), vector_w, 0, CV_REDUCE_SUM, -1);

        vector<int> spaces;
        vector<int> spaces_start;
        vector<int> spaces_end;
        int space_count=0;
        int last_one_idx;
        for (int s=0; s<vector_w.cols; s++)
        {
            if (vector_w.at<float>(0,s) == 0)
            {
                space_count++;
            } else {
                if (space_count!=0)
                {
                    spaces.push_back(space_count);
                    spaces_start.push_back(last_one_idx);
                    spaces_end.push_back(s-1);
                }
                space_count = 0;
                last_one_idx = s;
            }
        }
        Scalar mean_space,std_space;
        meanStdDev(Mat(spaces),mean_space,std_space);
        int num_word_spaces = 0;
        int last_word_space_end = 0;
        for (int s=0; s<spaces.size(); s++)
        {
            if (spaces_end.at(s)-spaces_start.at(s) > mean_space[0]+(mean_space[0]*1.1)) //TODO this 1.1 is a param
            {
                if (num_word_spaces == 0)
                {
                    //cout << " we have a word from  0  to " << spaces_start.at(s) << endl;
                    Mat word_mask, word_src;
                    Rect word_rect = Rect(0,0,spaces_start.at(s),mask.getMat().rows);
                    mask.getMat()(word_rect).copyTo(word_mask);
                    src.getMat()(word_rect).copyTo(word_src);

                    words_mask.push_back(word_mask);
                    words_src.push_back(word_src);
                    words_rect.push_back(word_rect);
                }
                else
                {
                    //cout << " we have a word from " << last_word_space_end << " to " << spaces_start.at(s) << endl;
                    Mat word_mask, word_src;
                    Rect word_rect = Rect(last_word_space_end,0,spaces_start.at(s)-last_word_space_end,mask.getMat().rows);
                    mask.getMat()(word_rect).copyTo(word_mask);
                    src.getMat()(word_rect).copyTo(word_src);

                    words_mask.push_back(word_mask);
                    words_src.push_back(word_src);
                    words_rect.push_back(word_rect);
                }
                num_word_spaces++;
                last_word_space_end = spaces_end.at(s);
            }
        }
        //cout << " we have a word from " << last_word_space_end << " to " << vector_w.cols << endl << endl << endl;
                    Mat word_mask, word_src;
                    Rect word_rect = Rect(last_word_space_end,0,vector_w.cols-last_word_space_end,mask.getMat().rows);
                    mask.getMat()(word_rect).copyTo(word_mask);
                    src.getMat()(word_rect).copyTo(word_src);

                    words_mask.push_back(word_mask);
                    words_src.push_back(word_src);
                    words_rect.push_back(word_rect);

  }

  for (int w=0; w<words_mask.size(); w++)
  {

  vector< vector<int> > observations;
  vector< vector<double> > confidences;
  vector<int> obs;
  // First find contours and sort by x coordinate of bbox
  Mat tmp;
  words_mask[w].copyTo(tmp);
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
    words_src[w](contours_rect.at(i)).copyTo(tmp_src);
    words_mask[w](contours_rect.at(i)).copyTo(tmp_mask);

    vector<int> out_class;
    vector<double> out_conf;
    classifier->eval(tmp_src,tmp_mask,out_class,out_conf);
    if (!out_class.empty())
      obs.push_back(out_class[0]);
    observations.push_back(out_class);
    confidences.push_back(out_conf);
  }


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

   //cout << path[best_idx] << endl;
   out_sequence = out_sequence+" "+path[best_idx];
	 component_rects->push_back(words_rect[w]);
   component_texts->push_back(path[best_idx]);
   component_confidences->push_back(max_prob);

  }
   
  return 0;


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

  if (contours.empty())
    return;

  int idx = 0;
  if (contours.size() > 1)
  {
    // this is to make sure we have the mask with a single contour
    // e.g "i" and "j" have two contours, but it may be also a part of a neighbour character
    // we take the larger one and clean the outside in order to have a single contour
    int max_area = 0;
    for (int cc=0; cc<contours.size(); cc++)
    {
      int area_c = boundingRect(contours[cc]).area();
      if ( area_c > max_area)
      {
        idx = cc;
        max_area = area_c;
      }
    }

    // clean-up the outside of the contour 
    Mat tmp_c = Mat::zeros(tmp.rows, tmp.cols, CV_8UC1);
    drawContours(tmp_c, contours, idx, Scalar(255), CV_FILLED);
    img = img & tmp_c;
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
    GaussianBlur(maps[i], maps[i], Size(7,7), 2, 2);
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

  //printf("\n The char sample may be one of: ");
  for (int j=0; j<predictions.cols; j++)
  {
      //cout << ascii[j] << "(" << predictions.at<double>(0,j) << ") ";
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


  //printf("\n !! The char sample is predicted as: %s \n\n", ascii[out_class[0]]);

  //cout << sample << endl;
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

  if (contours.empty())
    return;

  int idx = 0;
  if (contours.size() > 1)
  {
    // this is to make sure we have the mask with a single contour
    // e.g "i" and "j" have two contours, but it may be also a part of a neighbour character
    // we take the larger one and clean the outside in order to have a single contour
    int max_area = 0;
    for (int cc=0; cc<contours.size(); cc++)
    {
      int area_c = boundingRect(contours[cc]).area();
      if ( area_c > max_area)
      {
        idx = cc;
        max_area = area_c;
      }
    }

    // clean-up the outside of the contour 
    Mat tmp_c = Mat::zeros(tmp.rows, tmp.cols, CV_8UC1);
    drawContours(tmp_c, contours, idx, Scalar(255), CV_FILLED);
    img = img & tmp_c;
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
    GaussianBlur(maps[i], maps[i], Size(7,7), 2, 2);
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
  vector<vector<int> > equivalency_mat(62);
  equivalency_mat[2].push_back(28);  // c -> C
  equivalency_mat[28].push_back(2);  // C -> c
  equivalency_mat[8].push_back(34);  // i -> I
  equivalency_mat[8].push_back(11);  // i -> l
  equivalency_mat[11].push_back(8);  // l -> i
  equivalency_mat[11].push_back(34); // l -> I
  equivalency_mat[34].push_back(8);  // I -> i
  equivalency_mat[34].push_back(11); // I -> l
  equivalency_mat[9].push_back(35);  // j -> J
  equivalency_mat[35].push_back(9);  // J -> j
  equivalency_mat[14].push_back(40); // o -> O
  equivalency_mat[14].push_back(52); // o -> 0
  equivalency_mat[40].push_back(14); // O -> o
  equivalency_mat[40].push_back(52); // O -> 0
  equivalency_mat[52].push_back(14); // 0 -> o
  equivalency_mat[52].push_back(40); // 0 -> O
  equivalency_mat[15].push_back(41); // p -> P
  equivalency_mat[41].push_back(15); // P -> p
  equivalency_mat[18].push_back(44); // s -> S
  equivalency_mat[44].push_back(18); // S -> s
  equivalency_mat[20].push_back(46); // u -> U
  equivalency_mat[46].push_back(20); // U -> u
  equivalency_mat[21].push_back(47); // v -> V
  equivalency_mat[47].push_back(21); // V -> v
  equivalency_mat[22].push_back(48); // w -> W
  equivalency_mat[48].push_back(22); // W -> w
  equivalency_mat[23].push_back(49); // x -> X
  equivalency_mat[49].push_back(23); // X -> x
  equivalency_mat[25].push_back(51); // z -> Z
  equivalency_mat[51].push_back(25); // Z -> z

  
  //printf("\n K nearest responses: ");
  for (int j=0; j<responses.cols; j++)
  {
      if (responses.at<float>(0,j)<0)
        continue;
      //cout << ascii[(int)responses.at<float>(0,j)] << "(" << dists.at<float>(0,j) << ")  ";
      class_predictions.at<double>(0,(int)responses.at<float>(0,j)) += dists.at<float>(0,j);
      for (int e=0; e<equivalency_mat[(int)responses.at<float>(0,j)].size(); e++)
      {
        //cout << ascii[equivalency_mat[(int)responses.at<float>(0,j)][e]] << "(" << dists.at<float>(0,j) << ")  ";
        class_predictions.at<double>(0,equivalency_mat[(int)responses.at<float>(0,j)][e]) += dists.at<float>(0,j);
        dist_sum[0] +=  dists.at<float>(0,j);
      }
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
  
  //printf("\n !! The char sample is predicted as: %s \n\n", ascii[(int)predictions.at<float>(0,0)]);


  //cout << sample << endl;
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

}


Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierKNN(const std::string& filename)

{
      return makePtr<OCRHMMClassifierKNN>(filename);
}
