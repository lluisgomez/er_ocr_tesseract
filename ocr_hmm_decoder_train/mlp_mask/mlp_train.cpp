#include <cstdlib>
#include "opencv/cv.h"
#include "opencv/ml.h"
#include <vector>
#include <fstream>

using namespace std;
using namespace cv;

static const char* ascii[62] = {"a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"};

int main(int argc, char** argv) {
/* STEP 1. Loading training data from file */
//2. Read the file
ifstream infile("data_train.txt");

int num_samples = 18600; //number of lines in the dataset file
int num_classes = 62; //number of classes
int num_features = 200; //number of features

Mat hus = Mat(num_samples,num_features,CV_64FC1);
Mat labels = Mat::ones(num_samples,num_classes,CV_64FC1);
labels = labels * -1;
int label;
int count = 0;

while (infile >> label >> hus.at<double>(count,0) >> hus.at<double>(count,1) >> hus.at<double>(count,2) >> hus.at<double>(count,3) >> hus.at<double>(count,4) >> hus.at<double>(count,5) >> hus.at<double>(count,6) >> hus.at<double>(count,7) >> hus.at<double>(count,8) >> hus.at<double>(count,9) >> hus.at<double>(count,10) >> hus.at<double>(count,11) >> hus.at<double>(count,12) >> hus.at<double>(count,13) >> hus.at<double>(count,14) >> hus.at<double>(count,15) >> hus.at<double>(count,16) >> hus.at<double>(count,17) >> hus.at<double>(count,18) >> hus.at<double>(count,19) >> hus.at<double>(count,20) >> hus.at<double>(count,21) >> hus.at<double>(count,22) >> hus.at<double>(count,23) >> hus.at<double>(count,24) >> hus.at<double>(count,25) >> hus.at<double>(count,26) >> hus.at<double>(count,27) >> hus.at<double>(count,28) >> hus.at<double>(count,29) >> hus.at<double>(count,30) >> hus.at<double>(count,31) >> hus.at<double>(count,32) >> hus.at<double>(count,33) >> hus.at<double>(count,34) >> hus.at<double>(count,35) >> hus.at<double>(count,36) >> hus.at<double>(count,37) >> hus.at<double>(count,38) >> hus.at<double>(count,39) >> hus.at<double>(count,40) >> hus.at<double>(count,41) >> hus.at<double>(count,42) >> hus.at<double>(count,43) >> hus.at<double>(count,44) >> hus.at<double>(count,45) >> hus.at<double>(count,46) >> hus.at<double>(count,47) >> hus.at<double>(count,48) >> hus.at<double>(count,49) >> hus.at<double>(count,50) >> hus.at<double>(count,51) >> hus.at<double>(count,52) >> hus.at<double>(count,53) >> hus.at<double>(count,54) >> hus.at<double>(count,55) >> hus.at<double>(count,56) >> hus.at<double>(count,57) >> hus.at<double>(count,58) >> hus.at<double>(count,59) >> hus.at<double>(count,60) >> hus.at<double>(count,61) >> hus.at<double>(count,62) >> hus.at<double>(count,63) >> hus.at<double>(count,64) >> hus.at<double>(count,65) >> hus.at<double>(count,66) >> hus.at<double>(count,67) >> hus.at<double>(count,68) >> hus.at<double>(count,69) >> hus.at<double>(count,70) >> hus.at<double>(count,71) >> hus.at<double>(count,72) >> hus.at<double>(count,73) >> hus.at<double>(count,74) >> hus.at<double>(count,75) >> hus.at<double>(count,76) >> hus.at<double>(count,77) >> hus.at<double>(count,78) >> hus.at<double>(count,79) >> hus.at<double>(count,80) >> hus.at<double>(count,81) >> hus.at<double>(count,82) >> hus.at<double>(count,83) >> hus.at<double>(count,84) >> hus.at<double>(count,85) >> hus.at<double>(count,86) >> hus.at<double>(count,87) >> hus.at<double>(count,88) >> hus.at<double>(count,89) >> hus.at<double>(count,90) >> hus.at<double>(count,91) >> hus.at<double>(count,92) >> hus.at<double>(count,93) >> hus.at<double>(count,94) >> hus.at<double>(count,95) >> hus.at<double>(count,96) >> hus.at<double>(count,97) >> hus.at<double>(count,98) >> hus.at<double>(count,99) >> hus.at<double>(count,100) >> hus.at<double>(count,101) >> hus.at<double>(count,102) >> hus.at<double>(count,103) >> hus.at<double>(count,104) >> hus.at<double>(count,105) >> hus.at<double>(count,106) >> hus.at<double>(count,107) >> hus.at<double>(count,108) >> hus.at<double>(count,109) >> hus.at<double>(count,110) >> hus.at<double>(count,111) >> hus.at<double>(count,112) >> hus.at<double>(count,113) >> hus.at<double>(count,114) >> hus.at<double>(count,115) >> hus.at<double>(count,116) >> hus.at<double>(count,117) >> hus.at<double>(count,118) >> hus.at<double>(count,119) >> hus.at<double>(count,120) >> hus.at<double>(count,121) >> hus.at<double>(count,122) >> hus.at<double>(count,123) >> hus.at<double>(count,124) >> hus.at<double>(count,125) >> hus.at<double>(count,126) >> hus.at<double>(count,127) >> hus.at<double>(count,128) >> hus.at<double>(count,129) >> hus.at<double>(count,130) >> hus.at<double>(count,131) >> hus.at<double>(count,132) >> hus.at<double>(count,133) >> hus.at<double>(count,134) >> hus.at<double>(count,135) >> hus.at<double>(count,136) >> hus.at<double>(count,137) >> hus.at<double>(count,138) >> hus.at<double>(count,139) >> hus.at<double>(count,140) >> hus.at<double>(count,141) >> hus.at<double>(count,142) >> hus.at<double>(count,143) >> hus.at<double>(count,144) >> hus.at<double>(count,145) >> hus.at<double>(count,146) >> hus.at<double>(count,147) >> hus.at<double>(count,148) >> hus.at<double>(count,149) >> hus.at<double>(count,150) >> hus.at<double>(count,151) >> hus.at<double>(count,152) >> hus.at<double>(count,153) >> hus.at<double>(count,154) >> hus.at<double>(count,155) >> hus.at<double>(count,156) >> hus.at<double>(count,157) >> hus.at<double>(count,158) >> hus.at<double>(count,159) >> hus.at<double>(count,160) >> hus.at<double>(count,161) >> hus.at<double>(count,162) >> hus.at<double>(count,163) >> hus.at<double>(count,164) >> hus.at<double>(count,165) >> hus.at<double>(count,166) >> hus.at<double>(count,167) >> hus.at<double>(count,168) >> hus.at<double>(count,169) >> hus.at<double>(count,170) >> hus.at<double>(count,171) >> hus.at<double>(count,172) >> hus.at<double>(count,173) >> hus.at<double>(count,174) >> hus.at<double>(count,175) >> hus.at<double>(count,176) >> hus.at<double>(count,177) >> hus.at<double>(count,178) >> hus.at<double>(count,179) >> hus.at<double>(count,180) >> hus.at<double>(count,181) >> hus.at<double>(count,182) >> hus.at<double>(count,183) >> hus.at<double>(count,184) >> hus.at<double>(count,185) >> hus.at<double>(count,186) >> hus.at<double>(count,187) >> hus.at<double>(count,188) >> hus.at<double>(count,189) >> hus.at<double>(count,190) >> hus.at<double>(count,191) >> hus.at<double>(count,192) >> hus.at<double>(count,193) >> hus.at<double>(count,194) >> hus.at<double>(count,195) >> hus.at<double>(count,196) >> hus.at<double>(count,197) >> hus.at<double>(count,198) >> hus.at<double>(count,199))
{
  //cout << hus.row(count) << endl;
  labels.at<double>(count,label) = 1.0;
  count++;
}

/* STEP 2. Create and train the Neural Network */
//1. Declare the classifier
CvANN_MLP mlp;


ifstream ifile("./trained_mlp.xml");
if (ifile) 
{
	// The file exists, so we don't need to train 
	printf("Loading network from file ... \n");
	mlp.load("./trained_mlp.xml", "mlp");
} else {
  //1. Create the MLP with 3 layers
  Mat layers = Mat(3, 1, CV_32SC1);
  layers.row(0) = Scalar(num_features);
  layers.row(1) = Scalar(100);
  layers.row(2) = Scalar(num_classes);
  mlp.create(layers,CvANN_MLP::SIGMOID_SYM,0.6,1);
	CvANN_MLP_TrainParams params;
	CvTermCriteria criteria;
	criteria.max_iter = 1000;
	criteria.epsilon = 0.000001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	//2. Train it 
	printf("Training ... \n");
	mlp.train(hus, labels, Mat(), Mat(), params, 0);
}

/* STEP 3. Calculating the testing and training error */
Mat predictions;
mlp.predict(hus,predictions);
int tp=0;
int fp=0;
for (int i=0; i<predictions.rows; i++)
{
  int maxLabelIdx = 0;
  double maxLabelVal = -1;
  int maxPredIdx = 0;
  double maxPredVal = -1;
  for (int j=0; j<predictions.cols; j++)
  {
    if (predictions.at<double>(i,j)>maxPredVal)
    {
      maxPredVal = predictions.at<double>(i,j);
      maxPredIdx = j;
    }
    if (labels.at<double>(i,j)>maxLabelVal)
    {
      maxLabelVal = labels.at<double>(i,j);
      maxLabelIdx = j;
    }
  }

  if (maxLabelIdx == maxPredIdx)
      tp++;
  else 
      fp++;
}
cout << "Train Accuracy = " << (float)tp/(tp+fp) << endl;


ifstream test_infile("data_test.txt");

hus = Mat(num_samples,num_features,CV_64FC1);
labels = Mat::ones(num_samples,num_classes,CV_64FC1);
labels = labels * -1;
count = 0;

while (test_infile >> label >> hus.at<double>(count,0) >> hus.at<double>(count,1) >> hus.at<double>(count,2) >> hus.at<double>(count,3) >> hus.at<double>(count,4) >> hus.at<double>(count,5) >> hus.at<double>(count,6) >> hus.at<double>(count,7) >> hus.at<double>(count,8) >> hus.at<double>(count,9) >> hus.at<double>(count,10) >> hus.at<double>(count,11) >> hus.at<double>(count,12) >> hus.at<double>(count,13) >> hus.at<double>(count,14) >> hus.at<double>(count,15) >> hus.at<double>(count,16) >> hus.at<double>(count,17) >> hus.at<double>(count,18) >> hus.at<double>(count,19) >> hus.at<double>(count,20) >> hus.at<double>(count,21) >> hus.at<double>(count,22) >> hus.at<double>(count,23) >> hus.at<double>(count,24) >> hus.at<double>(count,25) >> hus.at<double>(count,26) >> hus.at<double>(count,27) >> hus.at<double>(count,28) >> hus.at<double>(count,29) >> hus.at<double>(count,30) >> hus.at<double>(count,31) >> hus.at<double>(count,32) >> hus.at<double>(count,33) >> hus.at<double>(count,34) >> hus.at<double>(count,35) >> hus.at<double>(count,36) >> hus.at<double>(count,37) >> hus.at<double>(count,38) >> hus.at<double>(count,39) >> hus.at<double>(count,40) >> hus.at<double>(count,41) >> hus.at<double>(count,42) >> hus.at<double>(count,43) >> hus.at<double>(count,44) >> hus.at<double>(count,45) >> hus.at<double>(count,46) >> hus.at<double>(count,47) >> hus.at<double>(count,48) >> hus.at<double>(count,49) >> hus.at<double>(count,50) >> hus.at<double>(count,51) >> hus.at<double>(count,52) >> hus.at<double>(count,53) >> hus.at<double>(count,54) >> hus.at<double>(count,55) >> hus.at<double>(count,56) >> hus.at<double>(count,57) >> hus.at<double>(count,58) >> hus.at<double>(count,59) >> hus.at<double>(count,60) >> hus.at<double>(count,61) >> hus.at<double>(count,62) >> hus.at<double>(count,63) >> hus.at<double>(count,64) >> hus.at<double>(count,65) >> hus.at<double>(count,66) >> hus.at<double>(count,67) >> hus.at<double>(count,68) >> hus.at<double>(count,69) >> hus.at<double>(count,70) >> hus.at<double>(count,71) >> hus.at<double>(count,72) >> hus.at<double>(count,73) >> hus.at<double>(count,74) >> hus.at<double>(count,75) >> hus.at<double>(count,76) >> hus.at<double>(count,77) >> hus.at<double>(count,78) >> hus.at<double>(count,79) >> hus.at<double>(count,80) >> hus.at<double>(count,81) >> hus.at<double>(count,82) >> hus.at<double>(count,83) >> hus.at<double>(count,84) >> hus.at<double>(count,85) >> hus.at<double>(count,86) >> hus.at<double>(count,87) >> hus.at<double>(count,88) >> hus.at<double>(count,89) >> hus.at<double>(count,90) >> hus.at<double>(count,91) >> hus.at<double>(count,92) >> hus.at<double>(count,93) >> hus.at<double>(count,94) >> hus.at<double>(count,95) >> hus.at<double>(count,96) >> hus.at<double>(count,97) >> hus.at<double>(count,98) >> hus.at<double>(count,99) >> hus.at<double>(count,100) >> hus.at<double>(count,101) >> hus.at<double>(count,102) >> hus.at<double>(count,103) >> hus.at<double>(count,104) >> hus.at<double>(count,105) >> hus.at<double>(count,106) >> hus.at<double>(count,107) >> hus.at<double>(count,108) >> hus.at<double>(count,109) >> hus.at<double>(count,110) >> hus.at<double>(count,111) >> hus.at<double>(count,112) >> hus.at<double>(count,113) >> hus.at<double>(count,114) >> hus.at<double>(count,115) >> hus.at<double>(count,116) >> hus.at<double>(count,117) >> hus.at<double>(count,118) >> hus.at<double>(count,119) >> hus.at<double>(count,120) >> hus.at<double>(count,121) >> hus.at<double>(count,122) >> hus.at<double>(count,123) >> hus.at<double>(count,124) >> hus.at<double>(count,125) >> hus.at<double>(count,126) >> hus.at<double>(count,127) >> hus.at<double>(count,128) >> hus.at<double>(count,129) >> hus.at<double>(count,130) >> hus.at<double>(count,131) >> hus.at<double>(count,132) >> hus.at<double>(count,133) >> hus.at<double>(count,134) >> hus.at<double>(count,135) >> hus.at<double>(count,136) >> hus.at<double>(count,137) >> hus.at<double>(count,138) >> hus.at<double>(count,139) >> hus.at<double>(count,140) >> hus.at<double>(count,141) >> hus.at<double>(count,142) >> hus.at<double>(count,143) >> hus.at<double>(count,144) >> hus.at<double>(count,145) >> hus.at<double>(count,146) >> hus.at<double>(count,147) >> hus.at<double>(count,148) >> hus.at<double>(count,149) >> hus.at<double>(count,150) >> hus.at<double>(count,151) >> hus.at<double>(count,152) >> hus.at<double>(count,153) >> hus.at<double>(count,154) >> hus.at<double>(count,155) >> hus.at<double>(count,156) >> hus.at<double>(count,157) >> hus.at<double>(count,158) >> hus.at<double>(count,159) >> hus.at<double>(count,160) >> hus.at<double>(count,161) >> hus.at<double>(count,162) >> hus.at<double>(count,163) >> hus.at<double>(count,164) >> hus.at<double>(count,165) >> hus.at<double>(count,166) >> hus.at<double>(count,167) >> hus.at<double>(count,168) >> hus.at<double>(count,169) >> hus.at<double>(count,170) >> hus.at<double>(count,171) >> hus.at<double>(count,172) >> hus.at<double>(count,173) >> hus.at<double>(count,174) >> hus.at<double>(count,175) >> hus.at<double>(count,176) >> hus.at<double>(count,177) >> hus.at<double>(count,178) >> hus.at<double>(count,179) >> hus.at<double>(count,180) >> hus.at<double>(count,181) >> hus.at<double>(count,182) >> hus.at<double>(count,183) >> hus.at<double>(count,184) >> hus.at<double>(count,185) >> hus.at<double>(count,186) >> hus.at<double>(count,187) >> hus.at<double>(count,188) >> hus.at<double>(count,189) >> hus.at<double>(count,190) >> hus.at<double>(count,191) >> hus.at<double>(count,192) >> hus.at<double>(count,193) >> hus.at<double>(count,194) >> hus.at<double>(count,195) >> hus.at<double>(count,196) >> hus.at<double>(count,197) >> hus.at<double>(count,198) >> hus.at<double>(count,199))
{

  labels.at<double>(count,label) = 1.0;
  count++;
}

mlp.predict(hus,predictions);
tp=0;
fp=0;
for (int i=0; i<predictions.rows; i++)
{
  int maxLabelIdx = 0;
  double maxLabelVal = -1000;
  int maxPredIdx = 0;
  double maxPredVal = -1000;
  for (int j=0; j<predictions.cols; j++)
  {
    if (predictions.at<double>(i,j)>maxPredVal)
    {
      maxPredVal = predictions.at<double>(i,j);
      maxPredIdx = j;
    }
    if (labels.at<double>(i,j)>maxLabelVal)
    {
      maxLabelVal = labels.at<double>(i,j);
      maxLabelIdx = j;
    }
  }

  if (maxLabelIdx == maxPredIdx)
      tp++;
  else 
      fp++;
}
cout << "Test Accuracy = " << (float)tp/(tp+fp) << endl;

//Predict an individual sample
double t = cvGetTickCount();
//static const double arr[] = {1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1}; // D
//static const double arr[] = {1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1}; // S
//static const double arr[] = {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1}; //O
//static const double arr[] = {1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1}; //Y
//static const double arr[] = {1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}; // C
//static const double arr[] = {0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0}; // N
//static const double arr[] = {1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1}; // G
//static const double arr[] = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0}; // A
//static const double arr[] = {1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0}; // R
static const double arr[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}; //H

hus = Mat(1,num_features,CV_64FC1,(void*)&arr);
mlp.predict( hus, predictions);
int maxPredIdx = 0; float maxPredVal = -1000;
printf("\n The char sample may be one of: ");
for (int j=0; j<predictions.cols; j++)
{
  if (predictions.at<double>(0,j)>-0.3)
    cout << ascii[j] << "(" << predictions.at<double>(0,j) << ")  ";
  if (predictions.at<double>(0,j)>maxPredVal)
  { maxPredVal = predictions.at<double>(0,j); maxPredIdx = j;}
}

printf("\n The char sample is predicted as: %s \n\n", ascii[maxPredIdx]);
t = cvGetTickCount() - t;
printf("\n Time elapsed for individual sample prediction = %gms\n", t*1000./cv::getTickFrequency());


/* STEP 4. Save your classifier */
// Save the trained classifier
mlp.save("./trained_mlp.xml", "mlp");

return EXIT_SUCCESS;
}
