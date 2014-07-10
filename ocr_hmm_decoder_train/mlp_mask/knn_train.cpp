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

int num_samples = 7192; //number of lines in the dataset file
int num_classes = 62; //number of classes
int num_features = 200; //number of features

Mat hus = Mat(num_samples,num_features,CV_32FC1);
Mat labels = Mat::ones(num_samples,1,CV_32FC1);
labels = labels * -1;
int label;
int count = 0;

while (infile >> label >> hus.at<float>(count,0) >> hus.at<float>(count,1) >> hus.at<float>(count,2) >> hus.at<float>(count,3) >> hus.at<float>(count,4) >> hus.at<float>(count,5) >> hus.at<float>(count,6) >> hus.at<float>(count,7) >> hus.at<float>(count,8) >> hus.at<float>(count,9) >> hus.at<float>(count,10) >> hus.at<float>(count,11) >> hus.at<float>(count,12) >> hus.at<float>(count,13) >> hus.at<float>(count,14) >> hus.at<float>(count,15) >> hus.at<float>(count,16) >> hus.at<float>(count,17) >> hus.at<float>(count,18) >> hus.at<float>(count,19) >> hus.at<float>(count,20) >> hus.at<float>(count,21) >> hus.at<float>(count,22) >> hus.at<float>(count,23) >> hus.at<float>(count,24) >> hus.at<float>(count,25) >> hus.at<float>(count,26) >> hus.at<float>(count,27) >> hus.at<float>(count,28) >> hus.at<float>(count,29) >> hus.at<float>(count,30) >> hus.at<float>(count,31) >> hus.at<float>(count,32) >> hus.at<float>(count,33) >> hus.at<float>(count,34) >> hus.at<float>(count,35) >> hus.at<float>(count,36) >> hus.at<float>(count,37) >> hus.at<float>(count,38) >> hus.at<float>(count,39) >> hus.at<float>(count,40) >> hus.at<float>(count,41) >> hus.at<float>(count,42) >> hus.at<float>(count,43) >> hus.at<float>(count,44) >> hus.at<float>(count,45) >> hus.at<float>(count,46) >> hus.at<float>(count,47) >> hus.at<float>(count,48) >> hus.at<float>(count,49) >> hus.at<float>(count,50) >> hus.at<float>(count,51) >> hus.at<float>(count,52) >> hus.at<float>(count,53) >> hus.at<float>(count,54) >> hus.at<float>(count,55) >> hus.at<float>(count,56) >> hus.at<float>(count,57) >> hus.at<float>(count,58) >> hus.at<float>(count,59) >> hus.at<float>(count,60) >> hus.at<float>(count,61) >> hus.at<float>(count,62) >> hus.at<float>(count,63) >> hus.at<float>(count,64) >> hus.at<float>(count,65) >> hus.at<float>(count,66) >> hus.at<float>(count,67) >> hus.at<float>(count,68) >> hus.at<float>(count,69) >> hus.at<float>(count,70) >> hus.at<float>(count,71) >> hus.at<float>(count,72) >> hus.at<float>(count,73) >> hus.at<float>(count,74) >> hus.at<float>(count,75) >> hus.at<float>(count,76) >> hus.at<float>(count,77) >> hus.at<float>(count,78) >> hus.at<float>(count,79) >> hus.at<float>(count,80) >> hus.at<float>(count,81) >> hus.at<float>(count,82) >> hus.at<float>(count,83) >> hus.at<float>(count,84) >> hus.at<float>(count,85) >> hus.at<float>(count,86) >> hus.at<float>(count,87) >> hus.at<float>(count,88) >> hus.at<float>(count,89) >> hus.at<float>(count,90) >> hus.at<float>(count,91) >> hus.at<float>(count,92) >> hus.at<float>(count,93) >> hus.at<float>(count,94) >> hus.at<float>(count,95) >> hus.at<float>(count,96) >> hus.at<float>(count,97) >> hus.at<float>(count,98) >> hus.at<float>(count,99) >> hus.at<float>(count,100) >> hus.at<float>(count,101) >> hus.at<float>(count,102) >> hus.at<float>(count,103) >> hus.at<float>(count,104) >> hus.at<float>(count,105) >> hus.at<float>(count,106) >> hus.at<float>(count,107) >> hus.at<float>(count,108) >> hus.at<float>(count,109) >> hus.at<float>(count,110) >> hus.at<float>(count,111) >> hus.at<float>(count,112) >> hus.at<float>(count,113) >> hus.at<float>(count,114) >> hus.at<float>(count,115) >> hus.at<float>(count,116) >> hus.at<float>(count,117) >> hus.at<float>(count,118) >> hus.at<float>(count,119) >> hus.at<float>(count,120) >> hus.at<float>(count,121) >> hus.at<float>(count,122) >> hus.at<float>(count,123) >> hus.at<float>(count,124) >> hus.at<float>(count,125) >> hus.at<float>(count,126) >> hus.at<float>(count,127) >> hus.at<float>(count,128) >> hus.at<float>(count,129) >> hus.at<float>(count,130) >> hus.at<float>(count,131) >> hus.at<float>(count,132) >> hus.at<float>(count,133) >> hus.at<float>(count,134) >> hus.at<float>(count,135) >> hus.at<float>(count,136) >> hus.at<float>(count,137) >> hus.at<float>(count,138) >> hus.at<float>(count,139) >> hus.at<float>(count,140) >> hus.at<float>(count,141) >> hus.at<float>(count,142) >> hus.at<float>(count,143) >> hus.at<float>(count,144) >> hus.at<float>(count,145) >> hus.at<float>(count,146) >> hus.at<float>(count,147) >> hus.at<float>(count,148) >> hus.at<float>(count,149) >> hus.at<float>(count,150) >> hus.at<float>(count,151) >> hus.at<float>(count,152) >> hus.at<float>(count,153) >> hus.at<float>(count,154) >> hus.at<float>(count,155) >> hus.at<float>(count,156) >> hus.at<float>(count,157) >> hus.at<float>(count,158) >> hus.at<float>(count,159) >> hus.at<float>(count,160) >> hus.at<float>(count,161) >> hus.at<float>(count,162) >> hus.at<float>(count,163) >> hus.at<float>(count,164) >> hus.at<float>(count,165) >> hus.at<float>(count,166) >> hus.at<float>(count,167) >> hus.at<float>(count,168) >> hus.at<float>(count,169) >> hus.at<float>(count,170) >> hus.at<float>(count,171) >> hus.at<float>(count,172) >> hus.at<float>(count,173) >> hus.at<float>(count,174) >> hus.at<float>(count,175) >> hus.at<float>(count,176) >> hus.at<float>(count,177) >> hus.at<float>(count,178) >> hus.at<float>(count,179) >> hus.at<float>(count,180) >> hus.at<float>(count,181) >> hus.at<float>(count,182) >> hus.at<float>(count,183) >> hus.at<float>(count,184) >> hus.at<float>(count,185) >> hus.at<float>(count,186) >> hus.at<float>(count,187) >> hus.at<float>(count,188) >> hus.at<float>(count,189) >> hus.at<float>(count,190) >> hus.at<float>(count,191) >> hus.at<float>(count,192) >> hus.at<float>(count,193) >> hus.at<float>(count,194) >> hus.at<float>(count,195) >> hus.at<float>(count,196) >> hus.at<float>(count,197) >> hus.at<float>(count,198) >> hus.at<float>(count,199))
{
  //cout << hus.row(count) << endl;
  labels.at<float>(count,1) = label;
  count++;
}
cv::FileStorage storage("knn_model_data.xml", cv::FileStorage::WRITE);
storage << "hus" << hus;
storage << "labels" << labels;
storage.release();  

/* STEP 2. Create and train the KNN model */
//1. Declare the classifier
CvKNearest knn;


ifstream ifile("./trained_knn.xml");
if (ifile) 
{
	// The file exists, so we don't need to train 
	printf("Loading model from file ... \n");
	knn.load("./trained_knn.xml", "knn");
} else {
	//2. Train it 
	printf("Training ... \n");
	knn.train(hus, labels, Mat(), false, 32);
	printf("Done! \n");

  /* STEP 3. Save your classifier */
  // Save the trained classifier
  //knn.save("./trained_knn.xml", "knn");
}

/* STEP 4. Calculating the testing and training error */
Mat predictions;
knn.find_nearest(hus,11,&predictions);
int tp=0;
int fp=0;
for (int i=0; i<predictions.rows; i++)
{

  if (labels.at<float>(i,0) == predictions.at<float>(i,0))
      tp++;
  else 
      fp++;
}
cout << "Train Accuracy = " << (float)tp/(tp+fp) << endl;


ifstream test_infile("data_test.txt");

num_samples = 7192; 
hus = Mat(num_samples,num_features,CV_32FC1);
labels = Mat::ones(num_samples,num_classes,CV_32FC1);
labels = labels * -1;
count = 0;

while (test_infile >> label >> hus.at<float>(count,0) >> hus.at<float>(count,1) >> hus.at<float>(count,2) >> hus.at<float>(count,3) >> hus.at<float>(count,4) >> hus.at<float>(count,5) >> hus.at<float>(count,6) >> hus.at<float>(count,7) >> hus.at<float>(count,8) >> hus.at<float>(count,9) >> hus.at<float>(count,10) >> hus.at<float>(count,11) >> hus.at<float>(count,12) >> hus.at<float>(count,13) >> hus.at<float>(count,14) >> hus.at<float>(count,15) >> hus.at<float>(count,16) >> hus.at<float>(count,17) >> hus.at<float>(count,18) >> hus.at<float>(count,19) >> hus.at<float>(count,20) >> hus.at<float>(count,21) >> hus.at<float>(count,22) >> hus.at<float>(count,23) >> hus.at<float>(count,24) >> hus.at<float>(count,25) >> hus.at<float>(count,26) >> hus.at<float>(count,27) >> hus.at<float>(count,28) >> hus.at<float>(count,29) >> hus.at<float>(count,30) >> hus.at<float>(count,31) >> hus.at<float>(count,32) >> hus.at<float>(count,33) >> hus.at<float>(count,34) >> hus.at<float>(count,35) >> hus.at<float>(count,36) >> hus.at<float>(count,37) >> hus.at<float>(count,38) >> hus.at<float>(count,39) >> hus.at<float>(count,40) >> hus.at<float>(count,41) >> hus.at<float>(count,42) >> hus.at<float>(count,43) >> hus.at<float>(count,44) >> hus.at<float>(count,45) >> hus.at<float>(count,46) >> hus.at<float>(count,47) >> hus.at<float>(count,48) >> hus.at<float>(count,49) >> hus.at<float>(count,50) >> hus.at<float>(count,51) >> hus.at<float>(count,52) >> hus.at<float>(count,53) >> hus.at<float>(count,54) >> hus.at<float>(count,55) >> hus.at<float>(count,56) >> hus.at<float>(count,57) >> hus.at<float>(count,58) >> hus.at<float>(count,59) >> hus.at<float>(count,60) >> hus.at<float>(count,61) >> hus.at<float>(count,62) >> hus.at<float>(count,63) >> hus.at<float>(count,64) >> hus.at<float>(count,65) >> hus.at<float>(count,66) >> hus.at<float>(count,67) >> hus.at<float>(count,68) >> hus.at<float>(count,69) >> hus.at<float>(count,70) >> hus.at<float>(count,71) >> hus.at<float>(count,72) >> hus.at<float>(count,73) >> hus.at<float>(count,74) >> hus.at<float>(count,75) >> hus.at<float>(count,76) >> hus.at<float>(count,77) >> hus.at<float>(count,78) >> hus.at<float>(count,79) >> hus.at<float>(count,80) >> hus.at<float>(count,81) >> hus.at<float>(count,82) >> hus.at<float>(count,83) >> hus.at<float>(count,84) >> hus.at<float>(count,85) >> hus.at<float>(count,86) >> hus.at<float>(count,87) >> hus.at<float>(count,88) >> hus.at<float>(count,89) >> hus.at<float>(count,90) >> hus.at<float>(count,91) >> hus.at<float>(count,92) >> hus.at<float>(count,93) >> hus.at<float>(count,94) >> hus.at<float>(count,95) >> hus.at<float>(count,96) >> hus.at<float>(count,97) >> hus.at<float>(count,98) >> hus.at<float>(count,99) >> hus.at<float>(count,100) >> hus.at<float>(count,101) >> hus.at<float>(count,102) >> hus.at<float>(count,103) >> hus.at<float>(count,104) >> hus.at<float>(count,105) >> hus.at<float>(count,106) >> hus.at<float>(count,107) >> hus.at<float>(count,108) >> hus.at<float>(count,109) >> hus.at<float>(count,110) >> hus.at<float>(count,111) >> hus.at<float>(count,112) >> hus.at<float>(count,113) >> hus.at<float>(count,114) >> hus.at<float>(count,115) >> hus.at<float>(count,116) >> hus.at<float>(count,117) >> hus.at<float>(count,118) >> hus.at<float>(count,119) >> hus.at<float>(count,120) >> hus.at<float>(count,121) >> hus.at<float>(count,122) >> hus.at<float>(count,123) >> hus.at<float>(count,124) >> hus.at<float>(count,125) >> hus.at<float>(count,126) >> hus.at<float>(count,127) >> hus.at<float>(count,128) >> hus.at<float>(count,129) >> hus.at<float>(count,130) >> hus.at<float>(count,131) >> hus.at<float>(count,132) >> hus.at<float>(count,133) >> hus.at<float>(count,134) >> hus.at<float>(count,135) >> hus.at<float>(count,136) >> hus.at<float>(count,137) >> hus.at<float>(count,138) >> hus.at<float>(count,139) >> hus.at<float>(count,140) >> hus.at<float>(count,141) >> hus.at<float>(count,142) >> hus.at<float>(count,143) >> hus.at<float>(count,144) >> hus.at<float>(count,145) >> hus.at<float>(count,146) >> hus.at<float>(count,147) >> hus.at<float>(count,148) >> hus.at<float>(count,149) >> hus.at<float>(count,150) >> hus.at<float>(count,151) >> hus.at<float>(count,152) >> hus.at<float>(count,153) >> hus.at<float>(count,154) >> hus.at<float>(count,155) >> hus.at<float>(count,156) >> hus.at<float>(count,157) >> hus.at<float>(count,158) >> hus.at<float>(count,159) >> hus.at<float>(count,160) >> hus.at<float>(count,161) >> hus.at<float>(count,162) >> hus.at<float>(count,163) >> hus.at<float>(count,164) >> hus.at<float>(count,165) >> hus.at<float>(count,166) >> hus.at<float>(count,167) >> hus.at<float>(count,168) >> hus.at<float>(count,169) >> hus.at<float>(count,170) >> hus.at<float>(count,171) >> hus.at<float>(count,172) >> hus.at<float>(count,173) >> hus.at<float>(count,174) >> hus.at<float>(count,175) >> hus.at<float>(count,176) >> hus.at<float>(count,177) >> hus.at<float>(count,178) >> hus.at<float>(count,179) >> hus.at<float>(count,180) >> hus.at<float>(count,181) >> hus.at<float>(count,182) >> hus.at<float>(count,183) >> hus.at<float>(count,184) >> hus.at<float>(count,185) >> hus.at<float>(count,186) >> hus.at<float>(count,187) >> hus.at<float>(count,188) >> hus.at<float>(count,189) >> hus.at<float>(count,190) >> hus.at<float>(count,191) >> hus.at<float>(count,192) >> hus.at<float>(count,193) >> hus.at<float>(count,194) >> hus.at<float>(count,195) >> hus.at<float>(count,196) >> hus.at<float>(count,197) >> hus.at<float>(count,198) >> hus.at<float>(count,199))
{

  labels.at<float>(count,0) = label;
  count++;
}

knn.find_nearest(hus,11,&predictions);
tp=0;
fp=0;
for (int i=0; i<predictions.rows; i++)
{

  if (labels.at<float>(i,0) == predictions.at<float>(i,0))
      tp++;
  else 
      fp++;
}
cout << "Test Accuracy = " << (float)tp/(tp+fp) << endl;

//Predict an individual sample
float t = cvGetTickCount();
static const float arr[] = {0, 0, 0, 0, 0, 0, 0.08403361344537814, 0.4134453781512605, 0, 0, 0, 0.02817126850740296, 0.1382953181272509, 0, 0, 0.007763105242096838, 0.226890756302521, 0.3563025210084033, 0.1228491396558623, 0, 0.01728691476590636, 0.2735494197679071, 0.1229291716686675, 0.2688275310124049, 0, 0, 0, 0, 0, 0, 0, 0, 0.4460984393757502, 0.04153661464585834, 0, 0, 0, 0.07354941976790716, 0.003761504601840736, 0, 0, 0, 0.2674669867947179, 0.03073229291716686, 0, 0, 0, 0, 0, 0, 0.0330532212885154, 0.1086034413765506, 0, 0, 0, 0.1591836734693877, 0.5202881152460984, 0.2996398559423769, 0.05186074429771909, 0, 0.1624649859943977, 0.5305322128851541, 0.06322529011604641, 0.01528611444577831, 0, 0.1591836734693877, 0.5202881152460984, 0.3968787515006002, 0.09563825530212083, 0, 0.0330532212885154, 0.1086034413765506, 0.114125650260104, 0.02737094837935174, 0, 0, 0, 0, 0, 0, 0, 0, 0.1571028411364546, 0.02088835534213685, 0, 0, 0, 0.2801120448179271, 0.0224889955982393, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01728691476590636, 0.4008803521408563, 0.4756302521008403, 0.3643857543017207, 0.005442176870748299, 0.007763105242096838, 0.2, 0.2976390556222489, 0.166546618647459, 0.002561024409763905, 0, 0.06762705082032812, 0.3322128851540616, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0602641056422569, 0.0399359743897559, 0, 0, 0, 0.1369347739095638, 0.1018807523009204, 0.4272909163665466, 0.003841536614645858, 0, 0, 0, 0.230812324929972, 0.01536614645858343, 0, 0, 0, 0, 0, 0, 0, 0, 0.1256502601040416, 0.01600640256102441, 0, 0.1643857543017207, 0.2143257302921168, 0.6086434573829531, 0.07186874749899959, 0, 0.07891156462585033, 0.04233693477390956, 0.6837134853941575, 0.01056422569027611, 0, 0.3807923169267707, 0.09027611044417767, 0.6300120048019208, 0.03057222889155662, 0, 0.07458983593437375, 0.01784713885554222, 0.09243697478991596, 0, 0, 0, 0, 0, 0, 0, 0.1509403761504602, 0.338375350140056, 0.1389355742296919, 0.007282913165266106, 0, 0.0399359743897559, 0.05986394557823128, 0.3394957983193277, 0.006722689075630252, 0, 0.2946778711484594, 0.1931972789115646, 0.3236494597839135, 0.01400560224089636, 0, 0.1374149659863945, 0.01656662665066026, 0.1535814325730292, 0}; //R

hus = Mat(1,num_features,CV_32FC1,(void*)&arr);
Mat responses,dists;
knn.find_nearest( hus, 11, &predictions, 0, &responses, &dists);

Scalar dist_sum = sum(dists);
Mat class_predictions = Mat::zeros(1,num_classes,CV_64FC1);

printf("\n K nearest responses: ");
for (int j=0; j<responses.cols; j++)
{
    cout << ascii[(int)responses.at<float>(0,j)] << "(" << dists.at<float>(0,j) << ")  ";
    class_predictions.at<double>(0,(int)responses.at<float>(0,j)) += dists.at<float>(0,j);
}

class_predictions = class_predictions/dist_sum[0];

printf("\n The char sample is predicted as: %s \n\n", ascii[(int)predictions.at<float>(0,0)]);
t = cvGetTickCount() - t;
printf("\n Time elapsed for individual sample prediction = %gms\n", t*1000./cv::getTickFrequency());


return EXIT_SUCCESS;
}
