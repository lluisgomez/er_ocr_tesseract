#include <tesseract/baseapi.h>
#include <tesseract/resultiterator.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
//#include <locale.h>

using namespace cv;
using namespace std;

enum 
{
  OCR_LEVEL_WORD     = 0,
  OCR_LEVEL_TEXTLINE = 1
};

class OCRTesseract
{
	private:
		tesseract::TessBaseAPI tess;

  public:
		//Default constructor
		OCRTesseract(); //(const char* datapath, const char* language, tesseract::OcrEngineMode oem, tesseract::PageSegMode psmode)
                    //(const char* char_whitelist, )

		~OCRTesseract();

	  void run(Mat& image, string& output_text, vector<Rect>* component_rects=NULL, 
             vector<string>* component_texts=NULL, vector<float>* component_confidences=NULL,
             int component_level=0);
	  //void run(Mat& image, vector<Rect> rois, string& output_text, vector<Rect>& component_rects=NULL, 
    //         vector<string>& component_texts=NULL, vector<float>& component_confidences=NULL);
	  //void run(Mat& img, vector<Rect>& rois, string& output);
	  //void run(Mat& img, vector<vector<Point> >& regions, string& output);
	  //void run(InputArrayOfArrays& channels, vector<vector<ERStat> >& regions, string& output);
};

