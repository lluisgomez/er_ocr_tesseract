#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;


enum decoder_mode
{
    DECODER_VITERBI = 0 // Other algorithms may be added
};

class CV_EXPORTS OCRHMMDecoder : public Algorithm
{
public:

    //! callback with the character classifier is made a class. This way we hide the feature extractor and the classifier itself
    class CV_EXPORTS ClassifierCallback
    {
    public:
        virtual ~ClassifierCallback() { }
        //! The classifier must return a (ranked list of) class(es) id('s)
        virtual void eval( InputArray src, InputArray mask, vector<int>& out_class, vector<double>& out_confidence) = 0;
    };

    //! Constructor
    OCRHMMDecoder( const Ptr<OCRHMMDecoder::ClassifierCallback> classifier,// The character classifier with built in feature extractor
                   string& vocabulary,                               // The language vocabulary (chars when ascii english text)
                                                                     //     size() must be equal to the number of classes
                   InputArray transition_probabilities_table,        // Table with transition probabilities between character pairs
                                                                     //     cols == rows == vocabulari.size()
                   InputArray emission_probabilities_table,          // Table with observation emission probabilities 
                                                                     //     cols == rows == vocabulari.size()
                   decoder_mode mode = DECODER_VITERBI);             // HMM Decoding algorithm (only Viterbi for the moment)

    ~OCRHMMDecoder();

    //! Decode a group of regions and output the most likely sequence of characters
    // output probability of the output sequence
    double run( InputArray src,              // RGB or greyscale original image (in case the feature extractor needs it)
              InputArray mask,               // single channel image with labeled regions
              string& out_sequence,          // output the most likely sequence
	            vector<Rect>* component_rects=NULL, 
              vector<string>* component_texts=NULL, 
              vector<float>* component_confidences=NULL,
              int component_level=0);  // specify words, lines, etc...

protected:

    Ptr<OCRHMMDecoder::ClassifierCallback> classifier;
    string vocabulary;
    Mat transition_p;
    Mat emission_p;
    decoder_mode mode;
};

Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierMLP(const std::string& filename);
Ptr<OCRHMMDecoder::ClassifierCallback> loadOCRHMMClassifierKNN(const std::string& filename);

