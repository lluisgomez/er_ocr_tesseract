#include "ocr_tesseract.h"

//Default constructor
OCRTesseract::OCRTesseract(const char* datapath, const char* language, const char* char_whitelist, tesseract::OcrEngineMode oemode, tesseract::PageSegMode psmode)
{

  const char *lang = "eng";
  if (language != NULL)
    lang = language;

  if (tess.Init(datapath, lang, oemode))
  {
    cout << "OCRTesseract: Could not initialize tesseract." << endl;
    throw 1;
  } 

  cout << "OCRTesseract: tesseract version " << tess.Version() << endl;

  tesseract::PageSegMode pagesegmode = psmode;
  tess.SetPageSegMode(pagesegmode);

  if(char_whitelist != NULL)
    tess.SetVariable("tessedit_char_whitelist", char_whitelist);
  else
    tess.SetVariable("tessedit_char_whitelist", "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

  tess.SetVariable("save_best_choices", "T");

}

OCRTesseract::~OCRTesseract()
{
  tess.End();
}

void OCRTesseract::run(Mat& image, string& output, vector<Rect>* component_rects, 
                       vector<string>* component_texts, vector<float>* component_confidences, int component_level)
{
  tess.SetImage((uchar*)image.data, image.size().width, image.size().height, image.channels(), image.step1());
  tess.Recognize(0); 
  output = string(tess.GetUTF8Text());

  if ( (component_rects != NULL) || (component_texts != NULL) || (component_confidences != NULL) )
  {
    tesseract::ResultIterator* ri = tess.GetIterator();
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (component_level == OCR_LEVEL_TEXTLINE)
      level = tesseract::RIL_TEXTLINE;

    if (ri != 0) {
      do {
        const char* word = ri->GetUTF8Text(level);
        if (word == NULL)
          continue;
        float conf = ri->Confidence(level);
        int x1, y1, x2, y2;
        ri->BoundingBox(level, &x1, &y1, &x2, &y2);

        if (component_texts != 0)
          component_texts->push_back(string(word));
        if (component_rects != 0)
          component_rects->push_back(Rect(x1,y1,x2-x1,y2-y1));
        if (component_confidences != 0)
          component_confidences->push_back(conf);

        delete[] word;
      } while (ri->Next(level));
    }
    delete ri;
  }

  /*tesseract::ResultIterator* ri = tess.GetIterator();
  tesseract::ChoiceIterator* ci; 

  if(ri != 0)
  {
    do
    {
      const char* symbol = ri->GetUTF8Text(tesseract::RIL_SYMBOL);

      if(symbol != 0)
      {
          float conf = ri->Confidence(tesseract::RIL_SYMBOL); 
          std::cout << "\tnext symbol: " << symbol << "\tconf: " << conf << "\n"; 

          const tesseract::ResultIterator itr = *ri; 
          ci = new tesseract::ChoiceIterator(itr);

          do
          {
              const char* choice = ci->GetUTF8Text(); 
              std::cout << "\t\t" << choice << " conf: " << ci->Confidence() << "\n"; 
          }
          while(ci->Next()); 

          delete ci; 
      }

      delete[] symbol;
    }
    while((ri->Next(tesseract::RIL_SYMBOL)));
  }
  delete ri;*/ 

  tess.Clear();
}
