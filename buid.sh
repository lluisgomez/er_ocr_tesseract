
OPENCV_DIR='/home/lluis/Escriptori/GSoC2014/opencv/'

echo "-------------------------------------------------------------------------------------"
echo "you are compiling against 3.0 library in ${OPENCV_DIR}"
echo "-------------------------------------------------------------------------------------"

g++ -O3 -march='core2' -I${OPENCV_DIR}include -I${OPENCV_DIR}include/opencv2 -I${OPENCV_DIR}modules/video/include/ -I${OPENCV_DIR}modules/objdetect/include/ -I${OPENCV_DIR}modules/legacy/include/ -I${OPENCV_DIR}modules/calib3d/include/ -I${OPENCV_DIR}modules/ml/include/ -I${OPENCV_DIR}modules/core/include/ -I${OPENCV_DIR}modules/features2d/include/ -I${OPENCV_DIR}modules/photo/include/ -I${OPENCV_DIR}modules/imgproc/include/ -I${OPENCV_DIR}modules/flann/include/ -I${OPENCV_DIR}modules/highgui/include/ -I${OPENCV_DIR}modules/contrib/include/ -c ocr_tesseract.cpp -o ocr_tesseract.o

g++ -O3 -march='core2' -I${OPENCV_DIR}include -I${OPENCV_DIR}include/opencv2 -I${OPENCV_DIR}modules/video/include/ -I${OPENCV_DIR}modules/objdetect/include/ -I${OPENCV_DIR}modules/legacy/include/ -I${OPENCV_DIR}modules/calib3d/include/ -I${OPENCV_DIR}modules/ml/include/ -I${OPENCV_DIR}modules/core/include/ -I${OPENCV_DIR}modules/features2d/include/ -I${OPENCV_DIR}modules/photo/include/ -I${OPENCV_DIR}modules/imgproc/include/ -I${OPENCV_DIR}modules/flann/include/ -I${OPENCV_DIR}modules/highgui/include/ -I${OPENCV_DIR}modules/contrib/include/ -c end_to_end_recognition.cpp -o end_to_end_recognition.o

libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o end_to_end_recognition ocr_tesseract.o end_to_end_recognition.o -L${OPENCV_DIR}lib/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_video -lopencv_videostab  -ltesseract

g++ -O3 -march='core2' -I${OPENCV_DIR}include -I${OPENCV_DIR}include/opencv2 -I${OPENCV_DIR}modules/video/include/ -I${OPENCV_DIR}modules/objdetect/include/ -I${OPENCV_DIR}modules/legacy/include/ -I${OPENCV_DIR}modules/calib3d/include/ -I${OPENCV_DIR}modules/ml/include/ -I${OPENCV_DIR}modules/core/include/ -I${OPENCV_DIR}modules/features2d/include/ -I${OPENCV_DIR}modules/photo/include/ -I${OPENCV_DIR}modules/imgproc/include/ -I${OPENCV_DIR}modules/flann/include/ -I${OPENCV_DIR}modules/highgui/include/ -I${OPENCV_DIR}modules/contrib/include/ -c pipeline_comparison.cpp -o pipeline_comparison.o

libtool --tag=CXX --mode=link g++ -O3 -march='core2' -o pipeline_comparison ocr_tesseract.o pipeline_comparison.o -L${OPENCV_DIR}lib/ -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_video -lopencv_videostab  -ltesseract

red='\033[0;31m'
NC='\033[0m' # No Color
echo "${red}-------------------------------------------------------------------------------------"
echo "remember to export export LD_LIBRARY_PATH=${OPENCV_DIR}lib/"
echo "-------------------------------------------------------------------------------------${NC}"
