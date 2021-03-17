
//#include "Orb.h"
#include "orb_extractor.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <time.h>
#include <chrono>
#include <thread>

using namespace std;


int main()
{
	ORB_OPENMP::ORBextractor* pORBextractor = new ORB_OPENMP::ORBextractor(1000, 1.2, 8, 20, 7);
	cv::Mat frame = cv::imread("E:\\project\\orb_extractor_with_openmp\\orb_extractor\\orb_extractor\\image\\runway.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	clock_t t1 = clock();
	std::vector<cv::KeyPoint> vKeys;
	cv::Mat descriptors;
	(*pORBextractor)(frame, cv::Mat(), vKeys, descriptors);
	clock_t t2 = clock();
	cout << "time: " << t2 - t1 << endl;

	drawKeypoints(frame, vKeys, frame, cv::Scalar(0, 255, 0));
	cv::imshow("output", frame);
	cv::waitKey(0);

	return 0;
}

