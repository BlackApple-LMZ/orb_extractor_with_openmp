
#include <iostream>
#include <opencv2/opencv.hpp>
#include "orb_extractor.h"
#include "orb_extractor1.h"
using namespace cv;
using namespace std;


#include <omp.h>
#include <chrono>
#include <vector>
#include <thread>


void test() {

	//通过private修饰该变量之后在并行区域内变为私有变量，进入并行   
		//区域后每个线程拥有该变量的拷贝，并且都不会初始化   
/*	int shared_to_private = 1;

#pragma omp parallel for firstprivate(shared_to_private)lastprivate(shared_to_private)
	for (int i = 0; i < 10; ++i)
	{
		//shared_to_private = i;
		std::cout << ++shared_to_private << std::endl;
	}
	cout << shared_to_private << endl;
	return ;
*/

	/*
	// 测试reduction解决数据共享的问题
	int sum = 0;
	std::cout << "Before: " << sum << std::endl;

#pragma omp parallel for //reduction(+:sum)   
	for (int i = 0; i < 10; ++i)
	{
		sum = sum + i;
		std::cout << sum << std::endl;
	}
	std::cout << "After: " << sum << std::endl;
	*/
	auto startTime = std::chrono::system_clock::now();
	int sum1 = 0, sum2 = 0;
	std::cout << "Before: " << sum1 << " " << sum2 << std::endl;
	
	/*
	//原子操作
#pragma omp parallel for   
	for (int i = 0; i < 2000; ++i)
	{
		#pragma omp atomic   
		sum++;
	}*/

	/*
	//section结合parallel 这个例子里面 不用openmp加速更快 但是意义不一样 
	#pragma omp parallel sections
	{
		#pragma omp section
		{
			#pragma omp parallel for   
			for (int i = 0; i < 2000; ++i)
			{
				#pragma omp atomic   
				sum1++;
			}
		}
		#pragma omp section
		{
			#pragma omp parallel for   
			for (int i = 0; i < 2000; ++i)
			{
				#pragma omp atomic   
				sum2++;
			}
		}
	}
	*/

	std::cout << "After: " << sum1 << " " << sum2 << std::endl;
	auto endTime = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = endTime - startTime;

	cout << elapsed_seconds.count() << endl;
}
void detectKeypoints() {
	ORB_OPENMP::ORBextractor* pORBextractor = new ORB_OPENMP::ORBextractor(1000, 1.2, 8, 20, 7);

	cv::Mat frame = cv::imread("E:\\project\\orb_extractor_with_openmp\\orb_extractor\\orb_extractor\\image\\runway.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//vector<Orb::Feature> features_l = TrackKeypoints(frame_l, orb);
	//vector<Orb::Feature> features_r = TrackKeypoints(frame_r, orb);
	//auto pairs = ty::BRIEF::matchFeatures_gpu(features_l, features_r, 64);
	clock_t t1 = clock();
	//double start = omp_get_wtime();

	/*
	for (int i = 0; i < 1000; i++) {
		cv::Mat frame = cv::imread("E:\\dataset\\lane\\image\\" + to_string(i) + ".jpg", CV_LOAD_IMAGE_GRAYSCALE);
		std::vector<cv::KeyPoint> vKeys;
		cv::Mat descriptors;
		(*pORBextractor)(frame, cv::Mat(), vKeys, descriptors);
	}
	*/
	std::vector<cv::KeyPoint> vKeys;
	cv::Mat descriptors;
	(*pORBextractor)(frame, cv::Mat(), vKeys, descriptors);
	//double end = omp_get_wtime();
	//printf_s("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start);
	clock_t t2 = clock();
	cout << "time: " << t2 - t1 << endl;
}
void detectKeypoints1() {
	ORB_OPENMP1::ORBextractor* pORBextractor = new ORB_OPENMP1::ORBextractor(1000, 1.2, 8, 20, 7);

	cv::Mat frame = cv::imread("E:\\project\\orb_extractor_with_openmp\\orb_extractor\\orb_extractor\\image\\runway.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	clock_t t1 = clock();

	std::vector<cv::KeyPoint> vKeys;
	cv::Mat descriptors;
	(*pORBextractor)(frame, cv::Mat(), vKeys, descriptors);
	//double end = omp_get_wtime();
	//printf_s("start = %.16g\nend = %.16g\ndiff = %.16g\n", start, end, end - start);
	clock_t t2 = clock();
	cout << "time1: " << t2 - t1 << endl;
}
int main3()
{
	//std::thread* ptorbExtractor = new thread(detectKeypoints);
	std::thread torbExtractor1(detectKeypoints);
	torbExtractor1.detach();
	std::thread torbExtractor2(detectKeypoints);
	torbExtractor2.detach();
	std::thread torbExtractor3(detectKeypoints1);
	torbExtractor3.detach();

	//std::thread t_test(test);
	//t_test.detach();
	//drawKeypoints(frame, vKeys, frame, cv::Scalar(0, 255, 0));
	cv::Mat frame = cv::imread("E:\\project\\orb_extractor_with_openmp\\orb_extractor\\orb_extractor\\image\\runway.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::imshow("output", frame);
	cv::waitKey(0);

	return 0;
}

int main2()
{
	test();
	//test
	//cout << ompt << endl;

	return 0;
}


int main1()
{
	Mat src = imread("C:\\Users\\lenovo\\Pictures\\11.png");

	if (src.empty()) {
		cout << "图片为空\n";
		return -1;

	}
	cout << "src.size " << src.size;
	namedWindow("TestImage", CV_WINDOW_NORMAL);
	imshow("TestImage", src);
	waitKey();

	return 0;
}