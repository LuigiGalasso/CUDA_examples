#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "GPU_rgb2gray_sobel.h"
#include <iostream>
#include <string>


using namespace cv;
using namespace std;


int main(void)
{
	cv::Mat gray;
	cv::Mat image;
	cv::Mat imageRGBA;
	cv::Mat imagesobel;	
	//image = cv::imread("./cinque_terre_small.jpg");
	image = cv::imread("./images.jpg");	
	gray.create(image.rows, image.cols, CV_8UC1);
	unsigned char h_in[image.rows*image.cols*3];	
	unsigned char h_out[image.rows*image.cols];
	if(image.empty())
	{
		cout << "image not found!!" << endl;
		system("pause");
		return -1;
	
	}
	cv::cvtColor(image,imageRGBA, COLOR_BGR2RGBA);
	cout << image.rows << endl << image.cols << endl;
	int w=0;
	for (int k=0;k<3;k++)
	{
		for(int i=0;i< image.rows;i++)
		{
			for(int j=0;j< image.cols;j++)
			{
				h_in[w]=image.at<Vec3b>(i,j)[k];				
				w++;			
			}
		}
	}

	rgb2gray_kernel_call(h_out,h_in,image.rows,image.cols);

	int i=0,j=0;
	for(int k=0;k< image.rows*image.cols;k++)
	{
		gray.at<unsigned char>(i,j)=h_out[k];
		j++;
		if(j==image.cols)
		{
			i++;
			j=0;
		}
	}

	imagesobel.create(image.rows, image.cols, CV_8UC1);
	unsigned char h_in1[image.rows*image.cols];	
	unsigned char h_out1[image.rows*image.cols];
	if(gray.empty())
	{
		cout << "image not found!!" << endl;
		system("pause");
		return -1;
	
	}
	
	cout << gray.rows << endl << gray.cols << endl;
	w=0;
		for(int i=0;i< gray.rows;i++)
		{
			for(int j=0;j< gray.cols;j++)
			{
				h_in1[w]=gray.at<unsigned char>(i,j);				
				w++;			
			}
		}
	sobel_kernel_call(h_out1,h_in1,image.rows,image.cols);


	 i=0,j=0;
	for(int k=0;k< gray.rows*gray.cols;k++)
	{
		imagesobel.at<unsigned char>(i,j)=h_out1[k];
		
		j++;
		if(j==gray.cols)
		{
			i++;
			j=0;
				
		}
		
	}


	
	String windowName="hello world";
	namedWindow(windowName);
	imshow(windowName, image);
	imshow("gray", gray);
	//cv::imwrite("gray.jpg",gray);
	imshow("sobel", imagesobel);

	waitKey(0);
	destroyWindow(windowName);
	return 0;
}

