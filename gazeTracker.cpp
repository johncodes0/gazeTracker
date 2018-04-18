#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <pthread.h>
#include <atomic> 
#include <stdio.h>
#include <time.h>
#include <unistd.h>


std::string file = "trimalarm.wav";
std::string command = "aplay " + file;
std::string facefile = "face.wav"; 
std::string command1 = "aplay " + facefile; 
#if defined (_APPLE_) && (_MACH_)
    command = "afplay " + file;
#endif
#if defined (_linux_)
    command = "aplay " + file;
#endif
time_t start; // sleeping eyes
time_t face; // face detection 

cv::Vec3f getEyeball(cv::Mat &eye, std::vector<cv::Vec3f> &circles)
{
  std::vector<int> sums(circles.size(), 0);
  for (int y = 0; y < eye.rows; y++)
  {
      uchar *ptr = eye.ptr<uchar>(y);
      for (int x = 0; x < eye.cols; x++)
      {
          int value = static_cast<int>(*ptr);
          for (int i = 0; i < circles.size(); i++)
          {
              cv::Point center((int)std::round(circles[i][0]), (int)std::round(circles[i][1]));
              int radius = (int)std::round(circles[i][2]);
              if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2))
              {
                  sums[i] += value;
              }
          }
          ++ptr;
      }
  }
  int smallestSum = 9999999;
  int smallestSumIndex = -1;
  for (int i = 0; i < circles.size(); i++)
  {
      if (sums[i] < smallestSum)
      {
          smallestSum = sums[i];
          smallestSumIndex = i;
      }
  }
  return circles[smallestSumIndex];
}

cv::Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
{
  int leftmost = 99999999;
  int leftmostIndex = -1;
  for (int i = 0; i < eyes.size(); i++)
  {
      if (eyes[i].tl().x < leftmost)
      {
          leftmost = eyes[i].tl().x;
          leftmostIndex = i;
      }
  }
  return eyes[leftmostIndex];
}

cv::Rect getRightmostEye(std::vector<cv::Rect> &eyes){

  int rightmost = 0;
  int rightmostIndex = -1;
  for (int i = 0; i < eyes.size(); i++)
  {
      if (eyes[i].tl().x > rightmost)
      {
          rightmost = eyes[i].tl().x;
          rightmostIndex = i;
      }
  }
  
  return eyes[rightmostIndex];
}



std::vector<cv::Point> centersLeft;
std::vector<cv::Point> centersRight;
cv::Point lastPointLeft;
cv::Point lastPointRight;

cv::Point stabilize(std::vector<cv::Point> &points, int windowSize)
{
  float sumX = 0;
  float sumY = 0;
  int count = 0;
  for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++)
  {
      sumX += points[i].x;
      sumY += points[i].y;
      ++count;
  }
  if (count > 0)
  {
      sumX /= count;
      sumY /= count;
  }
  return cv::Point(sumX, sumY);
}

int noFaceCounter = 0;

void detectface(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{ 
    cv::Mat grayscale;
    cv::cvtColor(frame, grayscale, CV_BGR2GRAY); // convert current frame to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast ??
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
    if (faces.size() == 0) {
      
        
        return; // none face was detected
    }
    else { 
      face = time(0); 
      }
    
    cv::Mat face = grayscale(faces[0]); // crop the face to the matrix within the rectangle of face 0
    
    cv::imshow("face_gray",face);
    std::vector<cv::Rect> eyes;
    //The coordinates of the eyes are in relation to the face not the absolute coordinate system
    eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30)); // same thing as above
    rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
    
    // Testing how the rectangle function works.
    // This shows that the coordinate system 0,0 starts in the top left of the screen
    // rectangle(frame,cv::Point(0,0),cv::Point(100,100),cv::Scalar(0,255,0),2);
    
    if (eyes.size() != 2){
        noFaceCounter++;
        double time1 = difftime(time(0), start);
        printf("\n Time: %f",time1);
        if (noFaceCounter > 10){
            //system(command.c_str());
        }
        return; // both eyes were not detected
    }
    else {
            //start = time(0); 
    }
    noFaceCounter = 0;
    
    for (cv::Rect &eye : eyes)
    {
        //         rectangle(frame, eye.tl(), eye.br(), cv::Scalar(0, 255, 0), 2);
        rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
    }
    
    //Gets the leftmosteye based on the x coordinate of the eyes
    cv::Rect eyeLeftRect = getLeftmostEye(eyes);
    cv::Rect eyeRightRect = getRightmostEye(eyes);
    
    cv::Mat eyeLeft = face(eyeLeftRect); // crop the leftmost eye
    cv::Mat eyeRight = face(eyeRightRect); //crop the rightmost eye
    
    //Shows the difference of before and after equalizing the histogram.
    //   cv::imshow("eye_nohist",eye);
    cv::equalizeHist(eyeLeft, eyeLeft);
    //   cv::imshow("eye_equalized",eye);
    cv::equalizeHist(eyeRight,eyeRight);
    
    //Finds the pupils of the left and right eye.
    std::vector<cv::Vec3f> circlesLeft;
    std::vector<cv::Vec3f> circlesRight;
    cv::HoughCircles(eyeLeft, circlesLeft, CV_HOUGH_GRADIENT, 1, eyeLeft.cols / 8, 250, 15, eyeLeft.rows / 8, eyeLeft.rows / 3);
    cv::HoughCircles(eyeRight, circlesRight, CV_HOUGH_GRADIENT, 1, eyeRight.cols / 8, 250, 15, eyeRight.rows / 8, eyeRight.rows / 3);
    
    
    if (circlesLeft.size() > 0)
    {
        cv::Vec3f eyeballLeft = getEyeball(eyeLeft, circlesLeft);
        cv::Point centerLeft(eyeballLeft[0], eyeballLeft[1]);
        centersLeft.push_back(centerLeft);
        centerLeft = stabilize(centersLeft, 5);
        if (centersLeft.size() > 1)
        {
            cv::Point diffLeft;
            diffLeft.x = (centerLeft.x - lastPointLeft.x);
            diffLeft.y = (centerLeft.y - lastPointLeft.y);
            //I want to know what this difference is and how much small I would need it to be gazing
        }
        lastPointLeft = centerLeft;
        int radiusLeft = (int)eyeballLeft[2];
        cv::circle(frame, faces[0].tl() + eyeLeftRect.tl() + centerLeft, radiusLeft, cv::Scalar(0, 0, 255), 2);
        cv::circle(eyeLeft, centerLeft, radiusLeft, cv::Scalar(255, 255, 255), 2);
        start = time(0);
    }
    if (circlesRight.size() > 0)
    {
        cv::Vec3f eyeballRight = getEyeball(eyeRight, circlesRight);
        cv::Point centerRight(eyeballRight[0], eyeballRight[1]);
        centersRight.push_back(centerRight);
        centerRight = stabilize(centersRight, 5);
        if (centersRight.size() > 1)
        {
            cv::Point diffRight;
            diffRight.x = (centerRight.x - lastPointRight.x);
            diffRight.y = (centerRight.y - lastPointRight.y);
        }
        lastPointRight = centerRight;
        int radiusRight = (int)eyeballRight[2];
        cv::circle(frame, faces[0].tl() + eyeRightRect.tl() + centerRight, radiusRight, cv::Scalar(0, 0, 255), 2);
        cv::circle(eyeRight, centerRight, radiusRight, cv::Scalar(255, 255, 255), 2);
        start = time(0);
    }

    cv::imshow("EyeLeft", eyeLeft);
    cv::imshow("EyeRight",eyeRight);
}

void *trackingsystem(void *no)
{
  start = 0;
  face = 0; 
  cv::CascadeClassifier faceCascade;
  cv::CascadeClassifier eyeCascade;
  if (!faceCascade.load("./haarcascade_frontalface_alt.xml"))
  {
      std::cerr << "Could not load face detector." << std::endl;
      //return -1;
  }    
  if (!eyeCascade.load("./haarcascade_eye_tree_eyeglasses.xml"))
  {
      std::cerr << "Could not load eye detector." << std::endl;
      //return -1;
  }
  // Open VideoCapture
  cv::VideoCapture capture;
  capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n");} //return -1; }
  
  
  cv::Mat frame;
  // Continually read frames
  while (capture.read(frame))
  {
      capture >> frame; // outputs the webcam image to a Mat
      if (!frame.data) break;
      detectface(frame, faceCascade, eyeCascade);
      cv::imshow("Webcam", frame); // displays the Mat
      cv::waitKey(30); // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
  }

}
void *alertsystem(void *no)
{
  int sec = 1; 
  while(1)
  {
    double eye = difftime(time(0), start);
    double warning = difftime(time(0),face); 
  if ( eye  > 9){
      system(command.c_str());
      printf("\n WAKE UP");

  }
  else if (warning > 2 && warning < 8){
    system(command1.c_str());
    printf("\n Can't find face");
  }
  }
}
void changetime(){ 
}
int main(int argc, char **argv)
{
  pthread_t thread1, thread2;
  int err; 
  start = 0; // Start of program 

    err = pthread_create(&thread1, NULL, trackingsystem, NULL);
    if (err != 0)
      printf("\ncan't create  Tracking System thread :[%s]", strerror(err));
    else
      printf("\n Thread created successfully\n");
    sleep(5);
    err = pthread_create(&thread2, NULL, alertsystem, NULL);
   if (err != 0)
      printf("\ncan't create  Alert System thread :[%s]", strerror(err));
    else
      printf("\n Thread created successfully\n");

pthread_join(thread1,NULL);
pthread_join(thread2,NULL);
 


  return 0; 
}

