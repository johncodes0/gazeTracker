#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <stdio.h>
#include <iostream>


using namespace std;
using namespace cv;
int detectAndDisplay( Mat frame );
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";

// New code
std::vector<cv::Point> centers;
cv::Point lastPoint;
cv::Point mousePoint;

bool isInHypnosis(int timePassed, clock_t currentTime){

	if(timePassed > 30){
		currentTime = clock();
		return true;
	} 
	return false;	
}

void alertSystem(){};

Rect getLeftmostEye(std::vector<cv::Rect> &eyes)
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



// returns 1 if the face/eyes has moved, 0 otherwise
int detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    
    //-- Detect faces 
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(60, 60) );
    // If no face is detected, then reset timer
    if (faces.size() == 0){
    	return 1;
    }
    for ( size_t i = 0; i < faces.size(); i++ )
    {
    	// Creates a point object with the four corners of the face
    	// faces[i].x is the left part of the face and face[i].x+faces[i].width is the 
        Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
        // Creates an ellipse object with the ellipse to be displayed
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        
        Mat face = frame_gray( faces[i] );
        std::vector<Rect> eyes;
        //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( face, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        


		rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
 		if (eyes.size() != 2) return 0; // both eyes were not detected
  		for (cv::Rect &eye : eyes)
  		{
   		   rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
  		}
  		cv::Rect eyeRect = getLeftmostEye(eyes);
  		cv::Mat eye = face(eyeRect); // crop the leftmost eye
 		cv::equalizeHist(eye, eye);
  		std::vector<cv::Vec3f> circles;
  		cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
  		if (circles.size() > 0)
  		{
      	cv::Vec3f eyeball = getEyeball(eye, circles);
      	cv::Point center(eyeball[0], eyeball[1]);
      	centers.push_back(center);
      	center = stabilize(centers, 5);
      	if (centers.size() > 1)
      	{
          cv::Point diff;
          diff.x = (center.x - lastPoint.x) * 20;
          diff.y = (center.y - lastPoint.y) * -30;
          mousePoint += diff;
      	}
      	lastPoint = center;
      	int radius = (int)eyeball[2];
      	cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
      	cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
  	}
  	cv::imshow("Eye", eye);
}
        
        
       //  for ( size_t j = 0; j < eyes.size(); j++ )
//         {
//             Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
//             int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
//             circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
//         }
//     }
    //-- Show what you got
    imshow( window_name, frame );
    // Face did not move
    return 0;
}

void detectEyes(cv::Mat &frame, cv::CascadeClassifier &faceCascade, cv::CascadeClassifier &eyeCascade)
{
  cv::Mat grayscale;
  cv::cvtColor(frame, grayscale, CV_BGR2GRAY); // convert image to grayscale
  cv::equalizeHist(grayscale, grayscale); // enhance image contrast 
  std::vector<cv::Rect> faces;
  faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(150, 150));
  if (faces.size() == 0) return; // none face was detected
  cv::Mat face = grayscale(faces[0]); // crop the face
  std::vector<cv::Rect> eyes;
  eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30)); // same thing as above    
  rectangle(frame, faces[0].tl(), faces[0].br(), cv::Scalar(255, 0, 0), 2);
  if (eyes.size() != 2) return; // both eyes were not detected
  for (cv::Rect &eye : eyes)
  {
      rectangle(frame, faces[0].tl() + eye.tl(), faces[0].tl() + eye.br(), cv::Scalar(0, 255, 0), 2);
  }
  cv::Rect eyeRect = getLeftmostEye(eyes);
  cv::Mat eye = face(eyeRect); // crop the leftmost eye
  cv::equalizeHist(eye, eye);
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(eye, circles, CV_HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);
  if (circles.size() > 0)
  {
      cv::Vec3f eyeball = getEyeball(eye, circles);
      cv::Point center(eyeball[0], eyeball[1]);
      centers.push_back(center);
      center = stabilize(centers, 5);
      if (centers.size() > 1)
      {
          cv::Point diff;
          diff.x = (center.x - lastPoint.x) * 20;
          diff.y = (center.y - lastPoint.y) * -30;
          mousePoint += diff;
      }
      lastPoint = center;
      int radius = (int)eyeball[2];
      cv::circle(frame, faces[0].tl() + eyeRect.tl() + center, radius, cv::Scalar(0, 0, 255), 2);
      cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2);
  }
  cv::imshow("Eye", eye);
}

int main( int argc, const char** argv )
{
    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{face_cascade|./haarcascade_frontalface_alt.xml|}"
        "{eyes_cascade|./haarcascade_eye_tree_eyeglasses.xml|}");
    parser.about( "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
                  "You can use Haar or LBP features.\n\n" );
    parser.printMessage();
    face_cascade_name = parser.get<String>("face_cascade");
    eyes_cascade_name = parser.get<String>("eyes_cascade");
    VideoCapture capture;
    Mat frame;
    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };
    //-- 2. Read the video stream
    capture.open( 0 );
    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }
    
    // Timer 
    clock_t time1 = clock();
    double timePassed;
    
    // Continuously check frames from feed
    while ( capture.read(frame) )
    {
    detectEyes(frame, face_cascade, eyes_cascade);
    cv::imshow(window_name, frame); // displays the Mat
    if (cv::waitKey(30) >= 0) break;  // takes 30 frames per second. if the user presses any button, it stops from showing the webcam
    
    
        // if( frame.empty() )
//         {
//             printf(" --(!) No captured frame -- Break!");
//             break;
//         }
//         //-- 3. Apply the classifier to the frame
// //         printf("Displaying frame\n");
//         
//         // Display elapsed time in gaze
// 		timePassed = (clock() - time1)/(double)CLOCKS_PER_SEC;
// 		printf("time = %f\n",timePassed);
// 		
// 		// Condition to check if the driver is in hypnosis
// 		// Alert driver
// 		if(isInHypnosis(timePassed,time1)){
// 			alertSystem();
// 		}
// 		
// 		// Analayze frame.
// 		// If there is face/eye movement, reset the timer
// //         if (detectAndDisplay( frame )){
// //         	printf("Movement detected\n");
// //         	time1 = clock();
// //         }
// 
// 		
//         
//         if( waitKey(10) == 27 ) { break; } // escape
    }


    return 0;
}



