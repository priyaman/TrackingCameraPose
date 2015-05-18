# TrackingCameraPose
Estimates the trajectory of the camera using only a video input stream.
Prerequities: OpenCV 3, Python 2.7
Given a video as input from a calibrated camera, 
the python script calculates the pose of the camera and plots it in a Axes3D GUI.
Takes 2 closely spaced frames form the video sequence. Estimates the fundamental matrix between the 2 images.
Uses the camera calibration details to extract the essential matrix 
Which is broken down into a rotation and translation matrix.
Cumulates the rotation and translation from each subsequent frame to generate the trajectory.

