import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

position = np.zeros((1,3))
old_pos  = np.zeros((1,3))
new_origin = [0,0,0]
rotation = np.eye(3)
plot = np.zeros([500,500],np.uint8)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

Rx = [[1,0,0],[0,1,0],[0,0,1]]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# camera parameters

def draw_rect(K, d, train_frame, R, T, name):
	#perform the rectification
	R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, d, K, d, train_frame.shape[:2], R, T, alpha=1)
	mapx1, mapy1 = cv2.initUndistortRectifyMap(K, d, R1, K, train_frame.shape[:2], cv2.CV_32F)
	mapx2, mapy2 = cv2.initUndistortRectifyMap(K, d, R2, K, query_frame.shape[:2], cv2.CV_32F)
	img_rect1 = cv2.remap(train_bckp, mapx1, mapy1, cv2.INTER_LINEAR)
	img_rect2 = cv2.remap(query_bckp, mapx2, mapy2, cv2.INTER_LINEAR)

	# draw the images side by side
	total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1],3)
	img = np.zeros(total_size, dtype=np.uint8)
	img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
	img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
	 
	# draw horizontal lines every 25 px accross the side by side image
	for i in range(20, img.shape[0], 25):
		cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
	

	h1, w1 = train_frame.shape[:2]
	h2, w2 = query_frame.shape[:2]
	org_imgs = np.zeros((max(h1, h2), w1+w2,3), np.uint8)
	org_imgs[:h1, :w1] = train_bckp
	org_imgs[:h2, w1:w1+w2] = query_bckp
	for m in good:
		# draw the keypoints
		# print m.queryIdx, m.trainIdx, m.distance
		color = tuple([np.random.randint(0, 255) for _ in xrange(3)])
		cv2.line(org_imgs, (int(train_keypoints[m.queryIdx].pt[0]), int(train_keypoints[m.queryIdx].pt[1])) , (int(query_keypoints[m.trainIdx].pt[0] + w1), int(query_keypoints[m.trainIdx].pt[1])), color)
		cv2.circle(org_imgs, (int(train_keypoints[m.queryIdx].pt[0]), int(train_keypoints[m.queryIdx].pt[1])) , 5, color, 1)
		cv2.circle(org_imgs, (int(query_keypoints[m.trainIdx].pt[0] + w1), int(query_keypoints[m.trainIdx].pt[1])) , 5, color, 1)
	cv2.imshow('original', org_imgs)
	cv2.imshow(name, img)

def in_front_of_both_cameras(first_points, second_points, rot, trans):
    # check if the point correspondences are in front of both images
    false_count = 0
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0]*rot[2, :], trans) / np.dot(rot[0, :] - second[0]*rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)
 
        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            false_count = false_count + 1
    if false_count ==0:
        return True, false_count
    else:
        return False, false_count

#Get Video File Name
videoFile = sys.argv[1]
print "Reading from Video:" + videoFile


image_downsize_const = 0.25 #Image downsize constant
offset_cnt = 10 #Offset frame count
REJECT_THRESH = 8
NUM_NEIGHBOUR = 2
#Initialize Matcher
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
#matcher = cv2.FlannBasedMatcher(index_params, search_params)
matcher = cv2.BFMatcher()

#Feature Detector
detector = cv2.SURF(500)#cv2.ORB()#cv2.SURF(500) ##cv2.SIFT()#

#Camera Calibration Constants
#Distortion
d = np.array([5.00139920e-03, 5.11738265e-01, -3.86138231e-03, 6.68182847e-04,  -1.50677281e+00, 0.0, 0.0, 0.0]).reshape(1, 8) # distortion coefficients
#Camera Instrinsics
K = np.array([415.0883213, 0.0, 239.3843051, 0.0, 416.37626494, 134.04803759, 0.0, 0.0, 1.0]).reshape(3, 3) # Camera matrix

K_inv = np.linalg.inv(K)

#Initialize Video Capture
cap = cv2.VideoCapture(videoFile)
ret, train_frame = cap.read()
init = True
backup_frame = train_frame
init_flag = True
while True:
    if init:
        ret, train_frame = cap.read()        
        init = False
    else:
        train_frame = backup_frame

    if train_frame == None:
        print "The File is fucked bro"
        sys.exit(2)
    for i in range(0,offset_cnt):
        cap.read()        
    ret, query_frame = cap.read()
    backup_frame = np.copy(query_frame)
    if query_frame==None:
        print "End of Video"
        sys.exit(1)
    train_frame = cv2.resize(train_frame, (0,0), fx=image_downsize_const, fy=image_downsize_const)
    query_frame = cv2.resize(query_frame, (0,0), fx=image_downsize_const, fy=image_downsize_const)
    train_bckp = train_frame
    query_bckp = query_frame
   #Get Grayscale images
    train_frame = cv2.cvtColor(train_frame,cv2.COLOR_BGR2GRAY) 
    query_frame = cv2.cvtColor(query_frame,cv2.COLOR_BGR2GRAY)
    #Undistort Images
    train_undistort = cv2.undistort(train_frame, K, d)
    query_undistort = cv2.undistort(query_frame, K, d)

    #Get Descriptors
    train_keypoints, train_descrip = detector.detectAndCompute(train_undistort, None)
    query_keypoints, query_descrip = detector.detectAndCompute(query_undistort, None)
    
    #Match
    matches = matcher.knnMatch(np.asarray(train_descrip,np.float32),np.asarray(query_descrip,np.float32), NUM_NEIGHBOUR)
    #Get Good matches
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    #Decide if matches are good enough
    if len(good) < REJECT_THRESH:
        print("Not enough good matches")
        continue

    train_match_points = np.float32([ train_keypoints[m.queryIdx].pt for m in good ])#.reshape(-1,1,2)
    query_match_points = np.float32([ query_keypoints[m.trainIdx].pt for m in good ])#.reshape(-1,1,2)
    # estimate fundamental matrix
    F, mask = cv2.findFundamentalMat(train_match_points, query_match_points, cv2.FM_RANSAC, 0.1, 0.99)
	 
    # decompose into the essential matrix
    E = K.T.dot(F).dot(K)
	 
    # decompose essential matrix into R, t (See Hartley and Zisserman 9.13)
    U, S, Vt = np.linalg.svd(E)
    W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
   # iterate over all point correspondences used in the estimation of the fundamental matrix
    first_inliers = []
    second_inliers = []
    for i in range(len(mask)):
   	if mask[i]:
		# normalize and homogenize the image coordinates
		first_inliers.append(K_inv.dot([train_match_points[i][0], train_match_points[i][1], 1.0]))
		second_inliers.append(K_inv.dot([query_match_points[i][0], query_match_points[i][1], 1.0]))
	 
    # Determine the correct choice of second camera matrix
    # only in one of the four configurations will all the points be in front of both cameras
    # First choice: R = U * Wt * Vt, T = +u_3 (See Hartley Zisserman 9.19)
    sol = 1
    R = U.dot(W).dot(Vt)
    T = U[:, 2]
    false_counts = []
    in_front, count = in_front_of_both_cameras(first_inliers, second_inliers, R, T)
    false_counts.append(count)
    if not in_front:	 
	# Second choice: R = U * W * Vt, T = -u_3
	T = - U[:, 2]
	sol = 2
        in_front, count = in_front_of_both_cameras(first_inliers, second_inliers, R, T)
        false_counts.append(count)
        if not in_front:
	   # Third choice: R = U * Wt * Vt, T = u_3
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]	 
            sol = 3
            in_front, count = in_front_of_both_cameras(first_inliers, second_inliers, R, T)
            false_counts.append(count)
            if not in_front: 
	   # Fourth choice: R = U * Wt * Vt, T = -u_3
	       T = - U[:, 2]
	       sol = 4
               in_front, count = in_front_of_both_cameras(first_inliers, second_inliers, R, T)
               false_counts.append(count)
    if len(false_counts)==4:
        print false_counts
        print "No valid Soln"
        #continue
        min_count =  np.argmin(false_counts)
        if(min_count==0):
            R = U.dot(W).dot(Vt)
            T = U[:, 2]
            sol =1
        elif min_count==1:
            R = U.dot(W).dot(Vt)
            T = -U[:, 2]
        if(min_count==2):
            R = U.dot(W.T).dot(Vt)
            T = U[:, 2]
            sol =3
        elif min_count==3:
            R = U.dot(W.T).dot(Vt)
            T = -U[:, 2]
            sol=4

    print "SOL NO:" + str(sol)
    print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
    #print R
    #print T

    #print "*******************************************"
    draw_rect(K, d, train_frame, R, T, 'rectified')
    camera_trans_wrt_train_frame = -np.dot(R.T,T)
    new_origin = new_origin + np.dot(R,T)
    rotation = np.matrix(rotation)*(np.matrix(R))
    rotation = rotation/np.amax(abs(rotation))
    #print "Translation:" + str(T)
    posMat = (-((np.matrix(rotation)))*(np.matrix(T).T))
    #print posMat.T
    position +=  new_origin + camera_trans_wrt_train_frame#posMat.T

    #position = position[:,0]
    #print position
    #position += np.dot(R.T,T)
	#position +=T
	#print position
	#print old_pos
    ax.scatter(position[0][0], position[0][1], position[0][2], c='r', marker='o')
    ax.plot([position[0][0],old_pos[0][0]],[position[0][1],old_pos[0][1]],zs=[position[0][2],old_pos[0][2]],color='g')
    #Draw pose axis
    x_offset = np.matrix([0.4,0,0])
    y_offset = np.matrix([0,0.4,0])
    z_offset = np.matrix([0,0,0.4])
    x_offset = rotation*x_offset.T
    x_offset = np.array(x_offset)
    y_offset = rotation*y_offset.T
    y_offset = np.array(y_offset)
    z_offset = rotation*z_offset.T
    z_offset = np.array(z_offset)
    ax.plot([position[0][0],position[0][0] + x_offset[0][0]],[position[0][1],position[0][1] + x_offset[1][0]],zs=[position[0][2],position[0][2] + x_offset[2][0]],color='r')    
    ax.plot([position[0][0],position[0][0] + y_offset[0][0]],[position[0][1],position[0][1] + y_offset[1][0]],zs=[position[0][2],position[0][2] + y_offset[2][0]],color='y')    
    ax.plot([position[0][0],position[0][0] + z_offset[0][0]],[position[0][1],position[0][1] + z_offset[1][0]],zs=[position[0][2],position[0][2] + z_offset[2][0]],color='b')    
    
    old_pos = np.copy(position)
    plt.draw()
    plt.show(block=False)
    train_frame =np.copy(query_frame)#backup_frame
    if(init_flag == True):        
        kk = cv2.waitKey(0)
        if kk==ord('c'):
            init_flag=False
    #if kk==ord('c'):
    #    continue   
    #elif (kk==ord('q')):
    #    cv2.destroyAllWindows()
    #    sys.exit(2)
    #    

plt.show(block=True)


