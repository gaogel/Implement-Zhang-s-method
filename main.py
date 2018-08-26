#！anaconda3/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import glob

############################### I.preparation for calibration #####################################
#image I/O
nx,ny=(9,6) #interior number of corners
#prepare corners in 3D world coordinate & 2D pixel coordinate
objp = np.zeros((nx*ny,3),np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) 
objps = []
imgps = []
images = glob.glob('*.jpg')

############################### II.find corners ###############################################
def draw_corners(image, corners, ret=1) :
    image = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
    cv2.imshow('img', image)
    cv2.waitKey(500)
     
def find_corners(images) :
    for i in images :
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        image_size = (image.shape[1], image.shape[0])
        #find coners in pixel coordinate (imgps)
        ret,corners= cv2.findChessboardCorners(gray, (nx,ny), None,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                                               + cv2.CALIB_CB_FAST_CHECK)
        if ret is True :
            objps.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1),criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
            imgps.append(corners2)
            #draw_corners(image, corners2)
    cv2.destroyAllWindows()
    return image_size, objps, imgps

############################### III.calibration ################################################
def calibration(objps, imgps, image_size):
    retval,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(objps,imgps,image_size,None,None)
    return retval, mtx, dist, rvecs, tvecs

def show_intrinsic_matrix(mtx) :
    print(mtx)

############################### IV.undistortion #########################################
def undistortion(images, mtx, dist) :
    for i in images :
        #optimize camera matrix we have get before
        image = cv2.imread(i)
        image_size = (image.shape[1], image.shape[0])
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,image_size,1,image_size)
        #undistortion
        undst = cv2.undistort(image,mtx,dist,None,newcameramtx)
        #show_undistortion(undst)  
    cv2.destroyAllWindows()
    return undst

def show_undistortion(undst) :
    cv2.imshow('undistort exmaple',undst)
    cv2.waitKey(500)

############################### V.calibration via Zhang's method #############################
#Assuming Hq=Q, making homogeneous coordinates of q and Q
def get_homocoors(objps, imgps) :
    q1 = []
    for i in objps :
        i[:,2]=1
        for j in i :
            q1.append(j)
    q = np.array(q1)
    Q1 = []
    for i in imgps :
        for j in i :              
            a = np.ones((1,1),np.float32)        
            j = np.hstack((j,a)) 
            for k in j :
                Q1.append(k)
    Q = np.array(Q1)
    return q, Q

#implete Zhang's method
#get homogeneous matrix H and hi
def cal_v(i,j,H) :
        return np.array([H[0,i]*H[0,j],H[0,i]*H[1,j]+H[1,i]*H[0,j],H[1,i]*H[1,j],H[2,i]*H[0,j]+H[0,i]*H[2,j],H[2,i]*H[1,j]+H[1,i]*H[2,j],H[2,i]*H[2,j]])

def get_H_and_V(q, Q) :
    H=[]
    V=np.array([0,0,0,0,0,0])
    i=1
    while i <= len(objps) :
        q2 = q[54*(i-1):54*(i-1)+54,:]
        Q2 = Q[54*(i-1):54*(i-1)+54,:]
        H1, mask = cv2.findHomography(q2,Q2,cv2.RANSAC, 5.0)
        H.append(H1)
        i=i+1
    #get V
    H = np.array(H)
    for i in range (0,len(H)) :            
        u1 = cal_v(0,1,H[i])
        u2 = cal_v(0,0,H[i])-cal_v(1,1,H[i])
        #合并矩阵    
        v = np.concatenate([u1,u2])
        V = np.concatenate([V,v])
    V = V[6:len(V)].reshape(2*len(objps),6)
    return H, V

#using overdetermined equations to solve b
def get_b(V) :
    x, b = np.linalg.eig(V.T @ V)
    b = b[:, -1]
    return b

#solving intrinsic matrix 
def get_intrinsic_matrix(b) :
    d1 = b[1]*b[3]-b[0]*b[4]
    d2 = b[0]*b[2]-b[1]*b[1]
    cy = d1/d2
    ld = b[5]-(b[3]**2+cy*d1)/b[0]
    fx = (ld/b[0])**0.5
    fy = (ld*b[0]/d2)**0.5
    cx = -b[3]*fx**2/ld
    mtx=np.array([
		[fx, 0, cx],
		[0,   fy,  cy],
		[0,     0,      1]])
    return mtx

############################### Main ##########################################
image_size, objps, imgps = find_corners(images)
retval, mtx, dist, rvecs, tvecs = calibration(objps, imgps, image_size)
undistortion(images, mtx, dist)
#Zhang's method
q, Q = get_homocoors(objps, imgps)
H, V = get_H_and_V(q, Q)
b = get_b(V)
zhang_mtx = get_intrinsic_matrix(b)
