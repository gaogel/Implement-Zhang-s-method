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
        cv2.imwrite('undistorted', undst)
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
    x, b = np.linalg.eig(np.dot(V.T,V))
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

#solving extrinsic matrix
def get_extrinsic_matrix(h, mtx) :
    h1 = []
    h2 = []
    h3 = []
    for j in range(0,3) :
        h1.append(h[j][0])
        h2.append(h[j][1])
        h3.append(h[j][2])
    h1 = np.array(h1)
    h1 = h1.reshape(len(h1),1)
    h2 = np.array(h2)
    h2 = h2.reshape(len(h2),1)
    h3 = np.array(h3)
    h3 = h3.reshape(len(h3),1)
    ld = 1/np.linalg.norm(np.dot(np.linalg.inv(mtx),h1))
    r1 = ld * np.dot(np.linalg.inv(mtx),h1)
    
    n_r1 = []
    n_r1.append(r1[0][0])
    n_r1.append(r1[1][0])
    n_r1.append(r1[2][0])
    n_r1.append(0)
    n_r1 = np.array(n_r1).reshape(4,1)
    
    r2 = ld * np.dot(np.linalg.inv(mtx),h2)
    n_r2 = []
    n_r2.append(r2[0][0])
    n_r2.append(r2[1][0])
    n_r2.append(r2[2][0])
    n_r2.append(0)
    n_r2 = np.array(n_r2).reshape(4,1)
    
    r3 = np.cross(r1.T,r2.T)
    n_r3 = []
    n_r3.append(r3[0][0])
    n_r3.append(r3[0][1])
    n_r3.append(r3[0][2])
    n_r3.append(0)
    n_r3 = np.array(n_r3).reshape(4,1)
    
    t = ld * np.dot(np.linalg.inv(mtx),h3)
    n_t = []
    n_t.append(t[0][0])
    n_t.append(t[1][0])
    n_t.append(t[2][0])
    n_t.append(1)
    n_t = np.array(n_t).reshape(4,1)
    
 
    E = np.array([n_r1, n_r2, n_r3, n_t])

    return E

#Undistort via Zhang's method
def get_D_and_d(intr_mtx, H, objps, imgps) :
    D=[]
    d=[]
    #get u0 and v0
    u0 = intr_mtx[0,2]
    v0 = intr_mtx[1,2]
    for i in range(0,len(objps)) :
        for j in range(0,len(objps[0])) : 
            #prepration for u, v, x, y
            homo_objp = np.array([objps[i][j][0],objps[i][j][1],0,1])
            E = get_extrinsic_matrix(H[i], intr_mtx)
            homo_nondist_coords = np.dot(E.T, homo_objp.T)
            #nondist_coords = homo_nondist_coords/homo_nondist_coords[0][-2]
            x,y = homo_nondist_coords[0][0], homo_nondist_coords[0][1]
            homo_nondist_coords = np.array([x, y, 1])
            homo_nondist_pixelcoords = np.dot(intr_mtx, homo_nondist_coords.T)
            [u,v,others] = homo_nondist_pixelcoords
            #calculate D
            l1 = [(u - u0)*(x**2 + y**2),(u - u0)*(x**2 + y**2)**2]
            D.append(l1)
            d.append(imgps[i][j][0][0]-u)
            l2 = [(v - v0)*(x**2 + x**2),(v - v0)*(x**2 + y**2)**2]
            D.append(l2)
            d.append(imgps[i][j][0][1]-v)
    D = np.array(D)
    d = np.array(d)
    return D, d
    
 
def get_K(D, d) :
    return np.dot(np.dot(np.linalg.inv(np.dot(D.T,D)),D.T),d)

############################### Main ##########################################
image_size, objps, imgps = find_corners(images)
retval, mtx, dist, rvecs, tvecs = calibration(objps, imgps, image_size)
undistortion(images, mtx, dist)
#calibrate via Zhang's method
q, Q = get_homocoors(objps, imgps)
H, V = get_H_and_V(q, Q)
b = get_b(V)
zhang_mtx = get_intrinsic_matrix(b)
E= []
for i in range (0,len(H)) :
    E.append(get_extrinsic_matrix(H[i], zhang_mtx))
#undistortion via Zhang's method
D, d = get_D_and_d(zhang_mtx, H, objps, imgps)
K = get_K(D, d)
#write results
'''
with open('/Users/lg/Downloads/python项目/flask/file1/Code/calibration results.txt','w') as f :
    f.write('intrinsic matrix via OpenCv function:\n') 
    f.write(str(mtx)+'\n')
    f.write('extrinsics :\n') 
    f.write('rvecs:\n')
    f.write(str(rvecs)+'\n')
    f.write('tvecs:\n')
    f.write(str(tvecs)+'\n')
    f.write('distortion coefficients:\n')
    f.write(str(dist))
with open("/Users/lg/Downloads/python项目/flask/file1/Code/Zhang's method results.txt",'w') as f :
    f.write("intrinsic matrix via Zhang's method:\n") 
    f.write(str(zhang_mtx)+'\n')
    f.write('extrinsics :\n') 
    f.write(str(E)+'\n')
    f.write('distortion coefficients(k1,k2):\n')
    f.write(str(K))
'''
