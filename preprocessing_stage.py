# Import necessary components for face detection
import numpy as np
import pandas as pd
import cv2
import dlib
import os
import fnmatch
###############################################################
#Functions
###############################################################
#Face detection parameters
PREDICTOR_PATH = '.../shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='.../haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
Indicesface=68
##############################################################################
##############################################################################
#Function to obtain landmarks:
def get_landmarks(im):
    #(x_s,y_s)=im.shape
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im, dtype='uint8')
    faces = cascade.detectMultiScale(im, 1.15,  4, 0, (100, 100))
    if (faces==()):
        return np.matrix([[0 for row in range(0,2)] for col in range(Indicesface)])        
    else:
        for (x,y,w,h) in faces:
            rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        #return np.array([[p.x, p.y] for p in predictor(gray, rect).parts()],dtype=np.float32)
        return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
##############################################################################
##############################################################################
#Function to extract face from image:
def crop_face(im):
    #(x_s,y_s)=im.shape
    #gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(im, 1.15,  4, 0, (100, 100))
    if (faces==()):
        l=np.matrix([[0 for row in range(0,2)] for col in range(Indicesface)])
        rect=dlib.rectangle(0,0,0,0)
        return np.empty(im.shape)*0,l
    else:
        for (x,y,w,h) in faces:
            rect=dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
            l=np.array([[p.x, p.y] for p in predictor(im, rect).parts()],dtype=np.float32)
            sub_face = im[y:y+h, x:x+w]
            #cv2.imshow('Result',sub_face)
        return sub_face ,l 
##############################################################################
##############################################################################
#Function to annotate landmarks on the face:
def annotate_landmarks(im, landmarks):
    #create a copy from the image
    img = im.copy()
    #create vectors to represent each landmark
    #print(landmarks)
    if (landmarks.all()==0):
        return im
    else:
        for idx, point in enumerate(landmarks):
            #print('índice e posição do ponto',idx,point)
            pos = (point[0, 0], point[0, 1])
            #Mark with circles the points found before
            cv2.circle(img, pos, 2, color=(255, 255, 255), thickness=-1)
        return img
##############################################################################
##############################################################################
#Function to create set of data from SILFA dataset videos
def read_img_silfa_mid_crop(im_s,location):
#setting parameters
    x_upper = [] #set array of testing data
    x_lower=[]
    y_upper = [] #set array of testing label
    y_lower=[]
    label_u = [] #name of label in folder will be stored
    label_l=[]
    t=0
    data = pd.read_csv(location, engine='python', sep = ',', skiprows=1)
    data_matrix = data.values
    data_matrix[:,2:4].astype(int)
    V_s=data_matrix[:,1]
    W=np.empty((int(len(data_matrix)),1))*0
    points_u=np.empty((21,2))*0
    points_l=np.empty((32,2))*0

    for s in range(0,len(V_s)):
            t_inicial=data_matrix[s,2]*0.001
            t_final=data_matrix[s,3]*0.001
            t_inicial=np.array(t_inicial)
            t_final=np.array(t_final)
            print(V_s[s])
            #getting the video from SILFA dataset: 
            cam = cv2.VideoCapture('.../SILFA/videos/%s.mp4' %(V_s[s]),0)
            Frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cam.get(cv2.CAP_PROP_FPS)
            print('Number of Frames=', Frames )
            #getting time and labels information: 
            t_i=np.empty(1)
            t_f=np.empty(1)
            for i in range(t_inicial.size):
                if i==0:
                    t_i[i]=int(t_inicial*fps)
                    t_f[i]=int(t_final*fps)
                else:
                    t_i[i]=int(t_inicial[i]*fps)
                    if t_f[i-1]==t_i[i]:
                        t_i[i]=int(t_inicial[i]*fps)+1
                        t_f[i]=int(t_final[i]*fps)
                    else:
                        t_i[i]=int(t_inicial[i]*fps)
                        t_f[i]=int(t_final[i]*fps)
            u=np.empty(int(len(t_i)))*0            
            for i in range(len(t_i)):
                u[i]=t_f[i]-t_i[i]
            w=np.empty(int(len(t_i)+sum(u)))*0
            for i in range(len(t_i)):
                if i==0:
                    v0=t_i[i]+range(0,int(u[i]+1))
                    w[0:int(len(v0))]=v0
                else:
                    v0=t_i[i-1]+range(0,int(u[i-1]+1))
                    v=t_i[i]+range(0,int(u[i]+1))
                    j=int(sum(u[0:i])+i)
                    k=j+int(u[i])+1
                    w[j:k]=v
            W[t,0]=len(w)
            t+=1
    count=0
    t=0
    for s in range(0,len(V_s)):
            t_inicial=data_matrix[s,2]*0.001
            t_final=data_matrix[s,3]*0.001
            t_inicial=np.array(t_inicial)
            t_final=np.array(t_final)
            print(V_s[s])
            cam = cv2.VideoCapture('.../SILFA/videos/%s.mp4' %(V_s[s]),0)
            Frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cam.get(cv2.CAP_PROP_FPS)
            print('Number of Frames', Frames )
            t_i=np.empty(int(t_inicial.size))
            t_f=np.empty(int(t_final.size))
            for i in range(t_inicial.size):
                if i==0:
                    t_i[i]=int(t_inicial*fps)
                    t_f[i]=int(t_final*fps)
                else:
                    t_i[i]=int(t_inicial[i]*fps)
                    if t_f[i-1]==t_i[i]:
                        t_i[i]=int(t_inicial[i]*fps)+1
                        t_f[i]=int(t_final[i]*fps)
                    else:
                        t_i[i]=int(t_inicial[i]*fps)
                        t_f[i]=int(t_final[i]*fps)
                u=np.empty(int(len(t_i)))*0            
            for i in range(len(t_i)):
                u[i]=t_f[i]-t_i[i]
            w=np.empty(int(len(t_i)+sum(u)))*0

            DT2=np.empty((int(len(t_i)+sum(u))))*0
            DT3=np.empty((int(len(t_i)+sum(u))))*0
            for i in range(len(t_i)):
                if i==0:
                    v0=t_i[i]+range(0,int(u[i]+1))
                    w[0:int(len(v0))]=v0
                    DT2[0:int(len(v0))]=data_matrix[s,5]
                    DT3[0:int(len(v0))]=data_matrix[s,6]
                else:
                    v0=t_i[i-1]+range(0,int(u[i-1]+1))
                    v=t_i[i]+range(0,int(u[i]+1))
                    j=int(sum(u[0:i])+i)
                    k=j+int(u[i])+1
                    w[j:k]=v
                    DT2[j:k]=data_matrix[s,5]
                    DT3[j:k]=data_matrix[s,6]

            label_u=np.append(label_u,[DT2])
            label_l=np.append(label_l,[DT3])
            t+=1
            for i, j in enumerate(w):
                print(i,int(j))
                cam.set(1, int(j))
                ret, im = cam.read()
                if ret is True:
                    a,l = crop_face(im)
                    c=get_landmarks(a)
                    #getting points and distance information:
                    points_u[:9,:]=c[17:26,:]
                    points_u[10:,:]=c[36:47,:]
                    vp=np.stack((points_u))
                    points_l[:12,:]=c[2:14,:]
                    points_l[13:,:]=c[48:67,:]
                    vb=np.stack((points_l))
                    vs_brown_e=np.squeeze(np.asarray(c[19]-c[17]))
                    vi_brown_e=np.squeeze(np.asarray(c[21]-c[17]))
                    vs_brown_d=np.squeeze(np.asarray(c[24]-c[26]))
                    vi_brown_d=np.squeeze(np.asarray(c[22]-c[26]))
                    a_brown_e=np.arccos(np.dot(vs_brown_e,vi_brown_e,out=None)/(np.linalg.norm(vs_brown_e)*np.linalg.norm(vi_brown_e)))
                    a_brown_d=np.arccos(np.dot(vs_brown_d,vi_brown_d,out=None)/(np.linalg.norm(vs_brown_d)*np.linalg.norm(vi_brown_d)))
                    v1_eye_e=np.squeeze(np.asarray(c[37]-c[41]))
                    v2_eye_e=np.squeeze(np.asarray(c[38]-c[40]))
                    v1_eye_d=np.squeeze(np.asarray(c[43]-c[47]))
                    v2_eye_d=np.squeeze(np.asarray(c[44]-c[46]))
                    vs=np.stack((vs_brown_e,vi_brown_e,vs_brown_d,vi_brown_d,v1_eye_e,v2_eye_e,v1_eye_d,v2_eye_d))
                    d_lips_h1=np.squeeze(np.asarray(c[48]-c[54]))
                    d_lips_h2=np.squeeze(np.asarray(c[60]-c[64]))
                    d_lips_v1=np.squeeze(np.asarray(c[51]-c[57]))
                    d_lips_v2=np.squeeze(np.asarray(c[62]-c[66]))
                    vl=np.stack((d_lips_h1,d_lips_h2,d_lips_v1,d_lips_v2))
                    p_u=[vp.tolist(), vs.tolist()]
                    points_upper=np.hstack([np.hstack(np.vstack(p_u)),a_brown_e,a_brown_d])
                    p_l=[vb.tolist(), vl.tolist()]
                    #print(np.hstack(np.vstack(p_l)).shape)
                    zer_0=np.empty(28)*0
                    points_lower=np.append(np.hstack(np.vstack(p_l)),zer_0).reshape((50,2))
                    r = cv2.resize(a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC)
                    r = r[:,:,1]
                    upper = np.array(r[:60,:])
                    lower = np.array(r[46:,:])
                    im_u = np.vstack((upper.T,points_upper))  
                    im_u = im_u.astype('float32')
                    im_u /= 255
                    im_l = np.vstack((lower.T,points_lower[:,0],points_lower[:,1]))
                    im_l = im_l.astype('float32')
                    im_l /= 255
                    x_upper=np.append(x_upper,[np.array(im_u.T)])
                    x_lower=np.append(x_lower,[np.array(im_l.T)])
                else:
                    continue
                
    x_upper=np.array(x_upper)
    x_lower=np.array(x_lower)
    y_upper=np.array(label_u)
    y_lower=np.array(label_l)   
    print('count=',count)
    count+=1
    return x_upper,x_lower,y_upper,y_lower
##############################################################################
##############################################################################
#Function to create set of data from HM dataset videos
def read_img_HM_mid_crop(im_s,location):
#set parameters:
    x_upper = [] 
    x_lower=[]
    y_upper = [] 
    y_lower=[]
    label_u = [] 
    label_l=[]
    t=0
    data = pd.read_csv(location, engine='python', sep = ',', skiprows=1)
    data_matrix = data.values
    data_matrix[:,2:4].astype(int)
    V_s=data_matrix[:,1]
    W=np.empty((int(len(data_matrix)),1))*0
    points_u=np.empty((21,2))*0
    points_l=np.empty((32,2))*0
    for s in range(0,len(V_s)):
        t_inicial=data_matrix[s,2]*0.001
        t_final=data_matrix[s,3]*0.001
        t_inicial=np.array(t_inicial)
        t_final=np.array(t_final)
        cam = cv2.VideoCapture('.../HMdataset/%s.mp4' %(V_s[s]),0)
        Frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print('Number of Frames=', Frames )
        t_i=np.empty(1)
        t_f=np.empty(1)
        for i in range(t_inicial.size):
            if i==0:
                t_i[i]=int(t_inicial*fps)
                t_f[i]=int(t_final*fps)
            else:
                t_i[i]=int(t_inicial[i]*fps)
                if t_f[i-1]==t_i[i]:
                    t_i[i]=int(t_inicial[i]*fps)+1
                    t_f[i]=int(t_final[i]*fps)
                else:
                    t_i[i]=int(t_inicial[i]*fps)
                    t_f[i]=int(t_final[i]*fps)
        u=np.empty(int(len(t_i)))*0            
        for i in range(len(t_i)):
            u[i]=t_f[i]-t_i[i]
        w=np.empty(int(len(t_i)+sum(u)))*0
        for i in range(len(t_i)):
            if i==0:
                v0=t_i[i]+range(0,int(u[i]+1))
                w[0:int(len(v0))]=v0
            else:
                v0=t_i[i-1]+range(0,int(u[i-1]+1))
                v=t_i[i]+range(0,int(u[i]+1))
                j=int(sum(u[0:i])+i)
                k=j+int(u[i])+1
                w[j:k]=v
        W[t,0]=len(w)
        t+=1
        
    #label=np.empty((int(sum(W))))*0
    #label.tolist()       
    count=0
    t=0
    for s in range(0,len(V_s)):
        t_inicial=data_matrix[s,2]*0.001
        t_final=data_matrix[s,3]*0.001
        t_inicial=np.array(t_inicial)
        t_final=np.array(t_final)
        cam = cv2.VideoCapture('.../HMdataset/%s.mp4' %(V_s[s]),0)
        Frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print('Number of Frames', Frames )
        t_i=np.empty(int(t_inicial.size))
        t_f=np.empty(int(t_final.size))
        for i in range(t_inicial.size):
            if i==0:
                t_i[i]=int(t_inicial*fps)
                t_f[i]=int(t_final*fps)
            else:
                t_i[i]=int(t_inicial[i]*fps)
                if t_f[i-1]==t_i[i]:
                    t_i[i]=int(t_inicial[i]*fps)+1
                    t_f[i]=int(t_final[i]*fps)
                else:
                    t_i[i]=int(t_inicial[i]*fps)
                    t_f[i]=int(t_final[i]*fps)
        u=np.empty(int(len(t_i)))*0            
        for i in range(len(t_i)):
            u[i]=t_f[i]-t_i[i]
        w=np.empty(int(len(t_i)+sum(u)))*0
        DT2=np.empty((int(len(t_i)+sum(u))))*0
        DT3=np.empty((int(len(t_i)+sum(u))))*0
        for i in range(len(t_i)):
            if i==0:
                v0=t_i[i]+range(0,int(u[i]+1))
                w[0:int(len(v0))]=v0
                #DT2[0:int(len(v0))]=data_matrix[s,7]
                #DT3[0:int(len(v0))]=data_matrix[s,8]
                DT2[0:int(len(v0))]=data_matrix[s,9]
                DT3[0:int(len(v0))]=data_matrix[s,10]
            else:
                v0=t_i[i-1]+range(0,int(u[i-1]+1))
                v=t_i[i]+range(0,int(u[i]+1))
                j=int(sum(u[0:i])+i)
                k=j+int(u[i])+1
                w[j:k]=v
                DT2[j:k]=data_matrix[s,9]
                DT3[j:k]=data_matrix[s,10]

        label_u=np.append(label_u,[DT2])
        label_l=np.append(label_l,[DT3])
        t+=1
        for i, j in enumerate(w):
            print(i,int(j))
            cam.set(1, int(j))
            ret, im = cam.read()
            if ret is True:
                a,l = crop_face(im)
                c=get_landmarks(a)
                points_u[:9,:]=c[17:26,:]
                points_u[10:,:]=c[36:47,:]
                vp=np.stack((points_u))
                points_l[:12,:]=c[2:14,:]
                points_l[13:,:]=c[48:67,:]
                vb=np.stack((points_l))
                vs_brown_e=np.squeeze(np.asarray(c[19]-c[17]))
                vi_brown_e=np.squeeze(np.asarray(c[21]-c[17]))
                vs_brown_d=np.squeeze(np.asarray(c[24]-c[26]))
                vi_brown_d=np.squeeze(np.asarray(c[22]-c[26]))
                a_brown_e=np.arccos(np.dot(vs_brown_e,vi_brown_e,out=None)/(np.linalg.norm(vs_brown_e)*np.linalg.norm(vi_brown_e)))
                a_brown_d=np.arccos(np.dot(vs_brown_d,vi_brown_d,out=None)/(np.linalg.norm(vs_brown_d)*np.linalg.norm(vi_brown_d)))
                v1_eye_e=np.squeeze(np.asarray(c[37]-c[41]))
                v2_eye_e=np.squeeze(np.asarray(c[38]-c[40]))
                v1_eye_d=np.squeeze(np.asarray(c[43]-c[47]))
                v2_eye_d=np.squeeze(np.asarray(c[44]-c[46]))
                vs=np.stack((vs_brown_e,vi_brown_e,vs_brown_d,vi_brown_d,v1_eye_e,v2_eye_e,v1_eye_d,v2_eye_d))
                d_lips_h1=np.squeeze(np.asarray(c[48]-c[54]))
                d_lips_h2=np.squeeze(np.asarray(c[60]-c[64]))
                d_lips_v1=np.squeeze(np.asarray(c[51]-c[57]))
                d_lips_v2=np.squeeze(np.asarray(c[62]-c[66]))
                vl=np.stack((d_lips_h1,d_lips_h2,d_lips_v1,d_lips_v2))
                p_u=[vp.tolist(), vs.tolist()]
                points_upper=np.hstack([np.hstack(np.vstack(p_u)),a_brown_e,a_brown_d])
                p_l=[vb.tolist(), vl.tolist()]
                zer_0=np.empty(28)*0
                points_lower=np.append(np.hstack(np.vstack(p_l)),zer_0).reshape((50,2))
                r = cv2.resize(a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC)
                r = r[:,:,1]
                upper = np.array(r[:60,:])
                lower = np.array(r[46:,:])
                im_u = np.vstack((upper.T,points_upper))  
                im_u = im_u.astype('float32')
                im_u /= 255
                im_l = np.vstack((lower.T,points_lower[:,0],points_lower[:,1]))
                im_l = im_l.astype('float32')
                im_l /= 255
                x_upper=np.append(x_upper,[np.array(im_u.T)])
                x_lower=np.append(x_lower,[np.array(im_l.T)])
            else:
                continue
                
    x_upper=np.array(x_upper)
    x_lower=np.array(x_lower)
    y_upper=np.array(label_u)
    y_lower=np.array(label_l)
    print('count=',count)
    count+=1
    return x_upper, x_lower, y_upper, y_lower
##############################################################################
##############################################################################
#Function to create set of data from DISFA dataset videos
def read_img_DISFA_mid_crop(im_s,name):
    x_upper=[]
    x_lower=[]
    y_upper=[]
    y_lower=[]
    count=0
    for s in name:
        points_u=np.empty((21,2))*0
        points_l=np.empty((32,2))*0
        data = pd.read_csv('.../DISFA/ActionUnit_Labels/SN0%d.csv'%(s), engine='python',sep = ',', skiprows=1)
        data_matrix = data.values
        data_matrix[:,1].astype(int)
        w=data_matrix[:,1]
        cam = cv2.VideoCapture('.../DISFA/Video_RightCamera/RightVideoSN0%d_Comp.avi' %(s),0)
        Frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cam.get(cv2.CAP_PROP_FPS)
        print('Number of Frames', Frames )
        for i, j in enumerate(w):
            print((i,j))
            cam.set(1, int(j))
            ret, im = cam.read()
            if ret is True:
                a,l = crop_face(im)
                c=get_landmarks(a)
                points_u[:9,:]=c[17:26,:]
                points_u[10:,:]=c[36:47,:]
                vp=np.stack((points_u))
                points_l[:12,:]=c[2:14,:]
                points_l[13:,:]=c[48:67,:]
                vb=np.stack((points_l))
                vs_brown_e=np.squeeze(np.asarray(c[19]-c[17]))
                vi_brown_e=np.squeeze(np.asarray(c[21]-c[17]))
                vs_brown_d=np.squeeze(np.asarray(c[24]-c[26]))
                vi_brown_d=np.squeeze(np.asarray(c[22]-c[26]))
                a_brown_e=np.arccos(np.dot(vs_brown_e,vi_brown_e,out=None)/(np.linalg.norm(vs_brown_e)*np.linalg.norm(vi_brown_e)))
                a_brown_d=np.arccos(np.dot(vs_brown_d,vi_brown_d,out=None)/(np.linalg.norm(vs_brown_d)*np.linalg.norm(vi_brown_d)))
                v1_eye_e=np.squeeze(np.asarray(c[37]-c[41]))
                v2_eye_e=np.squeeze(np.asarray(c[38]-c[40]))
                v1_eye_d=np.squeeze(np.asarray(c[43]-c[47]))
                v2_eye_d=np.squeeze(np.asarray(c[44]-c[46]))
                vs=np.stack((vs_brown_e,vi_brown_e,vs_brown_d,vi_brown_d,v1_eye_e,v2_eye_e,v1_eye_d,v2_eye_d))
                d_lips_h1=np.squeeze(np.asarray(c[48]-c[54]))
                d_lips_h2=np.squeeze(np.asarray(c[60]-c[64]))
                d_lips_v1=np.squeeze(np.asarray(c[51]-c[57]))
                d_lips_v2=np.squeeze(np.asarray(c[62]-c[66]))
                vl=np.stack((d_lips_h1,d_lips_h2,d_lips_v1,d_lips_v2))
                p_u=[vp.tolist(), vs.tolist()]
                points_upper=np.hstack([np.hstack(np.vstack(p_u)),a_brown_e,a_brown_d])
                p_l=[vb.tolist(), vl.tolist()]
                zer_0=np.empty(28)*0
                points_lower=np.append(np.hstack(np.vstack(p_l)),zer_0).reshape((50,2))
                r = cv2.resize(a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC)
                r = r[:,:,1]
                upper = np.array(r[:60,:])
                lower = np.array(r[46:,:])
                im_u = np.vstack((upper.T,points_upper))  
                im_u = im_u.astype('float32')
                im_u /= 255
                im_l = np.vstack((lower.T,points_lower[:,0],points_lower[:,1]))
                im_l = im_l.astype('float32')
                im_l /= 255
                x_upper=np.append(x_upper,[np.array(im_u.T)])
                x_lower=np.append(x_lower,[np.array(im_l.T)])
                label_upper=np.array(data_matrix[i,16])
                label_lower=np.array(data_matrix[i,17])
                y_upper=np.append(y_upper, [label_upper]).astype(float)
                y_lower=np.append(y_lower, [label_lower]).astype(float)
            else:
                continue
        count = count + 1
        print(count)
                
    x_upper=np.array(x_upper)
    x_lower=np.array(x_lower)
    y_upper=np.array(y_upper)
    y_lower=np.array(y_lower)
    return x_upper, x_lower, y_upper, y_lower
##############################################################################
##############################################################################
##############################################################################
##############################################################################
im_s=96
#location of SILFA dataset labels
location_SILFA="F:/Doutorado/Pesquisa/Python/Data_annotations/corpus FACs Code.csv"
#call the function
x_upper,x_lower,y_upper,y_lower=read_img_silfa_mid_crop(im_s,location_corpus)
#save outputs
np.save(".../Data_annotations/x_u_corpus", x_upper)
np.save(".../Data_annotations/x_l_corpus", x_lower)
np.save(".../Data_annotations/y_u_corpus", y_upper)
np.save(".../Data_annotations/y_l_corpus", y_lower)

#location of HM dataset labels
location_HM="F:/Doutorado/Pesquisa/Python/Data_annotations/HM Database FACs Code.csv"
#call the function
x_u_train2,x_l_train2,y_u_train2,y_l_train2=read_img_HM_mid_crop(im_s,location_HM)
#save outputs
np.save(".../Data_annotations/x_u_HM", x_u_train2)
np.save(".../Data_annotations/x_l_HM", x_l_train2)
np.save(".../Data_annotations/y_u_HM", y_u_train2)
np.save(".../Data_annotations/y_l_HM", y_l_train2)

#vactor with the number of videos in DISFA dataset
name=[1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,21,23,24,25,26,27,28,29,30,31,32]
#call the function
x_u,x_l,y_u,y_l=read_img_DISFA_mid_crop(im_s,name[:20])
#save outputs
np.save(".../Data_annotations/x_u_disfa", x_u)
np.save(".../Data_annotations/x_l_disfa", x_l)
np.save(".../Data_annotations/y_u_disfa", y_u)
np.save(".../Data_annotations/y_l_disfa", y_l)

##############################################################################
##############################################################################
