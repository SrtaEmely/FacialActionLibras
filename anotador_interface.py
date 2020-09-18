"""
###############################################################
Import for neural network
###############################################################
"""

from keras import backend as K
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

# Import necessary components for face detection
import numpy as np
import pandas as pd
import cv2
import dlib
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

from xml.etree import ElementTree, cElementTree
from xml.dom import minidom
import xml.etree.ElementTree as etree
import xml.etree.ElementTree as et
from xml.etree.ElementTree import tostring

#%%
"""
###############################################################
FUNCTIONS
###############################################################
"""
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
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
##############################################################################
##############################################################################
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
##############################################################################
##############################################################################
def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score
##############################################################################
##############################################################################
def fmeasure(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=1)

"""
###############################################################
PARAMETERS
###############################################################
"""
PREDICTOR_PATH = 'F:/Doutorado/Pesquisa/Python/Prototipo_anotador_elan/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade_path='F:/Doutorado/Pesquisa/Python/Prototipo_anotador_elan/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
Indicesface=68
#IMAGE
im_s = 96
#OUTPUTS
#output_u=[]
#output_l=[]
#M=[]

"""
###############################################################
#Model
###############################################################
"""
modelu = load_model('F:\Doutorado\Pesquisa\Python\FacialActionLibras\Trained_model\squeezenet_u_corpus_6.h5',custom_objects={'fmeasure': fmeasure, 'precision':precision, 'recall':recall})
modelu.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',fmeasure, precision, recall])

modell = load_model('F:\Doutorado\Pesquisa\Python\FacialActionLibras\Trained_model\squeezenet_l_corpus6.h5',custom_objects={'fmeasure': fmeasure, 'precision':precision, 'recall':recall})
modell.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy',fmeasure, precision, recall])

#x_u = np.load('x_u.npy')
#x_u_train1 = np.load('F:/Doutorado/Pesquisa/Python//Data_annotations/x_u_corpus.npy')
x_u_train2 = np.load('F:/Doutorado/Pesquisa/Python/cnn_lstm/data/x_u_train2.npy')
#
y_u = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_u.npy')
y_u_train1 = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_u_corpus.npy')
y_u_train2 = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_u_train2.npy')
##############################################################################
##############################################################################
#X_u=np.append(np.append(x_u,x_u_train1),x_u_train2)
Y_u=np.append(np.append(y_u,y_u_train1),y_u_train2[:int(x_u_train2.size/(60*97*1))])
#X_u=np.nan_to_num(X_u)
#X_u = X_u.astype('float32')
##############################################################################
##############################################################################
img_rows_u, img_cols_u = 60, 97
#X_u = X_u.reshape(int(X_u.shape[0]/(60*97*1)), img_rows_u, img_cols_u, 1)
input_shape = (img_rows_u, img_cols_u, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
encoder_u = LabelEncoder()
encoder_u.fit(Y_u)
encoded_Y_u = encoder_u.transform(Y_u)
# convert integers to dummy variables (i.e. one hot encoded)
Y_u = np_utils.to_categorical(encoded_Y_u)
num_classes_u = Y_u.shape[1]
labels_encoded_u=encoder_u.inverse_transform(encoded_Y_u)
labels_ordered_u=np.sort(labels_encoded_u)
labels_ordered_u=np.append(labels_ordered_u,74)
labels_ordered_u=set(labels_ordered_u)
labels_ordered_u=np.fromiter(labels_ordered_u, int, len(labels_ordered_u))
#print(labels_ordered_u)
##############################################################################
##############################################################################
#x_l = np.load("x_l.npy")
#x_l_train1 = np.load("F:/Doutorado/Pesquisa/Python/Data_annotations/x_l_corpus.npy")
x_l_train2 = np.load("F:/Doutorado/Pesquisa/Python/cnn_lstm/data/x_l_train2.npy")
#
y_l = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_l.npy')
y_l_train1 = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_l_corpus.npy')
y_l_train2 = np.load('F:/Doutorado/Pesquisa/Python/FacialActionLibras/Data_annotations/y_l_train2.npy')
##############################################################################
##############################################################################
#X_l=np.append(np.append(x_l,x_l_train1),x_l_train2)
Y_l=np.append(np.append(y_l,y_l_train1),y_l_train2[:int(x_l_train2.size/(36*98*1))])
#X_l=np.nan_to_num(X_l)
Y_l=np.nan_to_num(Y_l)
#X_l = X_l.astype('float32')
##############################################################################
##############################################################################
img_rows_l, img_cols_l = 36, 98
#X_l = X_l.reshape(int(X_l.shape[0]/(36*98*1)), img_rows_l, img_cols_l, 1)
input_shape = (img_rows_l, img_cols_l, 1)
##############################################################################
##############################################################################
# convert class vectors to binary class matrices
encoder_l = LabelEncoder()
encoder_l.fit(Y_l)
encoded_Y_l = encoder_l.transform(Y_l)
# convert integers to dummy variables (i.e. one hot encoded)
Y_l = np_utils.to_categorical(encoded_Y_l)
num_classes_l = Y_l.shape[1]
labels_encoded_l=encoder_l.inverse_transform(encoded_Y_l)
labels_ordered_l=np.sort(labels_encoded_l)
labels_ordered_l=np.append(labels_ordered_l,73)
labels_ordered_l=set(labels_ordered_l)
labels_ordered_l=np.fromiter(labels_ordered_l, int, len(labels_ordered_l))
#print(labels_ordered_l)
##############################################################################
##############################################################################
labels_u=['0','1','2','4','5','6','9','10','1+2','1+3','1+4','1+5','1+6','5+2+7','1+9','10+44','2+3','2+4','2+5','2+6','2+7','2+8','61+73','4+3+5+43','4+3+5+44','3+5','41','4+2','43','44','4+5','46','4+9','5+62','5+6+4','5+6','5+70','61','62','63','64','4+2+70+71','70','71','72','73','2+7+70+71','4+43+70+71','1+2+43+70+71','4+10+42','4+10+44','1+2+9+43','1+2+70+71','44+70+71','1+2+5+70+71','42+1+4','62+64','1+2+3','1+2+4','1+2+5','1+2+6','1+42','1+4+6','42+43','4+43','1+4+9','1+5','42+61','4+2+62','1+2+10','1+2','1+2','1+2','1+2','1+2+6','1+2','1+2','4+42+44','1+2+41','1+2+42','1+2+43','1+2+44','1+2+46','1+2+5',     '4+42+62',     '1+2+5',      '1+2+61',      '1+2+62',      '1+2+63',      '1+2+64',     '1+2+5',       '2+42',     '5+70+71',      '2+46',      '1+2+70',      '1+2+71',      '1+2+73',       '2+5+6',    '1+2+41+61',    '1+2+41+62',      '43+61',      '43+70',     '4+72',      '43+73',      '42',     '4+44',    '2+3+42+70',      '43',      '2+8+61',     '4+43+44',      '1+3+43',      '1+3+44',  '1+2+42+70+71',    '1+9+42',    '1+9+43',      '4+9+44',      '1+3+62',      '1+3+63',     '4+43+73',   '4+44+70+71',    '1+2+42+63',     '4+2+3+64',     '4+43', '4+44+46+70+71',      '1+9',    '1+2+42',    '1+2+43',       '4',       '4',      '70+71',     '4+44+46',      '1+4+42',      '1+4+43',       '4',       '4',       '4',      '1+9+61',       '4',       '4',     '4+44+62',       '9+44',      '1+4+64',       '4+41',       '4+42',       '4+43',       '4+44',     '4+44+73',       '4+46',       '4+61',       '4+62',       '4+64',       '4+71',     '4+44',       '4+73',     '4+70+71',     '4+42',     '4+43',     '4+44', '74']
labels_l=['0','1','2','3','4','5','10+25','25+62','22+25','9','10','25+70','12','13','14','15','16','17','18','24','20','25','22','23','24','25','26','27','28','34','61+72','12+25','12+20+25+26','33','34','35',       '15+72',   '15+16+20+25',     '13+16+25',    '18+22+25',     '16+23+25',        '72',         '61',         '62',   '12+22+25+26',   '20+22+25+26',     '15+16+17',       '26+28',       '21+17',     '12+19+25',     '15+16+20',         '72',         '73',       '26+33',     '15+16+25',       '16+17',       '16+20',        '26',       '16+23',       '16+25',       '16+26', '15+17+20+25+26',        '28',       '26+62',     '22+23+25',   '19+22+25+26',     '10+25+26',     '10+25+27',    '22+25+26',   '22+25+26+72',   '22+25+26+73',       '16+70',       '25+72',   '10+19+25+26',   '25+27+70+71',     '12+25+26',     '12+25+27',     '12+25+28',     '15+22+25',   '15+16+25+26',     '15+17+20',     '12+20+25',     '15+17+22',     '15+17+23',     '10+25+72',     '12+15+17',     '12+25+41',       '22+23',     '15+17+24',       '22+25',       '22+26',     '15+17+25',     '15+17+26',     '15+17+28',       '17+20',     '20+24+26',       '17+23',       '17+24',       '17+25',       '17+26',       '12+15',     '14+25+26',       '12+17',     '17+22+25',     '18+70+71',       '12+20',       '12+22',       '12+23',       '12+24',       '12+25',       '12+26',       '17+34',       '12+28',     '12+25+72',     '12+25+73',   '15+16+25+70',   '15+16+25+72',     '15+17+62',     '25+26+28',      '19+25',     '12+28+72',     '15+17+71',     '16+25+25',      '17+26',     '16+25+26',     '17+20+25',        '22+25',     '19+22+25',   '15+20+25+26',   '16+19+25+26',        '32',     '16+20+25',        '34',       '17+72',   '12+16+25+72',      '22+54',   '12172526',   '17+20+25+36',     '10+16+25',     '15+28+25',   '12+17+20+26', '15+16+17+20+26',     '18+25+26',     '25+26+72',     '25+26+73',       '28+25',     '16+25+72',     '23+26+34', '15+16+19+25+26',       '28+32',      '10+25',       '23+24',       '23+25',       '23+26',     '12+16+25',   '25+26+28+73',       '18+22',       '23+34',     '20+25+26',       '18+25',       '13+14',       '18+26',     '20+25+27',     '20+25+32',   '16+20+25+26', '12+17+20+25+26',       '18+34',       '13+23',   '15+22+25+72',       '13+25',      '18+22',      '25+26',       '13+28',      '25+27',     '20+15+17',   '10+15+17+25', '10+16+19+25+26',       '28+72',       '28+73',     '25+27+28',     '23+70+71', '10+12+16+25+72',     '22+25+26',     '22+25+27',   '16+22+25+26',       '23+70',       '23+71',       '23+72',       '23+73',      '15+25',    '19+25+26',       '18+72',       '18+73',     '10+22+25',     '13+19+25',   '15+17+19+25',     '24+25+26',   '10+16+25+26',   '17+20+25+26',     '22+25+61',   '17+20+25+27',   '17+20+25+28',     '25+27+72',     '25+27+73',   '17+20+25+29',   '17+20+25+30',   '17+20+25+31',      '16+25',     '10+12+25',   '17+20+25+32',     '22+25+72',     '22+25+73',   '17+20+25+33',   '17+20+25+34',   '17+20+25+35',     '12+22+25',        '13+72',     '15+19+25',     '12+17+20',       '24+26',       '24+28',     '12+17+25',   '15+16+17+25',   '25+26+70+71',       '19+22',       '19+25',       '34+62',       '19+28',    '15+16+25',     '17+24+28',   '10+16+25+72',       '14+23',       '34+72',       '14+25',       '34+73',   '15+18+22+25',   '19+25+26+28',       '25+26',   '10+12+16+25',   '18+20+25+26',   '26+20+25+26',        '18',       '24+70',       '24+72',   '20+15+17+20',     '13+25+26',   '15+17+25+26',   '22+25+70+71',     '16+22+25',        '34', '18+22+25+70+71',   '15+23+25+26',   '15+17+20+25',   '15+17+20+26',     '16+17+25',   '18+22+25+26',     '34+70+71',   '18+22+25+72',   '19+25+26+72',     '19+25+26',     '19+25+27',     '15+25+26',   '15+19+25+26',     '18+22+25',     '18+22+26',     '19+25+28',       '25+16',    '15+17+26',        '72',       '10+16',     '15+20+25',     '15+20+26',   '18+22+25+73',       '25',       '25+26',       '25+27',       '25+28',   '19+20+25+27',   '20+15+17+71',       '25+31',       '25+32',       '20+24',       '20+25',       '20+26',       '20+27',       '15+16',       '15+17',       '15+18',     '17+25+26',       '15+20',     '23+24+25',       '15+22',       '15+23',       '15+24',       '15+25',       '15+26',       '10+15',       '15+28',       '10+17',     '17+20+26',      '20+25+26',       '10+23']
##############################################################################
##############################################################################

#%%
def neural_net(path):
    v_entry=cv2.VideoCapture(path,0)
    Frames = int(v_entry.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = v_entry.get(cv2.CAP_PROP_FPS)
    #print('Number of Frames=', Frames )
    output_u=[]
    output_l=[]
    M=[]
    for i, j in enumerate(range(0,Frames)):
        print(i,int(j))
        v_entry.set(1, int(j))
        ret, im = v_entry.read()
        points_u=np.empty((21,2))*0
        points_l=np.empty((32,2))*0
        if ret is True:
            a,l = crop_face(im)
            c=get_landmarks(a)
              #
            points_u[:9,:]=c[17:26,:]
            points_u[10:,:]=c[36:47,:]
            vp=np.stack((points_u))
              #
            points_l[:12,:]=c[2:14,:]
            #points_l[4:7,:]=c[11:14,:]
            points_l[13:,:]=c[48:67,:]
            vb=np.stack((points_l))
              #
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
              #
            d_lips_h1=np.squeeze(np.asarray(c[48]-c[54]))
            d_lips_h2=np.squeeze(np.asarray(c[60]-c[64]))
            d_lips_v1=np.squeeze(np.asarray(c[51]-c[57]))
            d_lips_v2=np.squeeze(np.asarray(c[62]-c[66]))
            vl=np.stack((d_lips_h1,d_lips_h2,d_lips_v1,d_lips_v2))
              #
            p_u=[vp.tolist(), vs.tolist()]
            points_upper=np.hstack([np.hstack(np.vstack(p_u)),a_brown_e,a_brown_d])
            p_l=[vb.tolist(), vl.tolist()]
            points_lower=np.hstack(np.vstack(p_l)).reshape((36,2))
              ###
            r = cv2.resize(a, dsize=(im_s, im_s), interpolation=cv2.INTER_CUBIC)
            r = r[:,:,1]
            upper = np.array(r[:60,:])
            lower = np.array(r[60:,:])
              #
            im_u = np.vstack((upper.T,points_upper))  
            im_u = im_u.astype('float32')
            im_u /= 255
            im_l = np.vstack((lower.T,points_lower[:,0],points_lower[:,1]))
            im_l = im_l.astype('float32')
            im_l /= 255

            x_upper = np.expand_dims(im_u, axis=0)
            x_lower = np.expand_dims(im_l, axis=0)
            x_upper=x_upper.reshape((1, 60, 97, 1))
            x_lower=x_lower.reshape((1, 36, 98, 1))

            exit_u = modelu.predict(x_upper)
            exit_l = modell.predict(x_lower)
            exit_u=np.argmax(exit_u, axis=1)
            exit_l=np.argmax(exit_l, axis=1)
            e_labels_u=encoder_u.inverse_transform(exit_u)
            e_labels_l=encoder_l.inverse_transform(exit_l)
            print(e_labels_u)
            print(e_labels_l)
            
            output_u = np.append(output_u, e_labels_u)
            output_l = np.append(output_l, e_labels_l)
        else:
            output_u = np.append(output_u,74)
            output_l = np.append(output_l, 72)
            continue
    
    all_exit_u=np.matrix(zip(range(0,Frames),output_u))
    all_exit_l=np.matrix(zip(range(0,Frames),output_l))
    
    root = et.Element('TIERS', **{'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance'}, **{'xsi:noNamespaceSchemaLocation': 'file:avatech-tiers.xsd'})
    somedata = et.SubElement(root, 'TIER', columns="AUs")
    
    for m,n in enumerate(range(0,Frames)):
        print(m)
        if (np.where(labels_ordered_u==output_u[m])):
            a=np.where(labels_ordered_u==output_u[m])
            print(a)
            print(labels_u[int(a[0][0])])
            if (np.where(labels_ordered_l==output_l[m])):
                b=np.where(labels_ordered_l==output_l[m])
                print(b)
                print(labels_l[int(b[0][0])])
                ms_inicial=round((m*(1000 / (fps / 1.001)))*.001,3)
                ms_final=round(((m+1)*(1000 / (fps / 1.001)))*.001,3)
                full_elan_exit_u=("<span start= \"%s\" end=\"%s\" ><v>%s</v></span>"%(ms_inicial,ms_final,labels_u[int(a[0][0])]))
              
                child1 = ElementTree.SubElement(somedata,"span", start='%s'%(ms_inicial), end="%s"%(ms_final))
                  
                v = etree.Element("v")
                v.text = "%s+%s"%(labels_u[int(a[0][0])],labels_l[int(b[0][0])])
                child1.append(v)

                tree = cElementTree.ElementTree(root) # wrap it in an ElementTree instance, and save as XML
                
                t = minidom.parseString(ElementTree.tostring(root)).toprettyxml() # Since ElementTree write() has no pretty printing support, used minidom to beautify the xml.
                tree1 = ElementTree.ElementTree(ElementTree.fromstring(t))
                print(tree1)
                tree1.write("file.xml",encoding="utf-8", xml_declaration=True)

            else:
                continue
        else:
            continue
        
    return tree1            
    #return tree1

my_address='F:/Doutorado/Pesquisa/Python/FacialActionLibras/Video_test/bomdia_libras.mp4'
output=neural_net(my_address)
output.write("file.xml",encoding="utf-8", xml_declaration=True)