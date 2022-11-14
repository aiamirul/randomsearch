import numpy as np
import math
import pandas as pd
from collections import deque
from collections import Counter
import cv2

class keeptrack:
    def __init__(self,que_no=5,max_age=15,min_hits=30,iou_thrd=.01,dt=.3,frame_interval=1):
        self.deque_list=[]
        for i in range(1,que_no+1):
            self.deque_list.append(str(i))
        self.ids= deque(self.deque_list)
        self.tracklets=[]
        self.finals=[]
        self.max_age=np.maximum (1.0, max_age/frame_interval)
        self.min_hits=np.maximum (1.0, min_hits/frame_interval)
        self.iou_thrd=iou_thrd
        self.dt=dt*frame_interval
    def clear(self):
        self.tracklets.clear()
        self.ids= deque(self.deque_list)
        self.finals.clear()

def mode(sample):
    c = Counter(sample)
    return [k for k, v in c.items() if v == c.most_common(1)[0][1]]

def box_iou(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))

    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
    #print(s_intsec )
    return s_intsec 
    #return (float(s_intsec)/(s_a + s_b -s_intsec))

def box_intersection(a, b):
    '''
    Helper funciton to calculate the ratio between intersection 
    '''
    w_intsec = float(np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0]))))
    h_intsec = float(np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1]))))
    s_a = (a[2] - a[0])*(a[3] - a[1])
    if(s_a<1):#devide by zero issue
      return 0,0
    else:
      return float(w_intsec * h_intsec)/float(s_a),float(h_intsec/(a[3] - a[1]))



def non_max_suppression(boxes, overlapThresh):
	if len(boxes) == 0:
		return [],[],[]

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		
		#overlap = (w * h) / (area[idxs[:last]])
		#overlap = (w * h) / (area[idxs[:last]]+area[idxs[last]]-(w * h))
		overlap = (w * h) / (area[idxs[:last]]+area[idxs[last]]-(w * h))
		#print(overlap)

		#overlap = np.maximum( ((w * h) / area[idxs[:last]]) , ((w * h) / area[idxs[:i]]) )
		#overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	return np.array(boxes[pick].astype("int")),pick,idxs


def darkness(frame,gate):
	gray=frame[gate[1]:gate[3],gate[0]:gate[2]]
	blur = cv2.blur(gray, (5, 5))  # With kernel size depending upon image size
	return 1-cv2.mean(blur)[0]/255

	# hsv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[gate[1]:gate[3],gate[0]:gate[2]]
	# v_values = np.sum(hsv_img[:, :, 2])
	# area = hsv_img .shape[0] * hsv_img .shape[1]
	# avg_brightness = v_values/area
	# darkness=1-(avg_brightness/255.0)
	#return darkness


def hazze(frame,gate):
	xr=int((gate[3]-gate[1])*.30)
	#yr=int((gate[2]-gate[0])*.30)
	gray=frame[gate[1]+xr:gate[3]-xr,gate[0]:gate[2]]
	#blur = cv2.blur(gray, (5, 5)) 
	edges = cv2.Canny(gray,100,200)
	#cv2.imshow("edges",edges)
	return (1-np.sum(edges)/(edges.shape[0]*edges.shape[1]*255))


def scene_haze(frame):
	gray=frame[100:350,:]
	blur = cv2.blur(gray, (5, 5))  # With kernel size depending upon image size
	darkness=1-cv2.mean(blur)[0]/255
	edges = cv2.Canny(gray,100,200)
	hazze_score=1-np.sum(edges)/(edges.shape[0]*edges.shape[1]*255)
	if(darkness<.5 and hazze_score>.97):
		return 1
	else:
		return 0

def isitblank(frame):
    if(frame[200:300,200:300].sum()==0):
        return 1
    return 0


def write_to_csv(info,name,header):
	name_csv="{}.csv".format(name)
	df = pd.DataFrame(info, columns=header)
	df.to_csv(name_csv, header=True,index = False)



