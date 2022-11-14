import os
import sys
#from tkinter import Frame
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import cv2
import time
import numpy as np
import json
import pandas as pd
try:
    import kltracker as tracker
except ImportError:
    from . import kltracker as tracker
try:
    import utility as utl
except ImportError:
    from . import utility as utl
try:
    import findgpu as fg
except ImportError:
    from . import findgpu as fg
import collections
import statistics
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import asyncio
import logging
import glob
tf.get_logger().setLevel(logging.ERROR)

TRIGGER_URL_TEMPLATE = os.environ.get('TRIGGER_URL_TEMPLATE_FR', '')

class FranceThoroughbredBase:
    def __init__(self,in_filename,dst,frame_interval,model=None,debug=0,xtrigger=-1,aboutsignal=-1,triger_info=None,nh=15,vis=0, in_buffer=None, width=None, height=None, raceid=None, program=None, test_mode=1):
        self.in_filename=in_filename
        self.dst=dst
        self.reststream=0
        os.makedirs(self.dst, exist_ok=True)
        if in_filename is not None:
            self.test_mode = test_mode
           
            self.raceid=os.path.splitext(os.path.basename(in_filename))[0]
            self.cap = cv2.VideoCapture(self.in_filename)
            self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

            ret, self.frame = self.cap.read()
            self.rows, self.cols, _ = self.frame.shape
        else:
            self.in_buffer = in_buffer
            self.cols = width
            self.rows = height
            self.fps = 30
            self.frame = None
            self.raceid = raceid
            self.program = program
            self.test_mode = test_mode

        '''general param'''
        self.modelname='ef-d0-v1'
        self.model=model
        self.names=["horse",
                      "empty","open","full", 
                      "gate",
                      "loading","front","loading",
                      "flag-man","flag","jockey",
                      "T1"
                    ]

        self.horse_fraction=0.2
        self.Fn_delay=[91,2]
        self.Ld_delay=[91,20,10]
        self.AboutTimeFrame = 0
        self.activate_bl=0
        self.imminent = False
        self.aboutsignal=aboutsignal
        self.prev_trigger_type = 'M'
        self.trigger_type="M"
        self.debug=debug
        self.triger_info=triger_info
        self.frame_counter=0
        self.frame_interval=frame_interval
        self.missframe_counter=0
        self.frame=None
        self.frame_gray=None
        self.vis_img=None
        self.vis=vis
        self.pre_obj=[]
        self.post_obj=[]
        ''' flow param'''
        self.optical_flow=cv2.DISOpticalFlow_create(preset=cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.p_opt_frame=None
        self.opt_frame=None
        self.cam_transition=0
        self.flow_thresh=20
   
        '''gate param'''
        self.gate=None
        self.gate_counter=0 # jump or Thoroughbred
        '''stall param'''
        self.total_stall=0
        self.flag_man_flag=0
        self.stallratio=0
        self.stall_haziness =0
        self.stall_dimness=0
        '''timer param'''
        self.start_pattern=cv2.imread(os.path.join(os.path.dirname(__file__), '..', "EventLogo", "start.png"),0)
        self.start_roi=[91, 143, 11, 92]

        self.end_pattern=cv2.imread(os.path.join(os.path.dirname(__file__), '..', "EventLogo", "M.png"),0)
        self.end_roi=[360, 406, 127, 172]
        
        self.about_pattern=cv2.imread(os.path.join(os.path.dirname(__file__), '..', "EventLogo", "about2start.png"),0)
        self.about_roi=[88, 141, 30, 138]
        self.patterns=[]
        '''trigger param'''
        self.nh=nh
        self.frame_view=0
        
        self.view='neutral'
        self.view_transition=0
        #----------------------
        self.inframe=[]
        self.stop_signal=False
        tmp_file="tmp/{}.csv".format(raceid)
        self.out_horses=None
        if(os.path.isfile(tmp_file)):
            val=pd.read_csv(tmp_file,delimiter=',')
            self.imminent = int(val.imminent[0])
            self.out_horses=int(val.nh[0])
            log_str="read from temp file -> imm:{} nh:{}".format(self.imminent,self.out_horses)
            logging.info(log_str)
            os.remove(tmp_file)
        if(self.out_horses is None):
           self.out_horses=self.nh
        #print(self.out_horses,self.imminent)
        #-----------------------
        self.least_out_horses=self.out_horses
        self.detected_horses=0
        #self.out_horse_que = collections.deque(maxlen=15)
        self.out_horse_que = collections.deque(maxlen=7)
        self.active_loading=[]
        self.loading_begin=-1
        self.loading_counter=0
        self.loading_regulator=20
        self.back=0
        self.track=self.raceid[-5:-2]
        self.delay_to_trigger=0
        self.fog=0
        self.abflag=1
        self.abcounter=0
        self.about_thresh=5000
        '''config param'''
        #self.min_hsize_for_active_loading=4000
        self.fstall_run_conf=0
        self.bstall_run_conf=0
        self.fstall_corner_conf=.8
        self.fstall_normal_conf=.4
        self.bstall_corner_conf=.4
        self.bstall_normal_conf=.2
        self.counter_regulation=.6
        self.activate_flage=0
        self.horse_max_age=30
        self.horse_min_hit=10
        self.full_max_age=10
        self.full_min_hit=15
        self.objects_confidence=[.5,
             .3,.3,.3,
             .1,
             .3,.3,.3,
             .5,.5,.5,
             .2
             ]
        
        self.trigger_time=[self.track,self.raceid,"-1",xtrigger,nh,"-1","-1","-1"]
        ''' tracking param '''
        self.full_trk = utl.keeptrack(que_no=100,max_age=self.full_max_age,min_hits=self.full_min_hit,iou_thrd=.01,dt=.3,frame_interval=self.frame_interval)
        self.horse_trk = utl.keeptrack(que_no=50,max_age=self.horse_max_age,min_hits=self.horse_min_hit,iou_thrd=.1,dt=.05,frame_interval=self.frame_interval)
               
        self.front_trk = utl.keeptrack(que_no=4,max_age=20,min_hits=10,iou_thrd=.01,dt=.3,frame_interval=self.frame_interval)
        self.loading_trk = utl.keeptrack(que_no=4,max_age=20,min_hits=10,iou_thrd=.01,dt=.3,frame_interval=self.frame_interval)
    
    def set_track_param(self,config=os.path.join(os.path.dirname(__file__), 'config.json')):
        with open(config) as json_file:
            data = json.load(json_file)
            for p in data['param']:
                if(p['track']==self.track):
                    self.activate_flage=p['activate_flage']
                    self.fstall_normal_conf=p['fstall_normal_conf']
                    self.fstall_corner_conf=p['fstall_corner_conf']
                    self.bstall_normal_conf=p['bstall_normal_conf']
                    self.bstall_corner_conf=p['bstall_corner_conf']
                    self.bstall_run_conf=self.bstall_normal_conf
                    self.fstall_run_conf=self.fstall_normal_conf
                    self.counter_regulation=p['counter_regulator']
                    self.objects_confidence=p['objects_confidence']
                    self.horse_min_hit=p['horse_min_hit']
                    self.horse_max_age=p['horse_max_age']
                    self.full_min_hit=p['full_min_hit']
                    self.full_max_age=p['full_max_age']
                    self.flow_thresh=p['flow_thresh']
                    self.modelname=p['modelname']
                    self.about_thresh=p['about_thresh']
                    self.horse_fraction=p["horse_fraction"]
                    self.Fn_delay=p["Fn_delay"]
                    self.Ld_delay=p["Ld_delay"]
                    self.activate_bl=p["activate_bl"]
                    self.horse_trk = utl.keeptrack(que_no=50,max_age=self.horse_max_age,min_hits=self.horse_min_hit,iou_thrd=.01,dt=.05,frame_interval=self.frame_interval)
                    self.full_trk = utl.keeptrack(que_no=100,max_age=self.full_max_age,min_hits=self.full_min_hit,iou_thrd=.01,dt=.3,frame_interval=self.frame_interval)  
    
    def load_track_pattern(self):
        for img_path  in glob.glob(os.path.join(os.path.dirname(__file__), '..', "TracksLogo", self.track, "*.png")):
                self.patterns.append(cv2.imread(img_path ,0))   
    
    def right_tracks(self):
        if(len(self.patterns)<1):
            return -1
        #gray=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        roiID=[83,95,60,112]
        for templ  in self.patterns:
            res = cv2.matchTemplate(  self.frame_gray[roiID[0]:roiID[1],roiID[2]:roiID[3]],templ ,5)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if(max_val>.9):
                return 1
        return 0

    def rest_value(self):
        self.pre_obj=[]
        self.post_obj=[]
    
    def initialize_myque(self):
        for i in range(0, len(self.out_horse_que)):
            self.out_horse_que[i]=self.out_horses
    
    def clear_tracks(self):
        self.front_trk.clear()
        self.loading_trk.clear()
        self.full_trk.clear()
        self.horse_trk.clear()
        self.active_loading.clear()
        self.view_transition=0
        self.loading_begin=-1
        self.loading_counter=0
        self.loading_regulator=20
        self.back=0
        self.inframe=[]

    def disply_tracks(self,tracks,color=(0,255,255)):
        for trk in tracks.finals :
            p1 = (trk.box[0], trk.box[1])
            p2 = (trk.box[2], trk.box[3])
            cv2.rectangle(self.vis_img, p1, p2, color, 2, 15)
    
    def disply_info(self,trigger=0):
        im_pil = Image.fromarray(self.vis_img)
        text="View: -- "
        stall="Full stall: -- "
        #no_horse="Out horses: {} ".format(int(self.out_horses))
        no_horse="Out horses: {} ".format(int(self.out_horses))
        text="View: "+self.view
        stall="horse stat: {}-{}-{} ".format(len(self.full_trk.finals),self.detected_horses,self.out_horses)
        d = ImageDraw.Draw(im_pil)
        fnt= ImageFont.load_default()
        #fnt = ImageFont.truetype('Calibri_Bold.ttf', 20)
        d.text((10,20), text,align='center', font=fnt, fill=(159, 159, 159))
        #d.text((170,20), str(int(self.fle)),align='center', font=fnt, fill=(0,0 , 159))
        d.text((220,20), no_horse,align='center', font=fnt, fill=(159, 159, 159))
        d.text((400,20), stall,align='center', font=fnt, fill=(159, 159, 159))
        if(trigger):
            d.text((260,40), 'Trigger',align='center', font=fnt, fill=(0, 0, 159))
        self.vis_img = np.asarray(im_pil)
        stall_p="Size: {:.2f} Dullness: {:.2f} Drakness: {:.2f} inframe: {:.2f}".format(self.stallratio,self.stall_dimness,self.stall_haziness,len(self.inframe))
        cv2.putText(self.vis_img,stall_p,(50,50), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 159), 1, cv2.LINE_AA)

   
    def disply_objects(self):
        obj=self.post_obj
        #obj=self.pre_obj
        for i in range(len(obj)):
            class_name=self.names[obj[i][0]]
            if(class_name=="jockey"):
                cv2.rectangle(self.vis_img, (int(obj[i][2]), int(obj[i][3])), (int(obj[i][4]), 
                                                   int(obj[i][5])), (0,0,255), thickness=2)
            if(class_name=="flag-man" or class_name=="flag"):
                cv2.rectangle(self.vis_img, (int(obj[i][2]), int(obj[i][3])), (int(obj[i][4]), 
                                                   int(obj[i][5])), (255,0,255), thickness=2)
            if(class_name=="gate" ):
                cv2.rectangle(self.vis_img, (int(obj[i][2]), int(obj[i][3])), (int(obj[i][4]), 
                                                   int(obj[i][5])), (255,0,0), thickness=2)
            if(class_name=="horse"):
                cv2.rectangle(self.vis_img, (int(obj[i][2]), int(obj[i][3])), (int(obj[i][4]), 
                                                   int(obj[i][5])), (0,255,0), thickness=2)
    
    def tracker_base(self,z_box,l_box,tracks,check_box=0):
        x_box=[]
        if len(tracks.tracklets) > 0:
            for trk in tracks.tracklets:
                x_box.append(trk.box)#state
        matched, unmatched_dets, unmatched_trks \
            = tracker.assignment(x_box, z_box, iou_thrd = tracks.iou_thrd)
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracks.tracklets[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            tmp_trk.labels.append(l_box[det_idx])
            if(check_box==1): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                if(w<40):
                    xx[4]=xx[0]+40
                    w=40
                xx[6]=xx[2]+(w*2)
            if(check_box==2): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                h=abs(xx[2]-xx[6])
                if((2*w)>h):
                    xx[6]=xx[2]+(w*3)
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0
        # unmatched detections      
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker(dt=tracks.dt) # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            if(check_box==1): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                if(w<40):
                    xx[4]=xx[0]+40
                    w=40
                xx[6]=xx[2]+(w*2)
            if(check_box==2): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                h=abs(xx[2]-xx[6])
                if((2*w)>h):
                    xx[6]=xx[2]+(w*3)
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = tracks.ids.popleft() # assign an ID for the tracker
            tmp_trk.labels.append(l_box[idx])
            tracks.tracklets.append(tmp_trk)
            x_box.append(xx)
        # unmatched track_list       
        for trk_idx in unmatched_trks:
            tmp_trk = tracks.tracklets[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            if(check_box==1): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                if(w<40):
                    xx[4]=xx[0]+40
                    w=40
                xx[6]=xx[2]+(w*2)
            if(check_box==2): #temporary fixed size issue
                w=abs(xx[0]-xx[4])
                h=abs(xx[2]-xx[6])
                if((2*w)>h):
                    xx[6]=xx[2]+(w*3)
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
        # retun good track  
        tracks.finals.clear()
        for trk in tracks.tracklets:
            if ((trk.hits >= tracks.min_hits) and (trk.no_losses <=tracks.max_age)):
                #print(utl.mode(trk.labels)[0])
                trk.label=self.names[int(utl.mode(trk.labels)[0])]
                tracks.finals.append(trk)
        
        # cleaning
        deleted_tracks=[]
        deleted_tracks =filter(lambda x: x.no_losses >tracks.max_age, tracks.tracklets) 
        for trk in deleted_tracks:
            tracks.ids.append(trk.id)
        tracks.tracklets= [x for x in tracks.tracklets if x.no_losses<=tracks.max_age]
       
    def SBD(self):
        '''Shot Boundary Detection'''
        flow_error=0
        #self.opt_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)[100:400,:]
        self.opt_frame = self.frame_gray[100:400,:]
        self.opt_frame = cv2.resize(self.opt_frame,dsize=(None), fx=.3,fy=.3)
        if self.p_opt_frame is not None:
            flow = self.optical_flow.calc(self.p_opt_frame, self.opt_frame, None)
            h, w = flow.shape[:2]
            flow = -flow
            flow[:,:,0] += np.arange(w)
            flow[:,:,1] += np.arange(h)[:,np.newaxis]
            warp = cv2.remap(self.p_opt_frame, flow, None, cv2.INTER_LINEAR)
            absdiff=cv2.absdiff(warp,self.opt_frame)
            flow_error=np.mean(absdiff)
        self.p_opt_frame=self.opt_frame
        if(flow_error>self.flow_thresh):
            return 1
        else:
            return 0

    def detected_stalls(self,dataframe):
        '''I calculate total detected stalls'''
        stall=dataframe[(
                        (dataframe["label"] =="full") | 
                        (dataframe["label"] =="empty")  
                        #| (dataframe["label"] =="open")
                        ) & 
                        (dataframe["score"] >.4)]
        if(len(stall)>0):
            stall_width=stall.w.median()
            stall=self.contex_filter(input_list=stall.values,crowd_size=stall_width,wf=1)
            z_box,l_box=self.overlap_filter(input_list=stall,overlap_th=.1)
            return (len(z_box))
        return 0

    def count_gate(self,dataframe):
        '''I count how many time gate appears'''
        self.abcounter+=1
        if(len(dataframe[(dataframe["label"] =="gate") & (dataframe["score"] >=.3)])>0 and self.detected_horses>3):
            self.gate_counter+=1
        elif(self.detected_horses<=3 and self.abcounter>0):
            self.abcounter-=1
        #print(self.gate_counter,self.abcounter) 
         
    def find_flag_man(self,dataframe):
            flag_man=dataframe[ ((dataframe["label"] =="flag-man") | (dataframe["label"] =="flag"))]
            for em in flag_man.values:
                    self.post_obj.append(em)
            self.flag_man_flag=len(flag_man)

    def select_view(self,dataframe):
        gate_obj=dataframe[(dataframe["label"] =="gate")]
        if(self.trigger_time[-2]!="-1"):
                self.count_gate(dataframe) 
        if(len(gate_obj)==0): # No gate view is neutral
            self.view='neutral'
            return self.view
         
        gate_tmp=gate_obj.loc[gate_obj['score'].idxmax()].values
        self.gate=[int(gate_tmp[2]),int(gate_tmp[3]),int(gate_tmp[4]),int(gate_tmp[5])]
        self.post_obj.append(gate_tmp)
        self.total_stall=self.detected_stalls(dataframe=dataframe)
        if(self.total_stall>2):# not a jump race
            self.trigger_type="MS"
        if(self.cam_transition or self.view_transition>4): # shot has been changed, reset
            self.view='neutral'
            if(len(self.active_loading)<3 and len(self.active_loading)>0 and self.out_horses<self.nh and not self.back and self.out_horses>0 and self.detected_horses<3 ): # when camera view changes before horse is loaded
                self.out_horses-=1
            self.initialize_myque()
            return self.view
        
        potential_view=dataframe[ (dataframe["label"]=="front") | (dataframe["label"]=="loading")|(dataframe["label"]=="T1")]
        if(len(potential_view)>0): # find view type
                self.frame_view=potential_view.loc[potential_view['score'].idxmax()].values

        if(self.view=='neutral' and (len(potential_view)>0)):# only get new view if the shot changes
                self.view=self.frame_view[6]
                self.post_obj.append(self.frame_view)
        else:       
            z_box=[]
            l_box=[]
            if(len(potential_view)>0): # detect shot transition using view
                if(self.view!=self.frame_view[6]):
                    self.view_transition+=1
                elif(self.view_transition!=0):
                    self.view_transition-=1
            if(self.view=='T1'):
                
                roi_t=[self.frame_view[2],self.frame_view[3],self.frame_view[4],self.frame_view[5]]
                return self.view
            
            elif(self.view=='front'):
                
                self.back=0
                self.find_flag_man(dataframe)
                roi_t=[self.frame_view[2],self.frame_view[3],self.frame_view[4],self.frame_view[5]]
                z_box.append(roi_t)
                l_box.append(self.frame_view[0])
                self.tracker_base(z_box=z_box,l_box=l_box,tracks=self.front_trk,check_box=0)
                return self.view
            
            elif(self.view=='loading'):
                if(self.frame_view[0]==5):
                    self.back=1
                else:
                    self.back=0
                roi_t=[self.frame_view[2],self.frame_view[3],self.frame_view[4],self.frame_view[5]]
                z_box.append(roi_t)
                l_box.append(self.frame_view[0])
                self.tracker_base(z_box=z_box,l_box=l_box,tracks=self.loading_trk,check_box=0)
                return self.view
  
    def adjust_stall_threshold(self):
        self.stall_dimness=utl.darkness(self.frame_gray,self.gate)
        self.stall_haziness=utl.hazze(self.frame_gray,self.gate)
        self.bstall_run_conf=self.bstall_normal_conf
        self.fstall_run_conf=self.fstall_normal_conf
        if(self.stallratio<.2 or self.stall_haziness>.9 or self.stall_dimness>=.74):
            self.fstall_run_conf=self.fstall_corner_conf
        if(self.stallratio<.3 or self.stall_haziness>.9):
            self.bstall_run_conf=self.bstall_corner_conf


    
    def contex_filter(self,input_list,crowd_size,wf=0,l_ratio=1.5,s_ration=.6):
        filtred=[]
        for ri in input_list:
            lsize=0
            if(wf==0):
                lsize=ri[-2]#get hieght
            else:
                lsize=ri[-1] # get width
            pass_size=ri[-2]*ri[-1]/(self.rows*self.cols)
            if (((crowd_size*l_ratio > lsize and  crowd_size/l_ratio < lsize) and pass_size<s_ration)): # for less than 3 objects filter is not accurate
                 filtred.append(ri)
        return  filtred
    
    def overlap_filter(self,input_list,overlap_th=.1):
        z_box=[]
        l_box=[]
        if(len(input_list)>0):
            boxes=[]
            for ri in input_list:
                boxes.append([ri[2],ri[3],ri[4],ri[5]])
            boxes,pick,idx=utl.non_max_suppression(np.array(boxes, dtype=np.float32),  overlap_th)
            for idx,ri in enumerate (input_list):
                if(idx in pick) :    
                    z_box.append([ri[2],ri[3],ri[4],ri[5]])
                    l_box.append(ri[0])
                    self.post_obj.append(ri)
        return z_box,l_box

    def find_full_stall(self,dataframe,stall_confidence):
        stall=dataframe[(dataframe["label"] =="full") & (dataframe["score"] >stall_confidence)]
        if(len(stall)>0):
            stall_height=stall.h.median()
            stall_width=stall.w.median()
            self.stallratio=stall_height/(self.frame.shape[0])
            self.adjust_stall_threshold()
            stall=self.contex_filter(input_list=stall.values,crowd_size=stall_width,wf=1)
        z_box,l_box=self.overlap_filter(input_list=stall,overlap_th=.2)
        self.tracker_base(z_box=z_box,l_box=l_box,tracks=self.full_trk,check_box=2)
  
    def find_jockey_at_gate(self,dataframe):
        full_stall=dataframe[(dataframe["label"] =="jockey")]
        for em in full_stall.values:
                self.post_obj.append(em)
        return (len(full_stall))
    
    def get_horses(self,dataframe):
        horse=dataframe[dataframe["label"]=="horse"]
        if(len(horse)==0):
            return [],[],self.frame.shape[0]
        horse_hight=horse.h.median()
        if((horse_hight/self.frame.shape[0])>.2 or (len(horse)<4 and not self.back)):
            horse=dataframe[(dataframe["label"] =="horse") & (dataframe["score"] >=.7)]
        if(len(horse)==0):
            return [],[],self.frame.shape[0]
        horse=self.contex_filter(input_list=horse.values,crowd_size=horse_hight,wf=0)
        if(len(horse)==0):
            return [],[],self.frame.shape[0]
        z_box,l_box=self.overlap_filter(input_list=horse,overlap_th=.4)
        return z_box,l_box,horse_hight
    
    def count_horses(self,dataframe):
        z_box,l_box,horse_hight=self.get_horses(dataframe)
        #print("z",len(z_box))
        self.out_horse_que.append(len(z_box))
        self.detected_horses=statistics.median(self.out_horse_que)
        self.out_horses=min(max(self.out_horses,self.detected_horses),self.nh)
        fullstall=len(self.full_trk.finals)
        current_empty_stall=max(self.nh-fullstall,0)
        if((self.view=='front' or self.view=='T1')):
            if(len(z_box)==0):
                self.detected_horses=0
            self.least_out_horses=min(current_empty_stall,self.least_out_horses)
            self.out_horses=min(current_empty_stall,self.out_horses)
            if(self.view=='front' and current_empty_stall<=3 and self.stallratio<.4):
                cover_stall=max(self.nh-self.total_stall,0)
                self.out_horses=max(current_empty_stall-cover_stall,0)
        # loading
        elif(self.view=='loading'):
            self.out_horses=min(current_empty_stall,self.out_horses)
            jockey_no=self.find_jockey_at_gate(dataframe=dataframe)
            self.tracker_base(z_box=z_box,l_box=l_box,tracks=self.horse_trk,check_box=0)
            self.detected_horses=max(len(self.horse_trk.finals),self.detected_horses)
            for acl in self.active_loading: # check if the track still exist( otherwise horse is loaded)
                horse_is_loaded=True
                for trk in self.horse_trk.finals:
                    if(acl==trk.id):
                        horse_is_loaded=False
                if(horse_is_loaded and (len(self.active_loading)<4)):
                    #if(horse_is_loaded):
                    self.active_loading.remove(acl)
                    if(self.out_horses!=0 ):
                        if(self.loading_counter<2):
                                self.loading_regulator=20
                        else:
                            self.loading_regulator=((self.frame_counter-self.loading_begin)/25)
                        self.loading_begin=self.frame_counter
                        self.loading_counter+=1
                        if(self.loading_regulator>5):
                            self.out_horses-=1
                            self.inframe=[]
                            self.initialize_myque()

            for trk in self.horse_trk.finals:
                area_intersect,_=utl.box_intersection([trk.box[0],trk.box[1],trk.box[2], trk.box[3]],self.gate) # active loading are any horse at the gate area
                if(area_intersect>.1):
                    if trk.id not in self.active_loading:
                        self.active_loading.append(trk.id)
                    if (trk.id in self.active_loading and (
                        (trk.box[0]<=10  or trk.box[2]>=self.cols) )
                        ): # if horses went out of the scene
                        self.active_loading.remove(trk.id)
                elif(trk.id not in self.active_loading and (trk.box[0]<=10  or trk.box[2]>=self.cols)):
                     if(trk.id not in self.inframe):
                         self.inframe.append(trk.id )
            self.out_horses=max(min(self.out_horses,self.nh-jockey_no),0)
            if((horse_hight/self.frame.shape[0])<=self.horse_fraction and self.detected_horses!=0 and len(self.inframe)==0
                and ( (self.out_horses*self.counter_regulation<self.detected_horses ) or self.out_horses<3)
                ):
                self.out_horses=min(self.out_horses, self.detected_horses)
                  

    def post_process(self):
        df = pd.DataFrame(self.pre_obj, columns=["cls_id","score","xmin","ymin","xmax","ymax","label","cx","cy","h","w"])
        ret=self.select_view(dataframe=df)
        if(ret=='front'):
            self.find_full_stall(dataframe=df,stall_confidence=self.fstall_run_conf)
            if(len(self.front_trk.finals)>0):
                self.count_horses(dataframe=df)
        elif(ret=='T1'):
            self.find_full_stall(dataframe=df,stall_confidence=self.fstall_run_conf)
            self.count_horses(dataframe=df)
        elif(ret=='loading'):
             self.find_full_stall(dataframe=df,stall_confidence=self.bstall_run_conf)
             self.count_horses(dataframe=df)
        elif(ret=='neutral'):
            self.count_horses(dataframe=df)
            self.clear_tracks()

    def detect(self):
        detections=None
        try:
            input_tensor = tf.convert_to_tensor(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB))[tf.newaxis, ...]
            detections = self.model(input_tensor)
        except Exception as e :
            self.reststream=1
            log_str="allocation issue"
            logging.info(log_str)
            requests.get(TRIGGER_URL_TEMPLATE.split('trigger')[0] + 'broadcast', params={'event': 'restart', 'message': f'{self.program}-{int(self.raceid[-2:])}'})
            return 0
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        for i in range(num_detections):
            bbox = [float(b) for b in detections['detection_boxes'][i]]
            xmin = bbox[1] * self.cols
            ymin = bbox[0] * self.rows
            xmax = bbox[3] * self.cols
            ymax= bbox[2] * self.rows
            cx=xmin+(xmax-xmin)/2
            cy=ymin+(ymax-ymin)/2
            h=ymax-ymin
            w=xmax-xmin
            cls_id=detections['detection_classes'][i]-1 # labels index start from 1
            label=self.names[cls_id]
            score=detections['detection_scores'][i]
            if(score>self.objects_confidence[cls_id]):
                self.pre_obj.append([cls_id,score,xmin,ymin,xmax,ymax,label,cx,cy,h,w])
        return 1
  
    def detect_debug(self):
        detections=self.model.loc[self.model ['filename'] == self.frame_counter].values
        for obj in detections:
            label= obj[1]
            if(label=='back'):
                label='loading'
            idx=obj[2]
            score= obj[3]
            xmin = obj[4]
            ymin = obj[5]
            xmax = obj[6]
            ymax = obj[7]
            cx=xmin+((xmax-xmin)/2)
            cy=ymax+((ymax-ymin)/2)
            h=ymax-ymin
            w=xmax-xmin
            if(score>self.objects_confidence[idx]):
                self.pre_obj.append([idx,score,xmin,ymin,xmax,ymax,label,cx,cy,h,w])
    
    def detect_pattern(self,patten,roi):
        res = cv2.matchTemplate( self.frame_gray[roi[0]:roi[1],roi[2]:roi[3]],patten,5)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if(max_val>.8):
            return 1
        else:
            return 0      
      
    def trigger(self):
        if((self.AboutTimeFrame)<  200):
            return 0
        if(len(self.front_trk.finals)>0 and self.flag_man_flag and self.detected_horses<1 and self.out_horses<4 and self.activate_flage):
            self.trigger_type="Flag"
            self.delay_to_trigger+=20
        elif(len(self.loading_trk.finals)>0):# loading
            if(self.out_horses==0 and max(len(self.horse_trk.finals),self.detected_horses)==0):
                self.delay_to_trigger+=self.Ld_delay[0]
                self.trigger_type="L0"
            elif(self.out_horses<=1 and max(len(self.horse_trk.finals),self.detected_horses)==0):
                self.trigger_type="L1"
                self.delay_to_trigger+=self.Ld_delay[1]
            elif(self.activate_bl and (self.out_horses<=1  and max(len(self.horse_trk.finals),self.detected_horses)==1 and len(self.active_loading)!=0)):
                self.trigger_type="L11"
                self.delay_to_trigger+=self.Ld_delay[2]
            elif(not self.back and (self.out_horses<=1  and max(len(self.horse_trk.finals),self.detected_horses)==1 and len(self.active_loading)!=0)):
                self.trigger_type="L11"
                self.delay_to_trigger+=self.Ld_delay[2]
            elif(self.delay_to_trigger>0 ):
                self.delay_to_trigger-=1
        
        elif(len(self.front_trk.finals)>0):
            if(self.out_horses==0 and self.detected_horses==0):
                self.trigger_type="F0"
                self.delay_to_trigger+=self.Fn_delay[0]
            elif(self.out_horses==1 and self.detected_horses==0):
                self.trigger_type="F1"
                self.delay_to_trigger+=self.Fn_delay[1]
            elif(self.delay_to_trigger>=10 and self.out_horses>(self.nh/2)):
                self.delay_to_trigger-=10
          
        elif(self.view=='T1' and self.out_horses<5 and self.detected_horses==0):
            self.trigger_type="T1"
            self.delay_to_trigger+=20
        
        elif(self.prev_trigger_type=='neutral' and self.view=='front' and self.out_horses<2):
             self.trigger_type="Br"
             self.delay_to_trigger+=91
        self.prev_trigger_type=self.view
      
        if(utl.isitblank(self.frame_gray)):
            if(self.out_horses<4):
                self.trigger_type="Bl"
                self.delay_to_trigger+=20
            elif(self.delay_to_trigger>=20):
                self.delay_to_trigger-=20
                
                
        
        if(self.detected_horses==0 and self.out_horses<(self.nh*.7) and self.detect_pattern(self.start_pattern,self.start_roi)):
            self.trigger_type="Timer"
            self.trigger_time[-1]="Timer"
            return 1
        if(self.detect_pattern(self.end_pattern,self.end_roi)): # to fix mix races problem at the beginning of the stream
            self.trigger_type="M"
            self.trigger_time[-1]="M"
            return 1
            
        if(self.delay_to_trigger>90):
            self.trigger_time[-1]=self.trigger_type
            return 1
        else:
            return 0
     
    def send_trigger(self):
        if self.test_mode:
            #print('Trigger disabled')
            return
        raceno = int(self.raceid[-2:])
        before = False
        url = TRIGGER_URL_TEMPLATE.format(program=self.program, raceno=raceno, before="&before=1" if before else "") + f'&property={self.trigger_type}'
        print(url)
        print(requests.get(url))
    
    def load_model(self):
        gpu_idx=fg.pick_gpu()
        if(gpu_idx==-1):
            log_str="no gpu available"
            logging.info(log_str)
            return 0
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            tf.config.experimental.set_visible_devices(gpus[gpu_idx], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_idx], True)
            PATH_TO_SAVED_MODEL = os.path.join(os.path.dirname(__file__), '..', 'models', self.modelname)
            print('Loading model...', end='')
            start_time = time.time()
            self.model= tf.saved_model.load(PATH_TO_SAVED_MODEL)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print('Took {} seconds to load {} model'.format(elapsed_time,self.modelname))
            # allocate resorces for gpu
            gpuInitiateimage=cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'models', "gpuInitiateimage.jpg"))
            input_tensor = tf.convert_to_tensor(gpuInitiateimage)[tf.newaxis, ...]
            self.model(input_tensor)
        except Exception as e :
            self.reststream=1
            log_str="allocation issue"
            logging.info(log_str)
            requests.get(TRIGGER_URL_TEMPLATE.split('trigger')[0] + 'broadcast', params={'event': 'restart', 'message': f'{self.program}-{int(self.raceid[-2:])}'})
            return 2
        
        return 1

    async def iter_frames(self):
        if self.in_filename:
            while self.cap.isOpened():
                ret, self.frame = self.cap.read()
                if (ret == False):
                    if(self.gate_counter<30 and self.trigger_type=='M'):
                        self.trigger_type="Jump"
                        self.trigger_time[-1]="Jump"
                    break
                yield ret
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            while True:
                await asyncio.sleep(0.001)          # workaround to let other tasks run too
                raw_img = self.in_buffer.buffer.read(self.rows * self.cols * 3)
                if(self.stop_signal):
                    self.trigger_type="S3"
                    self.trigger_time[-1]=self.trigger_type
                    yield False
                    break
                if not raw_img:
                    self.missframe_counter+=1
                    #self.cap = cv2.VideoCapture(self.in_filename)
                    self.frame = np.zeros((self.rows, self.cols,3), np.uint8)
                    if(self.missframe_counter==1):
                        log_str="the buffer is empty {}".format(int(self.frame_counter))
                        logging.info(log_str)
                    # save temp file
                    tmp_result=os.path.join(os.path.dirname(__file__), '..', "tmp", self.raceid+".csv")
                    holdme=[self.imminent,self.out_horses]
                    pd.DataFrame(holdme).T.to_csv(tmp_result,header=["imminent","nh"],index = False)
                    if (self.out_horses<4):
                        self.trigger_type="S1"
                        self.trigger_time[-1]=self.trigger_type
                        self.send_trigger()
                        yield False
                        break
                    # buffer didn't receive any frame for 1 min (fps*delay time in second)
                    if(self.missframe_counter>(1500)):
                        self.trigger_type="S2"
                        self.trigger_time[-1]=self.trigger_type
                        self.send_trigger()
                        yield False
                        break
                else:
                    self.missframe_counter=0
                    img = np.frombuffer(raw_img, dtype='uint8')
                    self.frame = img.reshape((self.rows, self.cols, 3))
                yield True

    async def main(self):
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(self.dst+"/"+self.raceid+'.log')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        self.initialize_myque()
        self.load_track_pattern()
        self.set_track_param()
        model_loaded=0
        correct_track=0
        self.frame_counter=0
        break_flage=0
        out=None
        if(self.vis):
            out = cv2.VideoWriter(self.dst+"/"+self.raceid+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (self.cols,self.rows))

        try:
            async for ret in self.iter_frames():
                start_time = time.time()
                self.frame_counter+=1.0
                if(self.frame_counter%self.frame_interval!=0):
                    continue
                if(ret==False):
                    if(self.gate_counter<30 and self.trigger_type=='M'):
                        self.trigger_type="Jump"
                        self.trigger_time[-1]=self.trigger_type
                    if(self.vis):
                        out.release()
                    return 1
                if(not self.debug and not model_loaded ):
                        model_loaded=self.load_model()
                        if(model_loaded==2):
                            return 2 # allocation problem
                        if(model_loaded==0):
                            continue
                self.frame_gray=cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                self.fog+=utl.scene_haze(self.frame_gray)
                if(self.debug and self.aboutsignal==-1):
                    self.imminent=True
                if(self.trigger_time[-2]=="-1" and self.detect_pattern(self.about_pattern,self.about_roi)and correct_track):
                    self.trigger_time[-2]=self.frame_counter/30
                if(self.imminent or self.trigger_time[-2]!="-1"):
                    self.AboutTimeFrame+=self.frame_interval
                if(self.abcounter>self.about_thresh and self.abflag and self.gate_counter<30 and self.trigger_type=='M'):
                        if(self.debug):
                            sec_float=(self.length-self.frame_counter)/self.fps
                            str_time="{:.3f}".format(sec_float)
                        else:
                            time_sec=(self.frame_counter/30)
                            str_time="{}.{}".format(str(int(time_sec/60)).zfill(2),str(int(time_sec%60)).zfill(2))
                        self.trigger_type='Ab'
                        self.trigger_time[-3]=self.gate_counter
                        self.trigger_time[2]=str_time
                        self.trigger_time[-1]=self.trigger_type
                        self.send_trigger()
                        break
                  
                    
                if(correct_track==0):
                    state_track=self.right_tracks()
                    if(state_track==-1):
                        print(self.track, "is new")
                        correct_track=1
                    elif(state_track==1):
                        correct_track=1
                    if(self.frame_counter>3600):# after two min stop check the logo
                        correct_track=1
                else:
                    if(self.debug):
                        self.detect_debug()
                    elif(not self.detect()):
                        return 2 # allocation problem
                    self.cam_transition=self.SBD()
                    self.post_process()
                    if(self.vis):
                        self.vis_img=self.frame.copy()
                        self.disply_objects()
                        self.disply_tracks(tracks=self.full_trk)
                        self.disply_tracks(tracks=self.horse_trk)
                        self.disply_info(trigger=break_flage)
                        out.write(self.vis_img)
                        if(self.vis==2):
                            cv2.imshow("vis",self.vis_img)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                if(self.vis):
                                    out.release()
                                break
                    if (break_flage==0 and self.trigger()):
                        img_id="{}_{:08d}".format(self.raceid,int(self.frame_counter))
                        img_path="{}/{}.jpg".format(self.dst,img_id)
                        cv2.imwrite(img_path,self.frame)
                        if(self.debug):
                            sec_float=(self.length-self.frame_counter)/self.fps
                            str_time="{:.3f}".format(sec_float)
                        else:
                            time_sec=(self.frame_counter/25)
                            str_time="{}.{}".format(str(int(time_sec/60)).zfill(2),str(int(time_sec%60)).zfill(2))
                        if(self.gate_counter<30 and self.trigger_type=='M'):
                            self.trigger_type='NG'
                        self.trigger_type="v1"+self.trigger_type
                        if(self.fog>300):
                            self.trigger_type+="-Fog"
                        self.trigger_time[2]=str_time
                        self.trigger_time[-1]=self.trigger_type
                        self.trigger_time[-3]=self.gate_counter
                        print( self.trigger_time)
                        break_flage=1
                        self.send_trigger()
                        break
                self.rest_value()
                ps=(time.time() - start_time)
                #if(ps>0.0):
                    #print(1.0/ps,end='\r')
            if(self.vis):
                out.release() 
            if(not self.debug):
                log_str="{}".format( self.trigger_time)
                logging.info(log_str)
            else:
                self.triger_info.append(self.trigger_time)
            print( self.trigger_time)
            return  self.trigger_time
        except:
            logging.error('Unhandled exception', exc_info=True)
            raise



if __name__ == '__main__':
    WIN=1
    def get_list_and_nh(infile_csv):
        videos=pd.read_csv(infile_csv,delimiter=',').astype(str).iloc[:, 1].values+'.mp4'
        nhs=pd.read_csv(infile_csv,delimiter=',').iloc[:, 4].values
        aboutsignals=pd.read_csv(infile_csv,delimiter=',').iloc[:, -2].values
        triggers=pd.read_csv(infile_csv,delimiter=',').iloc[:, 2].values
        return videos,nhs,aboutsignals,triggers
    
    video_dir=r"/media/hedi/mydrive/media/FR"
    detection_dir=r"/media/hedi/mydrive/detection_file/FRT/FRT-d0-AWS3"
    #detection_dir=r"/media/hedi/mydrive/detection_file/FRT/FRT-D0-19SEP22"
    # /media/hedi/mydrive/detection_file/FRT-D0-19SEP22 #new mdoel
    dst_path=r"/media/hedi/mydrive/media/frtdebug/"
    if(WIN):
        video_dir=r"D:\Butterfly\Videos\FR"
        detection_dir=r"D:\Butterfly\detectionFiles\FRT\FRT-d0-AWS3"
        dst_path="result/"
    csv_dir = "045resultx.csv"
    videos,nhs,aboutsignals,triggers=get_list_and_nh(infile_csv=csv_dir)
    trackids =["045"]
    triger_info=[]
    for vid_name,nh,xtrigger,about in zip(videos[:],nhs[:],triggers[:],aboutsignals[:]):
        trackid=vid_name[-9:-6]
        if(trackid in trackids):
            dst=dst_path+str(trackid)
            #print(vid_name)
            video_file="{}/{}/{}".format(video_dir,str(trackid),vid_name)
            detection_file="{}/{}.csv".format(detection_dir,vid_name[:-4])
            if(not os.path.isfile(video_file)):
                print(video_file, "is not exist")
                continue
            if(not os.path.isfile(detection_file)):
                print(vid_name[:-4], "can not find")
                continue
            if( not cv2.VideoCapture(video_file).isOpened()):
                print(video_file, "bad video")
                continue
            objects_pd = pd.read_csv(detection_file)
            ap = FranceThoroughbredBase(in_filename=video_file, dst=dst, model=objects_pd ,debug=1,triger_info=triger_info,xtrigger=xtrigger,aboutsignal=about, frame_interval=3.0, nh=nh, vis=0,
                              in_buffer=sys.stdin, width=640, height=480, raceid=str(trackid), program='QVD', test_mode=1)
            loop = asyncio.get_event_loop()
            model_task = loop.create_task(ap.main())
            ws_task = None
            loop.run_until_complete(model_task)
            if ws_task is not None:
                ws_task.cancel()
    utl.write_to_csv(info=triger_info,name=str(trackids[0])+"result",header=["track","raceid","trigger","Xtrigger",'nh','gate','about','property'])   
            #print(model_task.result)
            
            

