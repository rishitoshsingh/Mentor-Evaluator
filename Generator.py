import cv2
import tensorflow as tf
import numpy as np
import glob
import random

class VideoBatchGenerator(tf.keras.utils.Sequence):
    
    def __init__(self,
                 glob_pattern,
                 batch_size=32,
                 target_size=(224,224),
                 channel=1,
                 fpm=2):
        
        self.glob_pattern = glob_pattern
        self.batch_size = batch_size
        self.target_size = target_size
        self.channel = channel
        self.files = glob.glob(self.glob_pattern)        
        self.n = len(self.files)
        self.fpm = fpm # default fps = 24
        self._calculate_total_frames()
        self.next_index = 0
        self.cap = self._load_new_video(self.next_index)
        
        print('Found total {} videos'.format(len(self.files)))
    
    def _load_new_video(self, index):
        cap = cv2.VideoCapture(self.files[index])
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_skip = int(fps * 60 / self.fpm)
        return cap
        
    def __getitem__(self, index):
        
        frames = np.zeros((self.batch_size,) + self.target_size + (self.channel,))
        
        batch_i = 0
        frame_i = 0
        
        while True:
            grabbed, frame = self.cap.read()
            
            if not grabbed:
                self.cap.release()
                self.next_index = random.randrange(0, self.n, 1)
                self.next_index = (self.next_index + 1) % self.n
                self.cap = self._load_new_video(self.next_index)
            
            frame_i += 1
            if frame_i == 1 or frame_i % self.frame_skip == 0:
                frame = cv2.resize(frame, self.target_size)
                if self.channel == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame,2)
                frames[batch_i] = frame
                batch_i += 1
                
                if batch_i == self.batch_size:
                    break

        return frames
    
    def _calculate_total_frames(self):
        total_frames = 0
        for file in self.files:
            cap = cv2.VideoCapture(file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_skip = int(fps * 60 / self.fpm)
            total_frames += int(frames/frame_skip)
            
        self.total_frames = total_frames
    
    def __len__(self):
        return self.total_frames // self.batch_size
    
    def __del__(self):
        self.cap.release()