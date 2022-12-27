import cv2
import os
import time
from retinaface import RetinaFace 
import argparse
from train_classify import *


# 獲取用戶的臉部照片
def log_in(opt):
    
    root = opt.root
    label = opt.label
    num = opt.num
    
    new_usr_dataset_path = os.path.join(root,"./dataset/new_usr_dataset")  
   
    cap = cv2.VideoCapture(0)
    cnt = 0
    
    print("start capture")
    
    while True :
        
        ret ,frame =  cap.read()
        if not ret :
            break 
        
        img_dir =os.path.join(new_usr_dataset_path,label)
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        img_path = os.path.join(img_dir,str(cnt)+".png")
        
        cv2.imwrite(img_path , frame)
        
        cnt+=1        
        time.sleep(0.5)
        if cnt >= num :
            break
        
    print("finish capture img")

# 照片處理
def alignment_img(root):
    
    new_usr_dataset_path = os.path.join(root,"./dataset/new_usr_dataset")  
    
    model = RetinaFace
    
    for subdir in os.listdir(new_usr_dataset_path):
        
        subdir_path = os.path.join(new_usr_dataset_path,subdir)
        
        for img in os.listdir(subdir_path):
            
            img_path = os.path.join(subdir_path,img)
          
            faces = model.extract_faces(img_path)
            if len(faces) != 1:
                os.remove(img_path)
                continue
            
            img = cv2.resize(faces[0],(160,160))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(img_path,img)

def main(opt):
    
    #log_in(opt)
    #alignment_img(opt.root)

    TC = train_classify(opt.root)
    TC.set_training_data()
    TC.train()
    
def parse_opt(known):
    
    ROOT = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type = str ,default =ROOT, help="where is root path")
    parser.add_argument('--label',type=str ,default="label",help="new people name")
    parser.add_argument("--num",type = int ,default = 100 ,help ="the number you want to train")

    return parser.parse_known_args()[0] if known else parser.parse_args()

def run(**kwargs):
    
    opt = parse_opt(True)
    for k ,v in kwargs.items():
        setattr(opt,k,v)
    
    main(opt)
    return opt
    

if __name__ =="__main__":
    
    opt = parse_opt(True)
    main(opt)