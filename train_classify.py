from deepface import DeepFace
from deepface.basemodels import Facenet
import cv2
from deepface.commons import functions, realtime, distance as dst
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from retinaface import RetinaFace 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import shutil

class train_classify:
    
    def __init__(self,opt):
        
        # 載入model
        self.embedding_model = DeepFace.build_model("Facenet")
        
        # the path where weight save 
        self.root = opt.root
        self.label = opt.label
        self.first = False

    def set_training_data(self):
        
        print("start set training data")
        
        # 讀取新的使用者資料
        npz_save_path =os.path.join(self.root ,"./faces-dataset.npz")
        # 讀取目前的dataset
        if not os.path.isfile(npz_save_path):
            np.savez(npz_save_path,emb =list(),label=list())
            self.first =True
            
        new_img_train ,new_label_train = self.load_dataset(os.path.join(self.root ,"./dataset/new_usr_dataset"),self.first)
     
        dataset = np.load(npz_save_path)
        # 取出 , 分別為臉部特徵和標籤
        emb_train,label_train = dataset["emb"] ,dataset["label"]
        
        embs =list()
        labels=list()
        # 取得臉部特徵 ,並加入
        for face in new_img_train:
            emb = self.get_embedding(face)
            embs.append(emb)
        embs = np.asarray(embs)

        if emb_train.shape[0] != 0:
            emb_train = np.append(emb_train,embs,axis=0)
        else :
            emb_train = embs

        # 加入新的標籤
        for label in new_label_train:
            labels.append(label)
            
        labels =np.asarray(labels)

        if label_train.shape[0] != 0:
            label_train = np.append(label_train,labels,axis=0)
        else :
            label_train = labels

        self.emb_train =emb_train
        self.label_train = label_train
        

        
        # 存成檔案,後續新增不需要全部使用者重用
        print("save the face vector")        
        np.savez_compressed(npz_save_path,emb = emb_train,label= label_train)
        print("end set training data")
        print("")
        
    def train(self):

        print("start train ")
        # normalize 
        nor =Normalizer()
        emd_norm_train = nor.transform(self.emb_train)

        label_encoder =LabelEncoder()
        # 把label mapping 
        label_encoder.fit(self.label_train)
        label_encode_train = label_encoder.transform(self.label_train)

        # fit model
        SVCmodel = SVC(kernel='linear', probability=True)
        SVCmodel.fit(emd_norm_train, label_encode_train)
        # predict
        predict_train = SVCmodel.predict(emd_norm_train)
        # score
        score_train = accuracy_score(label_encode_train, predict_train)
        print('Accuracy: train=%.3f' % (score_train*100))

        # save model
        print("save model")
        label_unique = np.unique(self.label_train)
        SVCmodel_save_path = os.path.join(self.root,"SVCmodel.pkl")
        with open(SVCmodel_save_path,"wb") as f:
            # 放model 和 class 種類
            pickle.dump((SVCmodel,label_unique),f)

        print("end train")
        print("")
        
        dir_path = os.path.join(self.root,"./dataset/new_usr_dataset/",self.label)
        print(dir_path)
        shutil.rmtree(dir_path)
        
    # 讀取資料
    def load_dataset(self,dir_path,first):
        
        X,Y = list(),list()

        
        ## 如果沒有用戶資料夾
        if len(os.listdir(dir_path)) == 1 and not first :
            print("There is no User's img")
            exit(0)

        for subdir in os.listdir(dir_path):
            
            # 不是第一次執行,就只轉換label
            if not first:
                if subdir != self.label:
                    continue
                
            path = os.path.join(dir_path,subdir)
            print(f"load {path}")
            
            # 如果資料夾裡面沒有照片
            if len(os.listdir(path)) == 0:
                print("There is no img")
                exit(0)
            
            faces = list()
            for img in os.listdir(path):
                img_path = os.path.join(path,img)
        
                face =cv2.imread(img_path)
                       
                faces.append(face)
            
            labels = [subdir for i in range(len(faces))]
            
            X.extend(faces)
            Y.extend(labels)
        
        return np.asarray(X) , np.asarray(Y)

    # 轉特徵
    def get_embedding(self,img):
        
        img =cv2.resize(img ,(160,160))
        img =image.img_to_array(img)
        img = np.expand_dims(img,axis=0)
        img /=255
        img = functions.normalize_input(img = img ,normalization = 'base')

        embedding = self.embedding_model.predict(img,verbose=0)[0].tolist()
        
        return embedding

    
    


   
