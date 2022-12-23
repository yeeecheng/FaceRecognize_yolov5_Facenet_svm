import os
import cv2

def run():
    
    src_path ="./img_processing"
    tar_path ="./dataset"
    if not os.path.isdir(tar_path):
        os.mkdir(tar_path)
        os.mkdir(f"{tar_path}/images")
        os.mkdir(f"{tar_path}/labels")
        os.mkdir(f"{tar_path}/images/train")
        os.mkdir(f"{tar_path}/images/val")
        os.mkdir(f"{tar_path}/labels/train")
        os.mkdir(f"{tar_path}/labels/val")
    
    
    for r , d ,f in os.walk(src_path):
        split_percent = 0.9
        train_num = int(len(f)*split_percent)
        print(f"total num :{len(f)} , train num :{train_num} , split percent : {split_percent}")
        
        for num,file in enumerate(f):
            img_path = os.path.join(r,file)
            img = cv2.imread(img_path)
            img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            where = "train"
            if num >= train_num:
                where = "val"
            dataset_image_path = os.path.join(tar_path,"images",where,file)
            dataset_label_path = os.path.join(tar_path,"labels",where,file.split(".")[0]+".txt")
            #print(dataset_image_path)
            #print(dataset_label_path)
            cv2.imwrite(dataset_image_path,img)
            h , w ,d = img.shape
            x_center = float(w)/2.0 * float(1.0/w)
            y_center = float(h)/2.0 *float(1.0/h)
            
            with open(dataset_label_path,"w") as f:
                f.write(f"0 {x_center} {y_center} 1 1\n")
            
   

if __name__ == "__main__":
    run()