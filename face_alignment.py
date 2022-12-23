from retinaface import RetinaFace 
import os  
import cv2

def run():
    
    src_path = "./img_align_celeba"
    tar_path = "./img_processing"
    
    training_img_num = 50
    cnt = 0
    for r, d ,f in os.walk(src_path):
        
        for file in f :
            img_path = os.path.join(r,file)
            faces = RetinaFace.extract_faces(img_path,align=True)
            
            if len(faces) != 1:
                continue
            
            img = cv2.resize(faces[0],(320,320))
            cv2.imwrite(f"{tar_path}/{file}",img)
            cnt+=1
            if cnt >= training_img_num:
                break
        
if __name__ == "__main__":
    run()