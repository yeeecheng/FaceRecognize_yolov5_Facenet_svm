import subprocess
import dataset_processing 
import face_alignment




def run():
    
    #face_detect.run()
    #dataset_processing.run()

    #subprocess.run("python3 ./yolov5/train.py --img 320 --batch 16 --epoch 1 --data ./dataset.yaml --weight yolov5s.pt --cfg ./yolov5/models/yolov5s.yaml".split(" "))
    #subprocess.run("python3 ./yolov5/export.py --weight ./yolov5/runs/train/exp/weights/best.pt --data ./dataset.yaml --img 320 --include edgetpu".split(" "))
    subprocess.run("python3 ./yolov5/detect.py --weight ./yolov5/runs/train/exp/weights/best-int8_edgetpu.tflite --source ./dataset/images/val/025803.jpg".split(" "))
    
    



if __name__ =="__main__":
    run()