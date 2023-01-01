git clone https://github.com/ultralytics/yolov5.git
mv ./best-int8_edgetpu.tflite ./yolov5/
mv ./face_detect.py ./yolov5/
echo "names:
- face
nc: 1
train: $(pwd)/dataset/face_detect_dataset/images/train
val: $(pwd)/dataset/face_detect_dataset/images/val
" > dataset.yaml
