# Face Recognize

## 注意事項
* best-int8_edgetpu.tflite 為人臉偵測的權重 imgsz 160
* face_detect.py 要放進yolov5
* 預設放入class - other , 

## Function using 

### log in new user

```cmd
python3 logIn_usr.py --root <根目錄> --label <label_name> --num <想要拍攝的數量>
```
### Detect Face 

 ```cmd
 python3 face_detect.py --weight <tfllie位置> --data <dataset.yaml位置> --svc <svc權重檔位置> --imgsz 160 --save <輸出結果位置>
 ```
## 以下為用yolov5 train

> 你會用到training_face_detect_yolov5
### 開啟client

```cmd
python3 client.py --root <根目錄> --data <要訓練資料集的路徑 --ip <連線ip addr>  --port <連線port>
```

### 重新train人臉偵測

```cmd
python3 face_alignment.py --root <根目錄> --train_dataset <練的資料集> --save <儲存的位置> --num <訓練的數量> --split <切的比例>
```
