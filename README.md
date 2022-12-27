# Face Recognize

## 註冊新的使用者

```cmd
python3 logIn_usr.py --root <根目錄> --label <label_name> --num <想要拍攝的數量>
```

## 開啟client

```cmd
python3 client.py --root <根目錄> --data <要訓練資料集的路徑 --ip <連線ip addr>  --port <連線port>
```

## 重新train人臉偵測

```cmd
python3 face_alignment.py --root <根目錄> --train_dataset <欲訊練的資料集> --save <儲存的位置> --num <訓練的數量> --split <切的比例>
```
