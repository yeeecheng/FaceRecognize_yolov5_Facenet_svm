import argparse
import os
import platform
import sys
from pathlib import Path

import pickle
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

import numpy as np
import cv2
from PIL import Image
from deepface.commons import functions, realtime, distance as dst
from tensorflow.keras.preprocessing import image as IMG
from deepface import DeepFace

@smart_inference_mode()
def run(
        weights=ROOT/"yolov5/runs/train/exp2/weights/best-int8_edgetpu.tflite",  # model path or triton URL
        source=0,  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'dataset.yaml',  # dataset.yaml path
        imgsz=(160, 160),  # inference size (height, width)
        save= ROOT/"./face_detect_result.txt",
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        svc=ROOT/"SVCmodel.pkl",
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    with open(svc,'rb') as f:
        SVCmodel ,class_name = pickle.load(f)
    
    emb_model = DeepFace.build_model("Facenet")
    
     # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
 

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    
    
   
    for path, im, im0s, vid_cap, s in dataset:
        
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)


        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # ******************************
            image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB))
            image = np.array(image)
            img_size = np.array(image.shape)[0:2]
            # ********************************
            
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    

                    if save_img or save_crop or view_img:  # Add bbox to image
                        
                        x1 = np.maximum(int(xyxy[0]) - 16, 0)
                        y1 = np.maximum(int(xyxy[1]) - 16, 0)
                        x2 = np.minimum(int(xyxy[2]) + 16, img_size[1])
                        y2 = np.minimum(int(xyxy[3]) + 16, img_size[0])
                        img = image[y1:y2, x1:x2]
                    

                        img =cv2.resize(img ,(160,160))
                        img =IMG.img_to_array(img)
                        img = np.expand_dims(img,axis=0)
                        img /=255
                        img = functions.normalize_input(img = img ,normalization = 'base')
                        
                        embedding =emb_model.predict(img,verbose=0)[0].tolist()
                        embedding = np.expand_dims(embedding, axis=0)
                        # 利用SVM对人脸特征进行分类
                        predictions = SVCmodel.predict_proba(embedding)
                        best_class_indices = np.argmax(predictions, axis=1)
                        tmp_label = class_name[best_class_indices]
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        
                        if type(tmp_label)!=str:
                            tmp_label=tmp_label[0]
                        # print(best_class_probabilities)
                        print(class_name, predictions)

                        if best_class_probabilities < 0.95:
                            tmp_label = "others"
                       
                        print(tmp_label)
                        # ***************************************************
                        
                        with open(save,"w") as res:
                            res.write(tmp_label)
      

    ## Print time (inference-only)
        LOGGER.info(f"{dt[1].dt * 1E3:.1f}ms")

   
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5/runs/train/exp2/weights/best-int8_edgetpu.tflite', help='model path or triton URL')
    parser.add_argument('--source', type=str, default="0", help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'dataset.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[160], help='inference size h,w')
    parser.add_argument('--save', type=str ,default=ROOT/"./face_detect_result.txt")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--svc',type=str,default=ROOT/'svc_path')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
