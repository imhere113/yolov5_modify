# yolov5-7.0 onnx模型推理简化流程
import torch
import cv2
import numpy as np
from copy import deepcopy
import onnxruntime as ort
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox
from utils.segment.general import masks2segments, process_mask, scale_image
from utils.plots import colors

model_path = "/home/vipuser/yolov5/runs/train/exp7/weights/simple.onnx"
img_path = "/home/vipuser/yolov5/hat3.jpg"

# 数据预处理
img = cv2.imread(img_path)
print('ori_img:', img.shape)
ori_img = deepcopy(img)
img = letterbox(img, new_shape=(640, 640), stride=32, auto=True)[0]  # padded resize
img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC to CHW, BGR to RGB,contiguous
# img = torch.from_numpy(img).to("cuda:0")  # ndarray to tensor
# img = img.float()  # uint8 to fp32
img = img.astype(np.float32)
img = img / 255  # 0 - 255 to 0.0 - 1.0
if len(img.shape) == 3:
    img = img[None]  # expand for batch dim

# 加载模型
sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])  # 'CPUExecutionProvider'
input_name = sess.get_inputs()[0].name
output_name = [output.name for output in sess.get_outputs()]
# 推理
outputs = sess.run(output_name, {input_name: img})

pred, proto = outputs[0:2]
pred = torch.from_numpy(pred).to("cuda:0")
proto = torch.from_numpy(proto).to("cuda:0")
# NMS
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, max_det=1000, nm=32)
# 画图
print('resize_img:', img.shape)
det = pred[0]
masks = process_mask(proto[0], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC

# 画mask
img = torch.from_numpy(img).to("cuda:0")
if len(masks) == 0:
    dst_img = img[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255
colors = [colors(x, True) for x in det[:, 5]]
colors = torch.tensor(colors, device="cuda:0", dtype=torch.float32) / 255.0
colors = colors[:, None, None]  # shape(n,1,1,3)
masks = masks.unsqueeze(3)  # shape(n,h,w,1)
masks_color = masks * (colors * 0.5)  # shape(n,h,w,3)

inv_alph_masks = (1 - masks * 0.5).cumprod(0)  # shape(n,h,w,1)
mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)
dst_img = img[0].flip(dims=[0])  # flip channel
dst_img = dst_img.permute(1, 2, 0).contiguous()  # shape(h,w,3)
dst_img = dst_img * inv_alph_masks[-1] + mcs
im_mask = (dst_img * 255).byte().cpu().numpy()
dst_img = scale_image(dst_img.shape, im_mask, ori_img.shape)

# 画box label score
det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], ori_img.shape).round()  # rescale boxes to ori_img size
det_cpu = pred[0].cpu().numpy()
for i, item in enumerate(det_cpu):
    # 画box
    box_int = item[0:4].astype(np.int32)
    cv2.rectangle(dst_img, (box_int[0], box_int[1]), (box_int[2], box_int[3]), color=(0, 255, 0), thickness=1)
    # 画label
    label = item[5]
    score = item[4]
    org = (min(box_int[0], box_int[2]), min(box_int[1], box_int[3]) - 8)
    text = '{}|{:.2f}'.format(int(label), score)
    cv2.putText(dst_img, text, org=org, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0),
                thickness=1)

cv2.imshow('result', dst_img)
cv2.waitKey()
cv2.imwrite('out.jpg', dst_img)