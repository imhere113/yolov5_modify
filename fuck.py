# 导入所需的库
import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision.ops import batched_nms

mapping = [('head', (0, 255, 0)), ('hat', (255, 0, 0)), ('person', (0, 0, 255))]

# 定义 onnx 模型的文件名，图像的文件名，检测阈值和非极大抑制阈值
model_file = 'weights/simple.onnx'
image_file = './PartB_01575.jpg'
conf_thres = 0.5
iou_thres = 0.6

# 加载 onnx 模型
session = ort.InferenceSession(model_file)

# 读取图像并转换为 RGB 格式
image = cv2.imread(image_file)
image_resized = cv2.resize(image, (640, 640))
saved_image = image_resized.copy()
image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# 获取图像的宽度和高度
height, width = image.shape[:2]

anchors = [[10,13, 16,30, 33,23],
           [30,61, 62,45, 59,119],
           [116,90, 156,198, 373,326]]


image_resized = (image_resized.astype(np.float32) - 0) / 255
image_resized = np.stack([image_resized])

image_resized = np.transpose(image_resized, (0, 3, 1, 2))

input_name = session.get_inputs()[0].name
output_names = [output.name for output in session.get_outputs()]
outputs = session.run(output_names, {input_name: image_resized})

res_boxes = []
res_bboxes_idx = []
res_scores = []

for nl, output in enumerate(outputs):
    output = torch.from_numpy(output)
    output = 1 / (1 + np.exp(-output))
    batch, na, grid_h, grid_w, num_attr = output.shape
    num_cls = num_attr - 4 - 1
    grid_shape = 1, 3, grid_h, grid_w, 2  # grid shape
    stride = 640 / grid_h
    xy = output[..., :2]
    wh = output[..., 2:4]
    conf = output[..., 4:]
    yv, xv = torch.meshgrid([torch.arange(grid_w), torch.arange(grid_h)])
    grid = torch.stack((xv, yv), 2).expand(grid_shape) - 0.5
    anchor_grid = (torch.Tensor(anchors[nl])).view((1, 3, 1, 1, 2)).expand(grid_shape)
    xy = (xy * 2 + grid) * stride
    wh = (wh * 2) ** 2 * anchor_grid
    xy = xy.view(1, na, grid_h, grid_w, 1, 2).expand(1, na, grid_h, grid_w, num_cls, 2)
    wh = wh.view(1, na, grid_h, grid_w, 1, 2).expand(1, na, grid_h, grid_w, num_cls, 2)

    cls_probs = conf[..., 1:] * conf[..., 0:1]
    mask = cls_probs >= 0.3
    label_idx = torch.arange(0, num_cls).view(1, 1, 1, 1, num_cls).expand(1, na, grid_h, grid_w, num_cls).long()
    target_xy = xy[mask].view(-1, 2)
    target_wh = wh[mask].view(-1, 2)
    label_idx = label_idx[mask].view(-1)
    cls_score = cls_probs[mask].view(-1)
    bboxes = torch.cat([target_xy - target_wh / 2, target_xy + target_wh / 2], dim=-1)
    res_boxes.append(bboxes)
    res_bboxes_idx.append(label_idx)
    res_scores.append(cls_score)

res_boxes = torch.cat(res_boxes, dim=0)
res_bboxes_idx = torch.cat(res_bboxes_idx, dim=0)
res_scores = torch.cat(res_scores, dim=0)

mask = batched_nms(res_boxes, res_scores, res_bboxes_idx, 0.5)
res_boxes = res_boxes[mask]
res_bboxes_idx = res_bboxes_idx[mask]
res_scores = res_scores[mask]

for box, cls_idx, score in zip(res_boxes, res_bboxes_idx, res_scores):
    cls_name = mapping[cls_idx][0]
    cls_color = mapping[cls_idx][1]
    cv2.rectangle(saved_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), cls_color, 2)
    cv2.putText(saved_image, cls_name, (int(box[0]), int(box[1]) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls_color, 2)
    cv2.putText(saved_image, f'{score.item():.4f}', (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cls_color, 2)

cv2.imwrite('result.jpg', saved_image)
