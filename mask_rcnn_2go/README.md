# MaskRCNN2GO Model

Owner: Peizhao Zhang (stzpz@fb.com)

Model: MaskRCNN2GO (bbox + segmentation) float32 and int8

Datasets used for evaluation: COCO 2014 minival

Input:
  * data (1, 3, H, W), min(H, W) = 320, BGR in range [0, 255]
  * im_info (1, 3) [scaled_height, scaled_width, scale]

Proposals (pre nms, post nms): 3000/100

Accuracy(mAP[IoU=0.50:0.95]):
   Model  | bbox | segmentation
  ------- | ---- | ------------
  float32 | 25.1 | 21.6
  int8    | 24.8 | 21.7

Evaluation:
* Download COCO 2014 minival dataset
* Update path in run_eval.sh
* Run run_eval.sh

Model source: f93520960, f96110081:934610037

## Acknowledgement

Thanks a lot for the help from Carole-Jean Wu, Fei Sun and Yanghan Wang.
