#from future import print_function
import sys
import os
import argparse
from numpy import number
import numpy
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import COCO_ROOT, COCO_CLASSES as labelmap
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import COCO_CLASSES, COCOAnnotationTransform, COCODetection
import torch.utils.data as data
from ssd import build_ssd
from data.coco_eval import CocoEvaluator
from data.coco_utils import get_coco_api_from_dataset, coco_to_excel
import logging
import json

COCO_change_category = ['0','1','2','3','4','5','6','7','8','9','10','11','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','31','32','33','34','35','36','37','38','39','40',
'41','42','43','44','46','47','48','49','50','51','52','53','54','55','56','57','58','59',
'60','61','62','63','64','65','67','70','72','73','74','75','76','77','78','79','80','81',
'82','84','85','86','87','88','89','90']

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/COCO.pth',
type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.25, type=float,
help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
help='Use cuda to train model')
parser.add_argument('--coco_root', default=COCO_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder + 'result.json'
    num_images = len(testset)
    #num_images = 100
    coco_results = []    
    
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]])        

        # ii -> category id        
        for ii in range(detections.size(1)):
            j = 0
            while detections[0, ii, j, 0] >= thresh:

                score = detections[0, ii, j, 0].cpu().data.numpy()
                pt = (detections[0, ii, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])

                # standard format of coco ->
                # [{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236},{...},...]                
                coco_results.extend(
                    [
                        {
                            "image_id": testset.pull_anno(i)[0]['image_id'],
                            "category_id": int(COCO_change_category[ii]),
                            "bbox": [coords[0], coords[1], coords[2], coords[3]],
                            "score": float('%.2f' %(score)),
                        }                     
                    ]
                )                
                j+=1
        
    iou_type = 'bbox'
    result_file = filename
    logger = logging.getLogger("SSD.inference")
    logger.info('Writing results to {}...'.format(result_file))
    with open(filename, 'w') as f:
        json.dump(coco_results, f, cls=MyEncoder)
    from pycocotools.cocoeval import COCOeval
    coco_gt = testset.coco
    coco_dt = coco_gt.loadRes(result_file)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result_strings = []
    keys = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    metrics = {}
    for i, key in enumerate(keys):
        metrics[key] = coco_eval.stats[i]
        logger.info('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))
        result_strings.append('{:<10}: {}'.format(key, round(coco_eval.stats[i], 3)))

    result_path = os.path.join(save_folder, 'result_{:07d}.txt'.format(num_images))        
    with open(result_path, "w") as f:
        f.write('\n'.join(result_strings))

def test_voc():
    # load net
    #num_classes = 81 # change
    net = build_ssd('test', 300, 201) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = COCODetection(args.coco_root, 'trainval35k', None, COCOAnnotationTransform)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset, BaseTransform(net.size, (104, 117, 123)), thresh=args.visual_threshold)


if __name__ == '__main__':
    test_voc()                