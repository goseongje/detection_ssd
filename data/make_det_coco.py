import json 
import cv2 
import os
import shutil
import random 

def split_data():
    scenes_path = '/SSD3/jumi/UAV-DT/UAV-benchmark-M/UAV-benchmark-M'
    train_set, val_set = [],[]
    count = 0
    for scene in os.listdir(scenes_path): #scene:M0101
        scene_path = os.path.join(scenes_path, scene)
        imgs = os.listdir(scene_path)
        len_scene = len(imgs)
        print(scene, len_scene)
        random.shuffle(imgs)
        unit = int((len_scene)/15)
        train_set = imgs[3*unit:]
        val_set = imgs[:3*unit]
        for trains in train_set:
            shutil.copy(os.path.join(scenes_path, scene, trains), os.path.join('/SSD3/jumi/UAV-DT','Train',scene+'_'+trains))
        for vals in val_set:
            shutil.copy(os.path.join(scenes_path, scene, vals), os.path.join('/SSD3/jumi/UAV-DT','Val',scene+'_'+vals))
        count += len_scene
    print('Train : ',len(os.listdir('/SSD3/jumi/UAV-DT/Train')))
    print('Test : ',len(os.listdir('/SSD3/jumi/UAV-DT/Val')))
    print('Whole : ',count)

def get_dict(phase):
    print(phase)
    imgs_path = os.path.join('/SSD3/jumi/UAV-DT/',phase)
    images = []
    annotations = []
    image_id = 0
    annot_id = 0
    
    for tr in os.listdir(imgs_path):
        tgt_scene, img_name = tr.split('_') #M0101,img000001.jpg
        
        img_num = int(img_name[3:9])

        cv2_img = cv2.imread(os.path.join('/SSD3/jumi/UAV-DT/',phase,tr))
        images.append({'date_captured':'2021',
                        'file_name':tr,
                        'id':image_id+1,
                        'url':"",
                        'height':cv2_img.shape[0],
                        'width':cv2_img.shape[1]}
                        )
        # 하나의 이미지에 대하여
        with open('/SSD3/jumi/UAV-DT/UAV-benchmark-MOTD_v1.0/UAV-benchmark-MOTD_v1.0/GT/'+tgt_scene+'_gt_whole.txt') as txt:
            annot_lines = txt.readlines()
            for line in annot_lines: 
                frame_idx, tgt_id, bbox_l, bbox_t, bbox_w, bbox_h, view, occ, object_id = line.split(',')
                object_id = object_id.replace('\n', '')
                
                if int(frame_idx) == img_num:
                    annotations.append({"segmentation" : [0.0, 3.0, 5.0, 7.0, 9.0, 10.0],
                                "area" : 10.0,
                                "iscrowd" : 0,
                                "image_id" : image_id+1,
                                "bbox" : [float(bbox_l),float(bbox_t),float(bbox_w),float(bbox_h)],
                                "category_id" : int(object_id),
                                "id": annot_id+1})
                    annot_id += 1
        image_id += 1
    
    return images, annotations

def get_whole_json(phase):
    info = {"year" : 2021,
        "version" : "1.0",
        "description" : "UAV-DT",
        "contributor" : "Jumi ",
        "url" : "http://",
        "date_created" : "2021"}
    licenses = [{"id": 1,
            "name": "Attribution-NonCommercial",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }]

    _type = "instances"

    categories = [{"id": 1, "name":"car", "supercategory": "vehicle"},
                  {"id": 2, "name":"truck", "supercategory": "vehicle"},
                  {"id": 3, "name":"bus", "supercategory": "vehicle"}]
    
    images,annotations = get_dict(phase)
    
    json_data = {"info":info, 
                "images":images,
                "licenses":licenses,
                "type":_type,
                "annotations" : annotations,
                "categories" : categories}
    with open("/SSD3/jumi/UAV-DT/annotations/instances_"+phase.lower()+"2017.json","w") as jsonfile:
        json.dump(json_data,jsonfile,sort_keys=True,indent=4)

if __name__ == '__main__':
    phases = ['Train','Val']
    for phase in phases:
        print(phase)
        get_whole_json(phase)

                


