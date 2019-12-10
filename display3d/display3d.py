import pickle
import os
import numpy as np

import viewer3d
from viewer3d import plot3d, inte_to_rgb, show_pillar_cuboid
from msic import get_corners_3d

from kitti import Object3d

car_th = 0.5
ped_th = 0.5

data_dir = '/data/Machine_Learning/ImageSet/KITTI/object/training/'

f = open('./results/car/step_296960/result.pkl', 'rb')   
res_cars = pickle.load(f)     
print(len(res_cars))
f.close()      

f = open('./results/ped/step_194880/result.pkl', 'rb')   
res_peds = pickle.load(f)     
print(len(res_peds))
f.close()     

f = open('./ImageSets/val.txt') 
ids = f.readlines()
f.close()
print(len(ids))

show_sets = [ '002565']


for i, id in enumerate(ids):
    id = id.replace('\n', '')
    # if id not in show_sets:
    #     continue
    pc_path =  os.path.join(data_dir,'velodyne', id+'.bin')
    print(pc_path)
    pc_velo = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
    print(pc_velo.shape)

    res_car = res_cars[i]
    res_ped = res_peds[i]
    results = []
    cls_list = []

    for j, score in enumerate(res_car['score']):
        if score > car_th:
            result = {}
            result['type'] = res_car['name'][j]
            result['alpha'] = res_car['alpha'][j]
            result['truncated'] = res_car['truncated'][j]
            result['occluded'] = res_car['occluded'][j]
            result['bbox'] = res_car['bbox'][j]
            result['dimensions'] = res_car['dimensions'][j]
            result['location'] = res_car['location'][j]
            result['rotation_y'] = res_car['rotation_y'][j]
            results.append(result)
            cls_list.append(result['type'])

    for j, score in enumerate(res_ped['score']):
        if score > ped_th:
            result = {}
            result['type'] = res_ped['name'][j]
            result['alpha'] = res_ped['alpha'][j]
            result['truncated'] = res_ped['truncated'][j]
            result['occluded'] = res_ped['occluded'][j]
            result['bbox'] = res_ped['bbox'][j]
            result['dimensions'] = res_ped['dimensions'][j]
            result['location'] = res_ped['location'][j]
            result['rotation_y'] = res_ped['rotation_y'][j]
            results.append(result)
            cls_list.append(result['type'])


    # p3d = plot3d()
    # points = pc_velo[:, 0:3]
    # pc_inte = pc_velo[:, 3]
    # pc_color = inte_to_rgb(pc_inte)
    # p3d.add_points(points, pc_color)
    # p3d.show()
    show_pillar_cuboid(pc_velo, pc_path, results, id=id)
