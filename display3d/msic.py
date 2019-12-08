from __future__ import division, print_function
import numpy as np

from shapely.geometry import Polygon
import cv2

from collections import defaultdict

from kitti import Calibration


def camera_to_lidar(points, r_rect, velo2cam):
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = np.dot(points, np.linalg.inv(np.dot(r_rect, velo2cam).T))
    return lidar_points[..., :3]


def lidar_to_camera(points, r_rect, velo2cam):
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = np.dot(points, np.dot(r_rect, velo2cam).T)
    return camera_points[..., :3]


def box_lidar_to_camera(data, r_rect, velo2cam):
    xyz_lidar = data[:, 0:3]
    w, l, h = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)


def box_camera_to_lidar(data, r_rect, velo2cam):
    xyz = data[:, 0:3]
    l, h, w = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    r = data[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def cuboid_to_corners(cuboid):
    (cls_id, x, y, z, w, l, h, theta) = cuboid
    theta = (theta + np.pi / 2) # (theta + np.pi / 2)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    centre_x = x
    centre_y = y

    rear_left_x = centre_x - l / 2 * cos_t - w / 2 * sin_t
    rear_left_y = centre_y - l / 2 * sin_t + w / 2 * cos_t
    rear_right_x = centre_x - l / 2 * cos_t + w / 2 * sin_t
    rear_right_y = centre_y - l / 2 * sin_t - w / 2 * cos_t
    front_right_x = centre_x + l / 2 * cos_t + w / 2 * sin_t
    front_right_y = centre_y + l / 2 * sin_t - w / 2 * cos_t
    front_left_x = centre_x + l / 2 * cos_t - w / 2 * sin_t
    front_left_y = centre_y + l / 2 * sin_t + w / 2 * cos_t
    corners = np.array([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                        front_right_x, front_right_y, front_left_x, front_left_y]).reshape((4, 2))
    return corners


def get_corners_list(reg_list):
    corners_list = []
    for reg in reg_list:
        (prob, w, l, h, centre_x, centre_y, z, theta) = reg

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        rear_left_x = centre_x - l / 2 * cos_t - w / 2 * sin_t
        rear_left_y = centre_y - l / 2 * sin_t + w / 2 * cos_t
        rear_right_x = centre_x - l / 2 * cos_t + w / 2 * sin_t
        rear_right_y = centre_y - l / 2 * sin_t - w / 2 * cos_t
        front_right_x = centre_x + l / 2 * cos_t + w / 2 * sin_t
        front_right_y = centre_y + l / 2 * sin_t - w / 2 * cos_t
        front_left_x = centre_x + l / 2 * cos_t - w / 2 * sin_t
        front_left_y = centre_y + l / 2 * sin_t + w / 2 * cos_t
        corners = np.array([rear_left_x, rear_left_y, rear_right_x, rear_right_y,
                            front_right_x, front_right_y, front_left_x, front_left_y]).reshape((4, 2))

        corners_list.append(corners)

    return corners_list



def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def get_corners_3d(reg_list):
    corners_list = []
    for reg in reg_list:
        (prob, w, l, h, centre_x, centre_y, z, theta) = reg

        R = rotz(-theta-np.pi/2)

        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        # z_corners = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]

        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # print corners_3d.shape
        corners_3d[0, :] = corners_3d[0, :] + centre_x
        corners_3d[1, :] = corners_3d[1, :] + centre_y
        corners_3d[2, :] = corners_3d[2, :] + z

        corners_3d = corners_3d.transpose(1, 0)

        corners_list.append(corners_3d)

    corners_list = np.array(corners_list)

    return corners_list


def decode_output_box3d(prediction, rpn_mode=False, anchors=None):
    reg_list, cls_list = get_reg_list_rpn(prediction, anchors)
    corners_3d = get_corners_3d(reg_list)
    # corners_list = get_corners_list(reg_list)
    return corners_3d, reg_list, cls_list


def get_det_info(prediction, bev_data, img_path, rpn_mode=False, anchors=None):
    if not rpn_mode:
        reg_list, cls_list = get_reg_list(prediction)
    else:
        reg_list, cls_list = get_reg_list_rpn(prediction, anchors)

    calib_path = img_path.replace('velodyne', 'calib')
    calib_path = calib_path.replace('.bin', '.txt')
    calib = Calibration(calib_path)

    reg_list[:, [5, 6, 4]] = calib.project_velo_to_rect(reg_list[:, 4:7])
    reg_list[:, 5] *= -1

    corners_list = get_corners_list(reg_list)
    prob_list = []

    for i in range(len(reg_list)):
        prob_list.append(reg_list[i][0])
    return corners_list, reg_list, prob_list, cls_list



def convert_format(boxes_array):
    """

    :param array: an array of shape [# bboxs, 4, 2]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [Polygon([(box[i, 0], box[i, 1]) for i in range(4)]) for box in boxes_array]
    return np.array(polygons)


def compute_iou(box1, box2):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = box1.intersection(box2).area / box1.union(box2).area

    return iou




def merge_mini_batch(batch_list, _unused=False):
    batch_size = len(batch_list)
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in ['pillar']:
            print('pillar shape', elems[0].shape)
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'coords':
            coors = []
            for i, coor in enumerate(elems):
                print('coor shape', coor.shape)
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)),
                    mode='constant',
                    constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret
