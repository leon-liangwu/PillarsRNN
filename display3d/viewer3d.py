import os
import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import cv2

from kitti import Object3d, Calibration


def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line, initmode=0) for line in lines]
    return objects

def read_results(results):
    objects = [Object3d(result, initmode=4) for result in results]
    return objects



def reg_list_to_objects(reg_list, mode=1):
    objects = [Object3d(reg, initmode=mode) for reg in reg_list]
    return objects


def read_pcdobjs(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line, initmode=2) for line in lines]
    return objects

# -----------------------------------------------------------------------------------------

def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


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


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.
    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P=None):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    # if np.any(corners_3d[2, :] < 0.1):
    #     corners_2d = None
    #     return corners_2d, np.transpose(corners_3d)
    # project the 3d bounding box into the image plane
    if P is None:
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


def compute_orientation_3d(obj, P=None):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        object orientation vector into the image plane.
        Returns:
            orientation_2d: (2,2) array in left image coord.
            orientation_3d: (2,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)
    # orientation in object coordinate system
    orientation_3d = np.array([[0.0, obj.l], [0, 0], [0, 0]])
    # rotate and translate in camera coordinate system, project in image
    orientation_3d = np.dot(R, orientation_3d)
    orientation_3d[0, :] = orientation_3d[0, :] + obj.t[0]
    orientation_3d[1, :] = orientation_3d[1, :] + obj.t[1]
    orientation_3d[2, :] = orientation_3d[2, :] + obj.t[2]

    if P is None:
        orientation_2d = None
        orientation_3d = np.transpose(orientation_3d)
        orientation_3d[:, 1] = orientation_3d[:, 1] - obj.h
        return orientation_2d, orientation_3d

    # vector behind image plane?
    if np.any(orientation_3d[2, :] < 0.1):
        orientation_2d = None
        return orientation_2d, np.transpose(orientation_3d)
    # project orientation into the image plane

    orientation_2d = project_to_image(np.transpose(orientation_3d), P)
    return orientation_2d, np.transpose(orientation_3d)


# -----------------------------------------------------------------------------------------

def create_bbox_mesh(p3d, gt_boxes3d, color):
    b = gt_boxes3d
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]], color)
        i, j = k + 4, (k + 1) % 4 + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]], color)
        i, j = k, k + 4
        p3d.add_line([b[i, 0], b[i, 1], b[i, 2]], [b[j, 0], b[j, 1], b[j, 2]], color)


class plot3d(object):
    def __init__(self, det=False):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        self.view.getViewport()

        coord = gl.GLAxisItem()
        glLineWidth(3)
        coord.setSize(3, 3, 3)
        self.view.addItem(coord)
        self.view.orbit(135, -10)
        self.view.opts['distance'] = 20
        self.view.pan(8, 0, 1.5)
        self.det = det

    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=1, color=colors)
        self.view.addItem(points_item)

    def add_line(self, p1, p2, color=None):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])

        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines', color = color, width=2, antialias=True)

        self.view.addItem(lines_item)

    def show(self):
        self.view.show()
        self.app.exec_()


def show_lidar_with_boxes(pc_velo, objects, cls_list=None, calib=None, det=False):
    p3d = plot3d(det)
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    p3d.add_points(points, pc_color)
    for i, obj in enumerate(objects):

        if obj.type == 'Car':
            color = (1, 0, 0, 1)
        if obj.type == 'Pedestrian':
            color = (0, 0, 1, 1)
        if obj.type == 'Cyclist':
            color = (0, 1, 0, 1)


        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)

        create_bbox_mesh(p3d, box3d_pts_3d_velo, color)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]

        p3d.add_line([x1, y1, z1], [x2, y2, z2], color)
    p3d.show()


def show_lidar_with_boxes_nocalib(pc_velo, objects, cls_list=None, det=False):
    p3d = plot3d(det)
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    p3d.add_points(points, pc_color)
    for i, obj in enumerate(objects):
        if obj.type == 'DontCare': continue
        if cls_list is not None:
            cls_id = cls_list[i]
        else:
            cls_id = 0
        if cls_id == 0:
            color = (1, 0, 0, 1)
        else:
            color = (0, 1, 0, 1)
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj)
        #print '------------- one object ------------'
        #print box3d_pts_3d
        box3d_pts_3d[:, 0] = -box3d_pts_3d[:, 0]
        box3d_pts_3d = np.c_[box3d_pts_3d[:, 2], box3d_pts_3d[:, 0], box3d_pts_3d[:, 1]]
        create_bbox_mesh(p3d, box3d_pts_3d, color)

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj)
        ori3d_pts_3d[:, 0] = -ori3d_pts_3d[:, 0]
        ori3d_pts_3d = np.c_[ori3d_pts_3d[:, 2], ori3d_pts_3d[:, 0], ori3d_pts_3d[:, 1]]
        x1, y1, z1 = ori3d_pts_3d[0, :]
        x2, y2, z2 = ori3d_pts_3d[1, :]
        p3d.add_line([x1, y1, z1], [x2, y2, z2])
    p3d.show()


def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte - minimum) / ((maximum - minimum)+0.0001)
    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    b[...] = 1
    r[...] = 1
    g[...] = 1
    return np.stack([r, g, b, np.ones_like(r)]).transpose()


def show_gt_cuboid(lidar_data, img_path, cls_list=None):
    if img_path[-1] == 'n':
        label_path = img_path.replace('velodyne','label_2')
        label_path = label_path.replace('bin', 'txt')
        calib_path = label_path.replace('label_2','calib')
        print(label_path)
        print(calib_path)

        calib = Calibration(calib_path)
        objects = read_label(label_path)
        show_lidar_with_boxes(lidar_data, objects, cls_list, calib)
    else:
        label_path = img_path.replace('pcds', 'txts')
        label_path = label_path + '.txt'

        objects = read_pcdobjs(label_path)
        show_lidar_with_boxes_nocalib(lidar_data, objects, cls_list)


def show_pillar_cuboid(lidar_data, img_path, results, cls_list=None, id=''):
    label_path = img_path.replace('velodyne','label_2')
    label_path = label_path.replace('bin', 'txt')
    calib_path = label_path.replace('label_2','calib')

    calib = Calibration(calib_path)
    objects = read_results(results)
    

    png_path = img_path.replace('velodyne', 'image_2')
    png_path = png_path.replace('bin', 'png')
    
    png = cv2.imread(png_path)

    for obj in objects:
        # if obj.type != 'Car':
        #     continue

        # Draw 3d bounding box

        if obj.type == 'Car':
            color = (0, 0, 255)
        if obj.type == 'Pedestrian':
            color = (255, 0, 0)
        if obj.type == 'Cyclist':
            color = (0, 255, 0)

        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            pts_2d = box3d_pts_2d.astype(np.int32)
            cv2.line(png, tuple(pts_2d[0]), tuple(pts_2d[1]), color, 2)
            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 2)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 2)
                i, j = k, k + 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 2)



    cv2.imshow('png_data', png)
    cv2.waitKey(1)
    save_path = './draws/'+ id + '.png'
    print(save_path)
    cv2.imwrite(save_path, png)

    show_lidar_with_boxes(lidar_data, objects, cls_list, calib)
    objects = []
    show_lidar_with_boxes(lidar_data, objects, cls_list, calib)




def show_det_cuboid(lidar_data, img_path, reg_list, cls_list):
    if img_path[-1] == 'n':
        label_path = img_path.replace('velodyne', 'label_2')
        label_path = label_path.replace('bin', 'txt')
        calib_path = label_path.replace('label_2', 'calib')

        objects = reg_list_to_objects(reg_list)
        calib = Calibration(calib_path)
        show_lidar_with_boxes(lidar_data, objects, cls_list, calib, True)
    else:
        objects = reg_list_to_objects(reg_list, 3)
        show_lidar_with_boxes_nocalib(lidar_data, objects, cls_list, True)


def compute_box_direction(corners):
    corners_plain = corners[:4, :]
    center = np.mean(corners_plain, axis=0)
    vect_dir = corners_plain[0] - corners_plain[3]
    point = center + vect_dir

    return center, point


def show_lidar_box3d(pc_velo, corners_3d, cls_list=None, withcolor=False):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    if withcolor:
        pc_color[:, :3] = (pc_velo[:, [6, 5, 4]] + 1.0) * 0.5

    p3d.add_points(points, pc_color)
    for i, corners in enumerate(corners_3d):
        if cls_list is not None or len(cls_list):
            cls_id = cls_list[i]
        else:
            cls_id = 0

        if cls_id == 0:
            color = (1, 0, 0, 1)
        else:
            color = (0, 1, 0, 1)

        # color = (0, 1, 0, 1)
        # Draw 3d bounding box
        box3d_pts_3d_velo = corners

        create_bbox_mesh(p3d, box3d_pts_3d_velo, color)
        p1, p2 = compute_box_direction(corners)

        p3d.add_line(p1, p2, color)
    p3d.show()


def show_cloud_points(pc_velo):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    pc_color[:, :3] = (pc_velo[:, [6, 5, 4]] + 1.0) * 0.5

    p3d.add_points(points, pc_color)
    p3d.show()


def lidar_to_image(corners_3d, calib):
    box_list = []
    for corners in corners_3d:
        corners_2d = calib.project_velo_to_image(corners)
        box_list.append(corners_2d)

    box_list = np.array(box_list)
    return box_list


def lidar_to_camera(pts_velo, calib):
    pts_ref = calib.project_velo_to_rect(pts_velo)

    return pts_ref


def get_det_img_box(reg_list, img_path):
    label_path = img_path.replace('velodyne', 'label_2')
    label_path = label_path.replace('bin', 'txt')
    calib_path = label_path.replace('label_2', 'calib')

    objects = reg_list_to_objects(reg_list)
    calib = Calibration(calib_path)

    png_path = img_path.replace('velodyne', 'image_2')
    png_path = png_path.replace('bin', 'png')

    png = cv2.imread(png_path)
    (h, w, c) = png.shape
    box_list = np.zeros((reg_list.shape[0], 4))
    for i, obj in enumerate(objects):
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            minx = box3d_pts_2d[:, 0].min()
            miny = box3d_pts_2d[:, 1].min()
            maxx = box3d_pts_2d[:, 0].max()
            maxy = box3d_pts_2d[:, 1].max()
            box_list[i, 0] = minx if minx > 0 else 0
            box_list[i, 1] = miny if miny > 0 else 0
            box_list[i, 2] = maxx if maxx < w else w-1
            box_list[i, 3] = maxy if maxy < h else h-1
        else:
            box_list[i, :] = 0

    return box_list


def show_det_img(reg_list, img_path):
    label_path = img_path.replace('velodyne', 'label_2')
    label_path = label_path.replace('bin', 'txt')
    calib_path = label_path.replace('label_2', 'calib')

    objects = reg_list_to_objects(reg_list)
    calib = Calibration(calib_path)

    png_path = img_path.replace('velodyne', 'image_2')
    png_path = png_path.replace('bin', 'png')

    png = cv2.imread(png_path)
    for obj in objects:
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            pts_2d = box3d_pts_2d.astype(np.int32)
            cv2.line(png, tuple(pts_2d[0]), tuple(pts_2d[1]), (0, 0, 255, 2))
            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))
                i, j = k, k + 4
                cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))

    cv2.imshow('det png_data', png)
    cv2.waitKey(10)


def show_img(scan_data, reg_list, img_path):
    if img_path[-1] == 'n':
        label_path = img_path.replace('velodyne', 'label_2')
        label_path = label_path.replace('bin', 'txt')
        calib_path = label_path.replace('label_2', 'calib')
        png_path = img_path.replace('velodyne', 'image_2')
        png_path = png_path.replace('bin', 'png')
        calib = Calibration(calib_path)
        objects = read_label(label_path)
        objects_det = reg_list_to_objects(reg_list)
        #objects[0].print_object()
        png = cv2.imread(png_path)
        # for obj in objects:
        #     if obj.type != 'Car':
        #         continue
        #
        #     # Draw 3d bounding box
        #     box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        #     if box3d_pts_2d is not None:
        #         pts_2d = box3d_pts_2d.astype(np.int32)
        #         cv2.line(png, tuple(pts_2d[0]), tuple(pts_2d[1]), (0, 0, 255, 2))
        #         for k in range(0, 4):
        #             i, j = k, (k + 1) % 4
        #             cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))
        #             i, j = k + 4, (k + 1) % 4 + 4
        #             cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))
        #             i, j = k, k + 4
        #             cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (0, 0, 255, 2))

        for obj in objects_det:
            if obj.type != 'Car':
                continue

            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
            if box3d_pts_2d is not None:
                pts_2d = box3d_pts_2d.astype(np.int32)
                cv2.line(png, tuple(pts_2d[0]), tuple(pts_2d[1]), (255, 255, 255, 2))
                for k in range(0, 4):
                    i, j = k, (k + 1) % 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 2))
                    i, j = k + 4, (k + 1) % 4 + 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 2))
                    i, j = k, k + 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 2))


        cv2.imshow('png_data', png)
        cv2.waitKey(1)
        #for corners in corners_list:

    else:
        pass


def get_box_list(box3d_image, img):
    (h, w, c) = img.shape
    box_list = np.zeros((box3d_image.shape[0], 4))
    for i, box3d_pts_2d in enumerate(box3d_image):

        if box3d_pts_2d is not None:
            minx = box3d_pts_2d[:, 0].min()
            miny = box3d_pts_2d[:, 1].min()
            maxx = box3d_pts_2d[:, 0].max()
            maxy = box3d_pts_2d[:, 1].max()
            box_list[i, 0] = minx if minx > 0 else 0
            box_list[i, 1] = miny if miny > 0 else 0
            box_list[i, 2] = maxx if maxx < w else w - 1
            box_list[i, 3] = maxy if maxy < h else h - 1
        else:
            box_list[i, :] = 0

    return box_list


def show_det_img(img_path, box_list):
    png_path = img_path.replace('velodyne', 'image_2')
    png_path = png_path.replace('bin', 'png')

    png = cv2.imread(png_path)

    for box in box_list:
        png = cv2.rectangle(png, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

    cv2.imshow('bbox png', png)
    cv2.waitKey(1)


def show_image_box3d(img_path, box_list, showbox=True):
    png_path = img_path.replace('velodyne', 'image_2')
    png_path = png_path.replace('bin', 'png')

    png = cv2.imread(png_path)

    if showbox:
        if box_list is not None and len(box_list) > 0:
            pts_2ds = box_list.astype(np.int32)
            for pts_2d in pts_2ds:
                cv2.line(png, tuple(pts_2d[0]), tuple(pts_2d[1]), (255, 255, 255, 1))
                for k in range(0, 4):
                    i, j = k, (k + 1) % 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 1))
                    i, j = k + 4, (k + 1) % 4 + 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 1))
                    i, j = k, k + 4
                    cv2.line(png, tuple(pts_2d[i]), tuple(pts_2d[j]), (255, 255, 255, 1))

    cv2.imshow('bbox png', png)
    cv2.waitKey(1)



# -----------------------------------------------------------------------------------------

# if __name__ == '__main__':
#     dataset = kitti_object('./', 'training')
#     indexes = (3, 4, 5)
#     for index in indexes:
#         data_idx = index
#         # PC
#         lidar_data = dataset.get_lidar(data_idx)
#         # OBJECTS
#         objects = dataset.get_label_objects(data_idx)
#         objects[0].print_object()
#         # CALIB
#         calib = dataset.get_calibration(data_idx)
#         print(calib.P)
#         # Show
#         show_lidar_with_boxes(lidar_data, objects, calib)
