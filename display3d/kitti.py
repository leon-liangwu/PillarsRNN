from __future__ import division, print_function
import numpy as np


class kitti_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
        #
        self.lidar_dir = os.path.join('velodyne', self.split_dir)
        self.label_dir = os.path.join('label', self.split_dir)
        self.calib_dir = os.path.join('calib', self.split_dir)

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        return load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return Calibration(calib_filename)

    def get_label_objects(self, idx):
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return read_label(label_filename)


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, param, initmode = 0):
        if initmode == 0:
            label_file_line = param
            data = label_file_line.split(' ')
            data[1:] = [float(x) for x in data[1:]]

            # extract label, truncation, occlusion
            self.type = data[0]  # 'Car', 'Pedestrian', ...
            self.truncation = data[1]  # truncated pixel ratio [0..1]
            self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            self.alpha = data[3]  # object observation angle [-pi..pi]

            # extract 2d bounding box in 0-based coordinates
            self.xmin = data[4]  # left
            self.ymin = data[5]  # top
            self.xmax = data[6]  # right
            self.ymax = data[7]  # bottom
            self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

            # extract 3d bounding box information
            self.h = data[8]  # box height
            self.w = data[9]  # box width
            self.l = data[10]  # box length (in meters)
            self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
            self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        elif initmode == 1:
            reg = param
            w, l, h, x, y, z, yaw = [float(x) for x in reg[1:]]
            y = -y
            # z = -z
            #print 'obj: ', (w, l, h, x, y, z, yaw)

            self.type = 'Car'
            self.w, self.l, self.h = w, l, h
            self.t = (y, z, x)  # (y, z, x) #(y, z, x)
            self.ry = -(yaw + np.pi / 2)
            # self.type = 'Car'
            # self.w, self.l, self.h = reg[1:4]
            # self.t = (-reg[5], reg[6], reg[4]) #(y, z, x)
            # #self.t = (reg[4], reg[5] + self.w, reg[6] - self.l)
            # self.ry = -(reg[7] + np.pi/2)

        elif initmode == 2:
            label_file_line = param
            data = label_file_line.split(' ')
            x, y, z, l, w, h, yaw = [float(x) for x in data[1:]]
            y = -y
            #yaw = yaw - np.pi / 2

            #print 'obj pcd: ', (w, l, h, x, y, z, yaw)

            self.type = 'Car'
            self.w, self.l, self.h = w, l, h
            self.t = (y, z+self.h/2, x)
            #self.ry = yaw  + np.pi / 2
            self.ry = -(yaw + np.pi / 2)

        elif initmode == 3:
            reg = param
            w, l, h, x, y, z, yaw = [float(x) for x in reg[1:]]
            y = -y
            #print 'obj det: ', (w, l, h, x, y, z, yaw)

            self.type = 'Car'
            self.w, self.l, self.h = w, l, h
            self.t = (y, z+self.h/2, x)  # (y, z, x) #(y, z, x)
            self.ry = -(yaw + np.pi / 2)
        
        elif initmode == 4:

            # extract label, truncation, occlusion
            self.type = param['type']  # 'Car', 'Pedestrian', ...
            self.truncation = param['truncated']  # truncated pixel ratio [0..1]
            self.occlusion = int(param['occluded'])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
            self.alpha = param['alpha']  # object observation angle [-pi..pi]

            # extract 2d bounding box in 0-based coordinates
            self.xmin = param['bbox'][0]  # left
            self.ymin = param['bbox'][1]  # top
            self.xmax = param['bbox'][2]  # right
            self.ymax = param['bbox'][3]  # bottom
            self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

            # extract 3d bounding box information
            self.l = param['dimensions'][0]  # box height
            self.h = param['dimensions'][1]  # box width
            self.w = param['dimensions'][2]  # box length (in meters)
            self.t = (param['location'][0], param['location'][1], param['location'][2])  # location (x,y,z) in camera coord.
            self.ry = param['rotation_y']  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0], self.t[1], self.t[2], self.ry))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = self.inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam['R'], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

