import mmcv
import numpy as np
import torch
import math
from pyquaternion import Quaternion
from mmdet.datasets.builder import PIPELINES
import cv2


@PIPELINES.register_module()
class LoadMultiViewMultiSweepImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        sweep_num=1,
        random_sweep=False,
        color_type="unchanged",
        file_client_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sweep_num = sweep_num
        self.random_sweep = random_sweep
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_image(self, img_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            img_bytes = self.file_client.get(img_filename)
            image = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except ConnectionError:
            image = mmcv.imread(img_filename, self.color_type)
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        results["filename"] = filename

        imgs = [self._load_image(name) for name in filename]

        sweeps_paths = results["cam_sweeps_paths"]
        sweeps_ids = results["cam_sweeps_id"]
        sweeps_time = results["cam_sweeps_time"]
        if self.random_sweep:
            random_num = np.random.randint(0, self.sweep_num)
            sweeps_paths = [_sweep[:random_num] for _sweep in sweeps_paths]
            sweeps_ids = [_sweep[:random_num] for _sweep in sweeps_ids]
        else:
            random_num = self.sweep_num

        sweeps_imgs = []
        for cam_idx in range(len(sweeps_paths)):
            sweeps_imgs.extend(
                [imgs[cam_idx]]
                + [self._load_image(name) for name in sweeps_paths[cam_idx]]
            )

        results["sweeps_paths"] = [
            [filename[_idx]] + sweeps_paths[_idx] for _idx in range(len(filename))
        ]
        results["sweeps_ids"] = np.stack([[0] + _id for _id in sweeps_ids], axis=-1)
        results["sweeps_time"] = np.stack(
            [[0] + _time for _time in sweeps_time], axis=-1
        )
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img"] = sweeps_imgs
        results["img_shape"] = [img.shape for img in sweeps_imgs]
        results["ori_shape"] = [img.shape for img in sweeps_imgs]
        # Set initial values for default meta_keys
        results["pad_shape"] = [img.shape for img in sweeps_imgs]
        results["pad_before_shape"] = [img.shape for img in sweeps_imgs]
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        # add sweep matrix to raw matrix
        results["lidar2img"] = [
            np.stack(
                [
                    results["lidar2img"][_idx],
                    *results["lidar2img_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2img"]))
        ]
        results["lidar2cam"] = [
            np.stack(
                [
                    results["lidar2cam"][_idx],
                    *results["lidar2cam_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2cam"]))
        ]
        results["cam_intrinsic"] = [
            np.stack(
                [
                    results["cam_intrinsic"][_idx],
                    *results["cam_sweeps_intrinsics"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["cam_intrinsic"]))
        ]
        results.pop("lidar2img_sweeps")
        results.pop("lidar2cam_sweeps")
        results.pop("cam_sweeps_intrinsics")

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class CustomPointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs = np.stack(results['img'])
        img_aug_matrix  = results['img_aug_matrix']
        post_rots = [torch.tensor(single_aug_matrix[:3, :3]).to(torch.float) for single_aug_matrix in img_aug_matrix]
        post_trans = torch.stack([torch.tensor(single_aug_matrix[:3, 3]).to(torch.float) for single_aug_matrix in img_aug_matrix])
        intrins = results['camera_intrinsics']
        depth_map_list = []
        
        for cid in range(len(imgs)):
            lidar2lidarego = torch.tensor(results['lidar2ego']).to(torch.float32)
            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(results['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)
            cam2camego = torch.tensor(results['camera2ego'][cid])

            camego2global = results['camego2global'][cid]

            cam2img = torch.tensor(intrins[cid]).to(torch.float32)
            
            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)

            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T.to(torch.float)) + lidar2img[:3, 3].to(torch.float).unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[1],
                                             imgs.shape[2])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        
        ##################################################################
        # global i
        # import cv2
        # for image_id in range(imgs.shape[0]):
        #     i+=1
        #     image = imgs[image_id]
        #     gt_depth_image = depth_map[image_id].numpy()
            
        #     gt_depth_image = np.expand_dims(gt_depth_image,2).repeat(3,2)
            
        #     #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
        #     im_color=cv2.applyColorMap(cv2.convertScaleAbs(gt_depth_image,alpha=15),cv2.COLORMAP_JET)
        #     #convert to mat png
        #     image[gt_depth_image>0] = im_color[gt_depth_image>0]
        #     im=Image.fromarray(np.uint8(image))
        #     #save image
        #     im.save('visualize_1/visualize_{}.png'.format(i))
        #################################################################

        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class LoadCameraParam(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        cam_nums=6,
        znear=1.0,
        zfar=50,
    ):
        self.cam_nums = cam_nums
        self.znear = znear
        self.zfar = zfar
    
    def _focal2fov(self, focal, pixel):
        # return 2 * math.atan(pixel / (2 * focal)) * (180 / np.pi) 
        return 2 * math.atan(pixel / (2 * focal))

    def _getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    def _getProjectionMatrixShift(self, znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        # the origin at center of image plane
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # shift the frame window due to the non-zero principle point offsets
        offset_x = cx - (width/2)
        offset_x = (offset_x/focal_x)*znear
        offset_y = cy - (height/2)
        offset_y = (offset_y/focal_y)*znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        # 用于从 w2c 中提取的 R 和 t，得到 平移和缩放后的 c2w
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """

        cam_params = []
        for cam_idx in range(self.cam_nums):
            height = results['pad_before_shape'][cam_idx][0]
            width = results['pad_before_shape'][cam_idx][1]

            #TODO
            if isinstance(results['cam_intrinsic'][cam_idx], list) or results['cam_intrinsic'][cam_idx].ndim >=3:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][0][1, 1], height)

                w2c = results['lidar2cam'][cam_idx][0]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = projmatrix_.transpose(0,1)

            else:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][1, 1], height)

                w2c = results['lidar2cam'][cam_idx]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = torch.tensor(projmatrix_, dtype=torch.float32).transpose(0,1)

            # # w2i
            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projmatrix_.unsqueeze(0))).squeeze(0)

            cam_pos = viewmatrix.inverse()[3, :3]
            cam_param = {'height': height, 
                         'width':width, 
                         'fovx':fovx, 
                         'fovy':fovy, 
                         'viewmatrix':viewmatrix, 
                         'projmatrix':full_proj_transform, 
                         'cam_pos':cam_pos}
            cam_params.append(cam_param)
        
        results['cam_params'] = cam_params

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadEgoCameraParam(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        cam_nums=6,
        znear=1.0,
        zfar=50,
    ):
        self.cam_nums = cam_nums
        self.znear = znear
        self.zfar = zfar
    
    def _focal2fov(self, focal, pixel):
        # return 2 * math.atan(pixel / (2 * focal)) * (180 / np.pi) 
        return 2 * math.atan(pixel / (2 * focal))

    def _getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P


    def _getProjectionMatrixShift(self, znear, zfar, focal_x, focal_y, cx, cy, width, height, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        # the origin at center of image plane
        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        # shift the frame window due to the non-zero principle point offsets
        offset_x = cx - (width/2)
        offset_x = (offset_x/focal_x)*znear
        offset_y = cy - (height/2)
        offset_y = (offset_y/focal_y)*znear

        top = top + offset_y
        left = left + offset_x
        right = right + offset_x
        bottom = bottom + offset_y

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def _getWorld2View2(self, R, t, translate=np.array([.0, .0, .0]), scale=1.0):
        # 用于从 w2c 中提取的 R 和 t，得到 平移和缩放后的 c2w
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return np.float32(Rt)


    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        ego2lidar = results['ego2lidar']
        lidar2ego = np.linalg.inv(ego2lidar)

        lidar2cam = np.stack(results['lidar2cam'])
        lidar2img = np.stack(results['lidar2img'])
        ego2cam = np.dot(lidar2cam, ego2lidar)
        ego2img2 = np.dot(lidar2img, ego2lidar)
 
        ego2img = np.stack([results['cam_intrinsic'][i] @ ego2cam[i] for i in range(len(ego2cam))])

        cam_params = []
        for cam_idx in range(self.cam_nums):
            height = results['pad_before_shape'][cam_idx][0]
            width = results['pad_before_shape'][cam_idx][1]

            ## TODO
            if isinstance(results['cam_intrinsic'][cam_idx], list) or results['cam_intrinsic'][cam_idx].ndim >=3:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][0][1, 1], height)

                w2c = ego2cam[cam_idx][0]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][0][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = torch.tensor(projmatrix_, dtype=torch.float32).transpose(0,1)

            else:
                fovx = self._focal2fov(results['cam_intrinsic'][cam_idx][0, 0], width)
                fovy = self._focal2fov(results['cam_intrinsic'][cam_idx][1, 1], height)

                w2c = ego2cam[cam_idx]
                viewmatrix = self._getWorld2View2(np.transpose(w2c[:3, :3]), w2c[:3, 3], translate=np.array([0.0, 0.0, 0.0]), scale=1.0)  
                viewmatrix = torch.tensor(viewmatrix, dtype=torch.float32).transpose(0, 1)
                
                projmatrix_ = self._getProjectionMatrixShift(self.znear, 
                                                             self.zfar, 
                                                             results['cam_intrinsic'][cam_idx][0, 0], 
                                                             results['cam_intrinsic'][cam_idx][1, 1], 
                                                             results['cam_intrinsic'][cam_idx][0, 2], 
                                                             results['cam_intrinsic'][cam_idx][1, 2], 
                                                             width, height, fovx, fovy)
                projmatrix_ = projmatrix_.transpose(0,1)

            # # w2i
            full_proj_transform = (viewmatrix.unsqueeze(0).bmm(projmatrix_.unsqueeze(0))).squeeze(0)
            # cam_pos = np.linalg.inv(viewmatrix)[3, :3]
            cam_pos = viewmatrix.inverse()[3, :3]

            cam_param = {'height': height, 
                         'width':width, 
                         'fovx':fovx, 
                         'fovy':fovy, 
                         'viewmatrix':viewmatrix, 
                         'projmatrix':full_proj_transform, 
                         'cam_pos':cam_pos
                         }
            cam_params.append(cam_param)
        
        results['cam_params'] = cam_params

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class RandomScaleImageMultiViewImageForGaussianPretrain(object):
    """Random scale the image
    Args:
        scales
    """

    def __init__(self, scales=[]):
        self.scales = scales
        assert len(self.scales) == 1


    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        rand_scale = np.random.choice(self.scales)

        y_size = [int(img.shape[0] * rand_scale) for img in results['img']]
        x_size = [int(img.shape[1] * rand_scale) for img in results['img']]
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= rand_scale
        scale_factor[1, 1] *= rand_scale
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx]), return_scale=False) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        cam_intrinsic = [scale_factor @ c2i for c2i in results['cam_intrinsic']]
        img_aug_matrix = [scale_factor for _ in results['lidar2img']]
        results['lidar2img'] = lidar2img
        results['cam_intrinsic'] = cam_intrinsic
        results['img_aug_matrix'] = img_aug_matrix
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        results['scale_factor'] = [rand_scale for _ in results['img']]
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.scales}, '
        return repr_str