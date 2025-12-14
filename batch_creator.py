import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import open3d as o3d
import json
import math

class PointCloudImageDataset(Dataset):
    def __init__(self, img_root, pc_root, rotation_root, split='train', val_ratio=0.1, test_ratio=0.1):
        self.img_root = img_root
        self.pc_root = pc_root
        self.rotation_root = rotation_root
        self.samples = []

        for category in os.listdir(img_root):
            cat_img_dir = os.path.join(img_root, category)
            if not os.path.isdir(cat_img_dir):
                continue
            for obj_prefix in set('_'.join(f.split('_')[:2]) for f in os.listdir(cat_img_dir)):
                img_files = sorted(glob.glob(os.path.join(cat_img_dir, obj_prefix + '_*.png')))
                pc_dir = os.path.join(pc_root, category, obj_prefix)
                pc_file = os.path.join(pc_dir, 'pcd_1024.ply')
                rot_file = os.path.join(rotation_root, category, obj_prefix + "_transforms.json")
                if not os.path.exists(pc_file):
                    continue
                for img_file in img_files:
                    self.samples.append((img_file, pc_file, rot_file))

        np.random.shuffle(self.samples)
        n = len(self.samples)
        val_n = int(n * val_ratio)
        test_n = int(n * test_ratio)
        if split == 'train':
            self.samples = self.samples[:n - val_n - test_n]
        elif split == 'val':
            self.samples = self.samples[n - val_n - test_n:n - test_n]
        elif split == 'test':
            self.samples = self.samples[n - test_n:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pc_path, rot_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = torch.tensor(np.array(img) / 255.0, dtype=torch.float32).permute(2, 0, 1)

        # Load point cloud (world coordinates)
        pcd1 = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd1.points)
        points = (points - np.min(points)) / (np.max(points) - np.min(points))

        # load transforms.json and find the matching frame for this image
        with open(rot_path, 'r') as f:
            data = json.load(f)

        img_filename = os.path.basename(img_path)
        frame = next((fr for fr in data['frames'] if fr['file_path'].endswith(img_filename[-7:])), None)
        if frame is None:
            raise ValueError(f"No transform found for image {img_filename} in {rot_path}")

        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)

        T_inv = np.linalg.inv(transform_matrix)

        # Convert point cloud to homogeneous coords
        pc_h = np.hstack([points, np.ones((points.shape[0], 1))])
 
        pc_cam_h = (T_inv @ pc_h.T).T

        pc_cam = pc_cam_h[:, :3]
        
        beta=math.pi/180*20
        viewmat=np.array([[np.cos(beta),0,-np.sin(beta)],[
			                0,1,0],[
                            np.sin(beta),0,np.cos(beta)]],dtype='float32')
        rotmat=transform_matrix[:3,:3].dot(np.linalg.inv(viewmat))

        pc_test =((points-[0.7,0.5,0.5])/0.4).dot(rotmat)+[1,0,0]

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points)
        pcd1.paint_uniform_color([1, 0, 0])  # Red color

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc_cam)
        pcd2.paint_uniform_color([0, 1, 0])  # Green color

        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(pc_test)
        pcd3.paint_uniform_color([0, 0, 1])  # Green color

        img.save("test.png")
        o3d.visualization.draw_geometries([pcd1, pcd3])
        pc_cam = torch.from_numpy(pc_cam)
        return img, pc_cam