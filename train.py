import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import open3d as o3d
from PIL import Image
import cv2

from batch_creator import PointCloudImageDataset
from distance import ChamferLoss, repulsion_loss

class PCGen(nn.Module):
    def __init__(self):
        super().__init__()

        # input 3 x 192 x 256 image
        self.inputconv16 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # input 16 1 x 192 x 256 feature maps
        self.conv16 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)

        # input 16 1 x 192 x 256 feature maps
        self.stridedconv32 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        # input 32 1 x 96 x 128 feature maps
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        # input 32 1 x 96 x 128 feature maps
        self.stridedconv64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        # input 64 1 x 48 x 64 feature maps
        self.conv64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        # input 64 1 x 48 x 64 feature maps
        self.stridedconv128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        # input 128 1 x 24 x 32 feature maps
        self.conv128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        # input 128 1 x 24 x 32 feature maps
        self.stridedconv256 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=1)
        # input 256 1 x 12 x 16 feature maps
        self.conv256 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        # input 256 1 x 12 x 16 feature maps
        self.stridedconv512 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=1)
        # input 512 1 x 6 x 8 feature maps
        self.conv512 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # need to figure out that last conv layer
        self.finalconv512 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5, stride=2, padding=1)

        # pred fc: we want to predict `num_points` points (each with 3 coords)
        self.num_points = 1024

        # compute flattened feature size by running a dummy input through the conv stack
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 192, 256)
            x = F.relu(self.inputconv16(dummy))
            x = F.relu(self.conv16(x))
            x = F.relu(self.stridedconv32(x))
            x = F.relu(self.conv32(x))
            x = F.relu(self.conv32(x))
            x = F.relu(self.stridedconv64(x))
            x = F.relu(self.conv64(x))
            x = F.relu(self.conv64(x))
            x = F.relu(self.stridedconv128(x))
            x = F.relu(self.conv128(x))
            x = F.relu(self.conv128(x))
            x = F.relu(self.stridedconv256(x))
            x = F.relu(self.conv256(x))
            x = F.relu(self.conv256(x))
            x = F.relu(self.stridedconv512(x))
            x = F.relu(self.conv512(x))
            x = F.relu(self.conv512(x))
            x = F.relu(self.conv512(x))
            x = F.relu(self.finalconv512(x))
            flat_dim = x.view(1, -1).size(1)

        self.predfc = nn.Linear(in_features=flat_dim, out_features=self.num_points * 3)

    def forward(self, input):
        # model forward pass
        c1 = F.relu(self.inputconv16(input))
        c2 = F.relu(self.conv16(c1))
        sc3 = F.relu(self.stridedconv32(c2))
        c4 = F.relu(self.conv32(sc3))
        c5 = F.relu(self.conv32(c4))
        sc6 = F.relu(self.stridedconv64(c5))
        c7 = F.relu(self.conv64(sc6))
        c8 = F.relu(self.conv64(c7))
        sc9 = F.relu(self.stridedconv128(c8))
        c10 = F.relu(self.conv128(sc9))
        c11 = F.relu(self.conv128(c10))
        sc12 = F.relu(self.stridedconv256(c11))
        c13 = F.relu(self.conv256(sc12))
        c14 = F.relu(self.conv256(c13))
        sc15 = F.relu(self.stridedconv512(c14))
        c16 = F.relu(self.conv512(sc15))
        c17 = F.relu(self.conv512(c16))
        c18 = F.relu(self.conv512(c17))
        sc19 = self.finalconv512(c18)
        if (sc19.ndim == 3):
            sc19 = torch.unsqueeze(sc19, 0)
        flat = sc19.view(sc19.size(0), -1)
        raw = self.predfc(flat)
        # reshape to (batch, num_points, 3)
        output = raw.view(sc19.size(0), self.num_points, 3)

        # layers = []
        # layers.append(self.inputconv16)
        # layers.append(nn.ReLU())
        # layers.append(self.conv16)
        # layers.append(nn.ReLU())
        # layers.append(self.stridedconv32)
        # layers.append(nn.ReLU())
        # layers.append(self.conv32)
        # layers.append(nn.ReLU())
        # layers.append(self.conv32)
        # layers.append(nn.ReLU())

        print("model gen success!")
        return output

if __name__ == "__main__":
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)

    # print(f"Using {device} device")
    pcgen = PCGen()
    print(pcgen)
    params = list(pcgen.parameters())
    print(len(params))
    print(params[0].size())

    # try input
    pil_image = Image.open("img_data/anise/anise_001_000.png")
    pil_image = pil_image.convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(pil_image)
    print("shape:", image_tensor.shape)
    print(image_tensor.dtype)
    out = pcgen(image_tensor)
    print(out)

    # run training loop I guess
    batch_size = 32
    dataset = PointCloudImageDataset('img_data', 'pc_data', 'rotation_data', split='train')
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=16)
    print("loader complete!")

    num_batches = len(loader)
    optimizer = optim.Adam(pcgen.parameters(), lr=0.001)
    chamfer_loss = ChamferLoss()
    batch_n = 1
    for img_batch, points_batch in loader:
        # img, points = dataset[b]

        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(transformed_points)
        # img.save("test.png")
        # o3d.visualization.draw_geometries([pcd])
        optimizer.zero_grad()
        output = pcgen(img_batch)
        loss = chamfer_loss(output, points_batch)  + 0.1 * repulsion_loss(points_batch)
        loss.backward()
        optimizer.step()
        print(f"finished iteration {batch_n} of {num_batches}")
        batch_n += 1
        if (batch_n == 50):
            break

    print("done training!")
    # pcgen = PCGen()
    # pcgen.load_state_dict(torch.load('model_batched_properly_reg_loss.pth', weights_only=True))

    pcgen.eval()
    with torch.no_grad():
        input = Image.open("test.png").convert('RGB')
        tensor_input = transforms.ToTensor()
        tensor_input = tensor_input(input)
        out_pc = pcgen(tensor_input)
        pcd = o3d.geometry.PointCloud()
        pc_np = np.squeeze(out_pc.numpy(), axis=0)
        pcd.points = o3d.utility.Vector3dVector(pc_np)
        o3d.visualization.draw_geometries([pcd])

    PATH = "model_batched_properly_reg_loss.pth"
    torch.save(pcgen.state_dict(), PATH)