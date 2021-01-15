import torch
from torch import nn
from torch.nn import functional as F
from ....utils import multi_view_utils 
import numpy as np
class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        self.res1 = BasicBlock(idims=128, odims=128, stride=2)
        self.res2 = BasicBlock(idims=128, odims=128, stride=2)
        self.res3 = BasicBlock(idims=128, odims=128, stride=2)
        # self.deconv2 = deconv2d(odims=128, stride=2)
        self.deconv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,padding =1, stride=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,padding =1, output_padding=1, stride=2)
        # self.deconv3 = deconv2d(odims=128, stride=4)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3,output_padding =1, stride=4)
        # self.conv = conv2d(odims=128, stride=1)
        self.conv = nn.Conv2d(in_channels=128*3, out_channels=256, kernel_size= 3,stride=1,padding =1,bias=False)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size= 3,stride=1,padding =1,bias=False)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)

    def forward(self, x):
        voxels_out1 = self.res1(x)
        voxels_out2 = self.res2(voxels_out1)
        voxels_out3 = self.res3(voxels_out2)
        voxels_out1 = self.deconv1(voxels_out1)
        voxels_out2 = self.deconv2(voxels_out2)
        voxels_out3 = self.deconv3(voxels_out3)
        voxels_out = torch.cat([voxels_out1, voxels_out2, voxels_out3], axis=1)
        x= self.conv(voxels_out)
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6

class IndentityLayer(nn.Module):
    def __init__(self):
        super(IndentityLayer, self).__init__()
    def forward(self, x):
        return x

class BasicBlock(nn.Module):
  """ResNet Basic Block."""

  def __init__(self, idims, odims, kernel_size=(3, 3), stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(idims, odims, kernel_size= kernel_size,stride=stride,padding =1, bias=False)
    self.bn1 = nn.BatchNorm2d(odims, eps=1e-3, momentum=0.01)
    self.relu1 = nn.ReLU()
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    return x

class PointNet(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-5, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
    def forward(self, points_feature, points_mask):
        x = self.linear(points_feature)
        torch.backends.cudnn.enabled = False  # [2,20000,128]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        torch.backends.cudnn.enabled = True
        x = F.relu(x) * points_mask.unsqueeze(-1).expand(x.shape) 
        return x

class SingleViewNet(nn.Module):
  """SingleViewNet.
     Bird view or Cylinderical view.
  """

  def __init__(self, grid_size = (512,512,1)):
    super(SingleViewNet, self).__init__()

    self.pointnet1 = PointNet(in_channels=64, out_channels=64)
    self.grid_size = [x for x in grid_size if x > 1]
    # self.backbone2d = BaseBEVBackbone()
    self.res1 = BasicBlock(idims=64, odims=64, stride=2)
    self.res2 = BasicBlock(idims=64, odims=64, stride=2)
    self.res3 = BasicBlock(idims=64, odims=64, stride=2)
    # self.deconv2 = deconv2d(odims=128, stride=2)
    self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,padding =1, stride=1)
    self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,padding =1, output_padding=1, stride=2)
    # self.deconv3 = deconv2d(odims=128, stride=4)
    self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3,output_padding =1, stride=4)
    # self.conv = conv2d(odims=128, stride=1)
    self.conv = nn.Conv2d(in_channels=64*3, out_channels=64, kernel_size= 3,stride=1,padding =1,bias=False)

  def forward(self, points_xyz, points_feature, points_mask, points_voxel):
    batch_size, _ ,_ = points_feature.shape
    points_feature_new = self.pointnet1(points_feature, points_mask)  # (points_feature_new!=0).sum() 121w shape [2,3w,64] 
    voxels = multi_view_utils.batched_unsorted_segment_max(
        batched_data=points_feature_new,
        batched_segment_ids=points_voxel['indices'],
        num_segments=points_voxel['num_voxels'],  
        batched_padding=points_voxel['paddings'])
    _, _, nc = voxels.shape
    voxels_in = torch.reshape(voxels, [batch_size] + self.grid_size + [nc]) # [1,432,496,64]      
    voxels_in = voxels_in.permute(0,3,1,2) 
    torch.backends.cudnn.enabled = False 
    voxels_out1 = self.res1(voxels_in)
    voxels_out2 = self.res2(voxels_out1)
    voxels_out3 = self.res3(voxels_out2)
    voxels_out1 = self.deconv1(voxels_out1)
    voxels_out2 = self.deconv2(voxels_out2)
    voxels_out3 = self.deconv3(voxels_out3)
    voxels_out = torch.cat([voxels_out1, voxels_out2, voxels_out3], axis=1)
    voxels= self.conv(voxels_out)

    torch.backends.cudnn.enabled = True
    points_out = multi_view_utils.bilinear_interpolate_torch(voxels.permute(0,3,2,1),points_voxel['voxel_xyz'][:, :, :2])  #points_out [4,3w,64]
    # points_out = self.pointnet2(points_out,points_mask)
    return points_out

class Multiview2Conv(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super(Multiview2Conv, self).__init__()

        self.xy_view_grid_size = (432,496, 1)
        self.xy_view_grid_range_x = (0, 69.12)
        self.xy_view_grid_range_y = (-39.68, 39.68)
        self.xy_view_grid_range_z = (-3.0, 1.0)

        self.cylinder_view_grid_size = (1280, 120, 1)
        self.cylinder_view_grid_range_x = (-np.pi, np.pi)
        self.cylinder_view_grid_range_y = (-3.0, 1.0)
        self.cylinder_view_grid_range_z = (0.0, 69.12)

        self.num_filters = model_cfg.NUM_FILTERS
        self.xy_view = SingleViewNet(grid_size = self.xy_view_grid_size)
        self.cylinder_view = SingleViewNet(grid_size = self.cylinder_view_grid_size)

        self.pointnet1 = PointNet(in_channels = 45, out_channels = 64)
        self.pointnet2 = PointNet(in_channels = 64, out_channels = 64)
        self.pointnet3 = PointNet(in_channels = 64*3, out_channels = 128)


        # self.backbone2d = BaseBEVBackbone()
        # self.fcn = BEVNet(in_features=128, num_filters=128)
    def get_output_feature_dim(self):
            return self.num_filters[-1]
        
    def forward(self, batch_dict, **kwargs):
        # torch.cuda.synchronize()
        # start = time.time()
        points_xyz = batch_dict['points_xyz']  # [2,2w,3]
        points_feature = batch_dict['points_feature']  # [2,2w]
        points_mask = batch_dict['points_mask']  # [2,2w]
        batch_size, num_points, _ = points_xyz.shape
        xy_view_voxels = multi_view_utils.points_to_voxels(points_xyz,
                                                    points_mask,
                                                    self.xy_view_grid_size,
                                                    self.xy_view_grid_range_x,
                                                    self.xy_view_grid_range_y,
                                                    self.xy_view_grid_range_z)
        xy_view_voxels_stats = multi_view_utils.points_to_voxels_stats(points_xyz,
                                                                xy_view_voxels)
        xy_view_points_xyz = points_xyz - xy_view_voxels['centers']

        points_cylinder = multi_view_utils.points_xyz_to_cylinder(points_xyz)
        cylinder_view_voxels = multi_view_utils.points_to_voxels(
            points_cylinder, points_mask, self.cylinder_view_grid_size,
            self.cylinder_view_grid_range_x,
            self.cylinder_view_grid_range_y,
            self.cylinder_view_grid_range_z)
        cylinder_view_voxels_stats = multi_view_utils.points_to_voxels_stats(
            points_cylinder, cylinder_view_voxels)
        cylinder_view_points = points_cylinder - cylinder_view_voxels['centers']

    #此处改为attention
        points_feature = torch.cat([
            points_xyz,  # [4,20000,3]                                                                           ok
            xy_view_points_xyz,  # [4,20000,3]                                                                   ok
            torch.reshape(xy_view_voxels['voxel_point_count'], [batch_size, num_points, 1]), # [4,20000,1]       ok
            xy_view_voxels_stats['centered_xyz'],  # [4,20000,3]                                                  ok 
            xy_view_voxels_stats['points_covariance'],  #[4,20000,9]                                             ok
            xy_view_voxels_stats['centroids'],  # [4,20000,3]                                                    ok
            points_cylinder,                                                                                    #ok
            cylinder_view_points,                                                                               #ok
            torch.reshape(cylinder_view_voxels['voxel_point_count'], [batch_size, num_points, 1]),               #ok
            cylinder_view_voxels_stats['centered_xyz'],                                                         #ok
            cylinder_view_voxels_stats['points_covariance'],                                                    #ok
            cylinder_view_voxels_stats['centroids'],                                                            #ok
            points_feature.unsqueeze(-1)], axis=-1)  # [4,20000,45]                                                ok

        x = self.pointnet1(points_feature, points_mask)  # [4,20000,128]         points_feature.requires_grad   False

        x_xy_view = self.xy_view(points_xyz,
                                    x,
                                    points_mask,
                                    xy_view_voxels)  # x_xy_view[4,20000,128]

        x_cylinder_view = self.cylinder_view(points_cylinder,
                                                x,
                                                points_mask,
                                                cylinder_view_voxels) #x_cylinder_view [4,2w,128]

        x_pointwise = self.pointnet2(x, points_mask)
        x = torch.cat([
            x_xy_view,
            x_cylinder_view,
            x_pointwise], axis=-1)
        x = self.pointnet3(x, points_mask)          #x [4,2w,128*3] --> [4,2w,128]

        pillars = multi_view_utils.batched_unsorted_segment_max(
            batched_data=x,
            batched_segment_ids=xy_view_voxels['indices'],
            num_segments=xy_view_voxels['num_voxels'],
            batched_padding=xy_view_voxels['paddings'])
        
        _, _, nc = pillars.shape     # pillars [4,214272,128]          

        nx, ny, nz = self.xy_view_grid_size
        pillars = torch.reshape(pillars, [batch_size, nx, ny, nz * nc]).permute(0,3,2,1) # [4,128,496,432]
        # batch_dict['spatial_features'] = pillars

        # x = self.backbone2d(batch_dict)
        batch_dict['spatial_features'] = pillars
        return batch_dict
