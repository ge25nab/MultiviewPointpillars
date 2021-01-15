import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vfe_template import VFETemplate
from ....utils import multi_view_utils 
import time

# Point-wise attention 
class PALayer(nn.Module):
    def __init__(self, dim_pa, reduction_pa):
        super(PALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_pa, dim_pa // reduction_pa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_pa // reduction_pa, dim_pa)
        )

    def forward(self, x):
        b, w, _ = x.size()
        y = torch.max(x, dim=2, keepdim=True)[0].view(b, w)
        out1 = self.fc(y).view(b, w, 1)
        return out1


# Feature-wise attention 
class CALayer(nn.Module):
    def __init__(self, dim_ca, reduction_ca):
        super(CALayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim_ca, dim_ca // reduction_ca),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ca // reduction_ca, dim_ca)
        )

    def forward(self, x):
        b, _, c = x.size()
        y = torch.max(x, dim=1, keepdim=True)[0].view(b, c)
        y = self.fc(y).view(b, 1, c)
        return y


class PointNet(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                use_norm=True,
                reduction_r=4):
        super().__init__()
        self.use_norm = use_norm
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        num_points = 30000
        self.pa = PALayer(num_points,  num_points // reduction_r)
        self.ca = CALayer(out_channels,  out_channels // reduction_r)
        self.sig = nn.Sigmoid()


    def forward(self, points_feature, points_mask):
        x = self.linear(points_feature)
        torch.backends.cudnn.enabled = False  # [2,20000,128]
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) 
        torch.backends.cudnn.enabled = True
        x = F.relu(x) * points_mask.unsqueeze(-1).expand(x.shape) 
        
        pa_weight = self.pa(x)
        ca_weight = self.ca(x)
        paca_weight = torch.mul(pa_weight, ca_weight)
        paca_normal_weight = self.sig(paca_weight)
        x = torch.mul(x, paca_normal_weight)
        
        return x

class SingleViewNet(nn.Module):
  """SingleViewNet.
     Bird view or Cylinderical view.
  """

  def __init__(self, grid_size = (512,512,1)):
    super(SingleViewNet, self).__init__()

    self.pointnet = PointNet(in_channels=64, out_channels=64)
    self.res1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= 1,stride=1, bias=False), 
                nn.BatchNorm2d(64, eps=1e-3, momentum=0.01), nn.ReLU())
    self.grid_size = [x for x in grid_size if x > 1]

  def forward(self, points_xyz, points_feature, points_mask, points_voxel):
    # torch.cuda.synchronize()
    # start = time.time()

    batch_size, _ ,_ = points_feature.shape
    points_feature_new = self.pointnet(points_feature, points_mask)  # (points_feature_new!=0).sum() 121w shape [2,2w,128] 
    voxels = multi_view_utils.batched_unsorted_segment_max(
        batched_data=points_feature_new,
        batched_segment_ids=points_voxel['indices'],
        num_segments=points_voxel['num_voxels'],
        batched_padding=points_voxel['paddings'])

    _, _, nc = voxels.shape
    voxels_in = torch.reshape(voxels, [batch_size] + self.grid_size + [nc]) # [4,432,496,128]      
    voxels_in = voxels_in.permute(0,3,1,2) 
    voxels= self.res1(voxels_in) # [4,128,432,496]

    points_out = multi_view_utils.bilinear_interpolate_torch(voxels.permute(0,3,2,1),points_voxel['voxel_xyz'][:, :, :2])  #points_out [4,2w,128]

    return points_out

class Attention2ViewPillarNet(nn.Module):
  """Pillar Net."""

  def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
#   norm_type='sync_batch_norm', act_type='relu',
#                stride=(2, 1, 2), up_stride=(1, 1, 2)):
    super(Attention2ViewPillarNet, self).__init__()

    self.xy_view_grid_size = (432, 496, 1)
    self.xy_view_grid_range_x = (0, 69.12)
    self.xy_view_grid_range_y = (-39.68, 39.68)
    self.xy_view_grid_range_z = (-3.0, 1.0)

    self.cylinder_view_grid_size = (2560, 100, 1)
    self.cylinder_view_grid_range_x = (-np.pi, np.pi)
    self.cylinder_view_grid_range_y = (-3.0, 1.0)
    self.cylinder_view_grid_range_z = (0.0, 69.12)

    self.num_filters = model_cfg.NUM_FILTERS
    self.xy_view = SingleViewNet(grid_size = self.xy_view_grid_size)
    self.cylinder_view = SingleViewNet(grid_size = self.cylinder_view_grid_size)

    self.pointnet1 = PointNet(in_channels = 45, out_channels = 64)
    self.pointnet2 = PointNet(in_channels = 64, out_channels = 64)
    self.pointnet3 = PointNet(in_channels = 64*3, out_channels = 64)

  def get_output_feature_dim(self):
        return self.num_filters[-1]
      
  def forward(self, batch_dict, **kwargs):
    # torch.cuda.synchronize()
    # start = time.time()
    del batch_dict['points']
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
    batch_dict['spatial_features'] = pillars
    return batch_dict
