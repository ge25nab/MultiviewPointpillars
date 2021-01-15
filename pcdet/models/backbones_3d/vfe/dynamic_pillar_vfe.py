import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from .vfe_template import VFETemplate
from mmdet3d.ops import DynamicScatter
from mmdet3d.ops import Voxelization

class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range,mode='max',norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01)):
        super().__init__(model_cfg=model_cfg)
        self.point_cloud_range = point_cloud_range
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]

        # self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 5 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0

        self.num_filters  = [num_point_features] + list(self.num_filters)
        pfn_layers = []
        # TODO: currently only support one PFNLayer

        for i in range(len(self.num_filters ) - 1):
            in_filters = self.num_filters [i]
            out_filters = self.num_filters [i + 1]
            if i > 0:
                in_filters *= 2
            _, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        
        self.pfn_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
        voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1))
        self.voxel_layer = Voxelization(**voxel_layer)

    def get_output_feature_dim(self):
        return self.num_filters[-1]
    
    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map the centers of voxels to its corresponding points.

        Args:
            pts_coors (torch.Tensor): The coordinates of each points, shape
                (M, 3), where M is the number of points.
            voxel_mean (torch.Tensor): The mean or aggreagated features of a
                voxel, shape (N, C), where N is the number of voxels.
            voxel_coors (torch.Tensor): The coordinates of each voxel.

        Returns:
            torch.Tensor: Corresponding voxel centers of each points, shape
                (M, C), where M is the numver of points.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = int(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = int(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        #generate mask of voxel
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        #input points [80373,5 batch_ind,x,y,z,intensity] 
        points = batch_dict['points']
        
        # generate point_coords
        coors = []
        voxel_features = []
        # points list len(points) = 6 points[0] [23570,4]
        # dynamic voxelization only provide a coors mapping
        for batch_id in range(points[:,0].max().int().item()+1):
            res = points[points[:,0].int()==batch_id][:,1:].contiguous()
            voxel_features.append(res)
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        #points [118824,4] caoncate batch size
        voxel_features = torch.cat(voxel_features,dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors = torch.cat(coors_batch, dim=0)

        #cluster_center
        # voxel_features [batch_ind,x,y,z,density] coors [batch_id,]
        voxel_mean, mean_coors = self.cluster_scatter(voxel_features, coors)
        points_mean = self.map_voxel_center_to_point(
            coors, voxel_mean, mean_coors)  # mean_coors [batch_id,z,y,x]
        # TODO: maybe also do cluster for reflectivity
        f_cluster = voxel_features[:, :3] - points_mean[:, :3]

        # f_cluster [batch_ind,x,y]
        #pillar center
        f_center = voxel_features.new_zeros(size=(voxel_features.size(0), 2))
        f_center[:, 0] = voxel_features[:, 0] - (coors[:, 3].type_as(voxel_features) * self.vx + self.x_offset)
        f_center[:, 1] = voxel_features[:, 1] - (coors[:, 2].type_as(voxel_features) * self.vy + self.y_offset)
        # f_center[:, 2] = voxel_features[:, 2] - (coors[:, 1].type_as(voxel_features) * self.vz + self.z_offset)
        
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = self.pfn_scatter(point_feats, coors)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        # features = features.squeeze()
        batch_dict['pillar_features'] = voxel_feats
        batch_dict['pillar_coords'] = voxel_coors # [24264,4 batch_ind,z,y,x]
        del batch_dict['voxel_coords']
        return batch_dict
