import torch
import numpy as np
import functools
from torch_scatter import scatter_add,scatter_max,scatter_mean,scatter_min
def pad_or_trim_to(x, shape, pad_val=0):
  """Pad and slice x to the given shape.
  Args:
    x: A tensor.
    shape: The shape of the returned tensor.
    pad_val: An int or float used to pad x.
  Returns:
    'x' is padded with pad_val and sliced so that the result has the given
    shape.
  """
  max_num_points = shape[0]
  pad = shape - np.minimum(x.shape, shape)

  zeros = np.zeros_like(pad)
  x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
  return x[:max_num_points,:]

def sequence_mask(lengths, maxlen):
    if maxlen is None:
        maxlen = lengths.max()
    mask = [False]*maxlen
    mask[:lengths] = [True]*len(mask[:lengths])
    return mask

def points_to_voxels(points_xyz,
                     points_mask,
                     grid_size,
                     grid_range_x,
                     grid_range_y,
                     grid_range_z):
  """Mapping points to voxels."""

  batch_size, num_points, _ = points_xyz.shape
  voxel_size_x = (grid_range_x[1]-grid_range_x[0]) / grid_size[0]  #xy_view_grid_size = (512, 512, 1)
  voxel_size_y = (grid_range_y[1]-grid_range_y[0]) / grid_size[1]  #voxel size ~0.16
  voxel_size_z = (grid_range_z[1]-grid_range_z[0]) / grid_size[2]  #
  voxel_size = torch.Tensor([voxel_size_x, voxel_size_y, voxel_size_z]).to(points_xyz.device)
  num_voxels = grid_size[0] * grid_size[1] * grid_size[2]
  grid_offset = torch.Tensor([grid_range_x[0], grid_range_y[0], grid_range_z[0]]).to(points_xyz.device)
  points_xyz_new = points_xyz - grid_offset
  voxel_xyz = points_xyz_new / voxel_size  #[2,2w,3]/[3]
  voxel_coords = voxel_xyz.type(torch.int32)# [batch_size,num_points,3   x y z ]   
  grid_size = torch.Tensor(grid_size).type(torch.int32).to(points_xyz_new.device)
  zeros = torch.zeros_like(grid_size)
  voxel_padding = ((points_mask < 1.0) | ((voxel_coords >= grid_size) | (voxel_coords < zeros)).any(axis=-1))
  voxel_indices = raval_index(
      torch.reshape(voxel_coords, [batch_size * num_points, 3]), grid_size) #voxel_coords 每个点对应点voxel xyz 的index voxel_indicies 每个点对应voxel的总index   
  voxel_indices = torch.reshape(voxel_indices, [batch_size, num_points])
  voxel_indices = torch.where(voxel_padding,
                          torch.zeros_like(voxel_indices),       #对voxel_ind 进行mask，当points_mask 是0 或者 点在grid之外 或者 坐标为负（不在range范围内） ，他的坐标即为0
                          voxel_indices)
  voxel_centers = ((0.5 + voxel_coords) * voxel_size + grid_offset)    #voxel 中心的坐标
       
  voxel_coords = torch.where(torch.unsqueeze(voxel_padding, axis=-1),  
                          torch.zeros_like(voxel_coords),
                          voxel_coords)
  voxel_xyz = torch.where(torch.unsqueeze(voxel_padding, axis=-1),
                      torch.zeros_like(voxel_xyz),
                      voxel_xyz)

  points_per_voxel = batched_unsorted_segment_sum(
      batched_data=torch.ones((batch_size, num_points), dtype=torch.int32,device=voxel_coords.device),
      batched_segment_ids=voxel_indices,
      num_segments=num_voxels,
      batched_padding=voxel_padding)       #points_per_voxel torch.Size([2, 214272])

  num_valid_voxels = torch.sum(points_per_voxel,axis=1)
  voxel_point_count = torch.gather(points_per_voxel,
                                  index=voxel_indices,
                                  dim=1).type_as(voxel_xyz)
  output = {'coords': voxel_coords,
              'centers': voxel_centers,
              'indices': voxel_indices,
              'paddings': voxel_padding,
              'num_voxels': num_voxels,
              'grid_size': grid_size,
              'voxel_xyz': voxel_xyz,
              'voxel_point_count': voxel_point_count,
              'num_valid_voxels': num_valid_voxels,
              'points_per_voxel': points_per_voxel}
  return output

def raval_index(coords, dims):
    #dims [432,496,1]
    multiplier = torch.Tensor([dims[1]*dims[2],dims[2],1]).type(torch.int32).to(dims.device)
    indices = torch.sum(coords * multiplier, axis=1) #coords [N,3 xyz] 496*x+1*y+1*z --> [N,1] 每个点对应的index
    return indices

def points_to_voxels_stats(points_xyz, voxels):
  """Get additional features for points."""

  batch_size, num_points, _ = points_xyz.shape

  # Compute centroids of each voxel.
  voxel_centroids = batched_unsorted_segment_mean(
      batched_data=points_xyz,
      batched_segment_ids=voxels['indices'],
      num_segments=voxels['num_voxels'],
      batched_padding=voxels['paddings'])
  point_centroids = torch.gather(voxel_centroids, index=voxels['indices'].unsqueeze(-1).expand(batch_size,num_points,3), dim=1)
  points_xyz_center = points_xyz - point_centroids  

  points_outer_prod = (points_xyz_center.unsqueeze(-1)*points_xyz_center.unsqueeze(-2))
  points_outer_prod = torch.reshape(points_outer_prod, [batch_size, num_points, 9])
  voxel_covariance = batched_unsorted_segment_mean(
      batched_data=points_outer_prod,
      batched_segment_ids=voxels['indices'],
      num_segments=voxels['num_voxels'],
      batched_padding=voxels['paddings'])
  points_covariance = torch.gather(voxel_covariance,
                                index = voxels['indices'].unsqueeze(-1).expand(batch_size,num_points,9),
                                dim=1)

  output = {'centroids': point_centroids,
            'centered_xyz': points_xyz_center,
            'points_covariance': points_covariance}

  return output


def points_xyz_to_cylinder(points_xyz):
  points_x, points_y, points_z = torch.unbind(points_xyz, axis=-1)
  points_rho = torch.sqrt(points_x**2 + points_y**2)
  points_phi = torch.atan2(points_y, points_x)
  points_cylinder = torch.stack([points_phi, points_z, points_rho], axis=-1)
  return points_cylinder


def points_cylinder_to_xyz(points_cylinder):
  points_phi, points_z, points_rho = torch.unbind(points_cylinder, axis=-1)
  points_x = points_rho * torch.cos(points_phi)
  points_y = points_rho * torch.sin(points_phi)
  points_xyz = torch.stack([points_x, points_y, points_z], axis=-1)
  return points_xyz



def _batched_unsorted_segment_fn(batched_data,
                                 batched_segment_ids,
                                 num_segments,
                                 unsorted_segment_fn,
                                 batched_padding=None,
                                 name=None):
  """Calls an unsorted segment function on a batch of data."""
  batch_size = batched_data.shape[0]
  batched_segment_shape = batched_segment_ids.shape

  segment_id_start = torch.Tensor(range(0, batch_size)).type(batched_segment_ids.dtype).to(batched_segment_ids.device)
  segment_id_start *= num_segments 

  segment_id_start = segment_id_start.reshape([-1] + [1] * (len(batched_segment_shape) - 1))
  batched_segment_ids_new = batched_segment_ids + segment_id_start

  if batched_padding is not None:
    batched_segment_ids_new = torch.where(batched_padding,
        -torch.ones_like(batched_segment_ids_new).type(batched_segment_ids_new.dtype).to(batched_segment_ids_new.device),
        batched_segment_ids_new)

  if unsorted_segment_fn == scatter_add:
    batched_segment_output = unsorted_segment_fn(batched_data[batched_segment_ids_new!=-1],batched_segment_ids_new[batched_segment_ids_new!=-1])
    zero = batch_size * num_segments-batched_segment_output.shape[0]
    batched_segment_output = torch.nn.functional.pad(batched_segment_output,[0,zero],value=0)
  elif unsorted_segment_fn == scatter_max:
    batched_segment_output,_ = unsorted_segment_fn(batched_data[batched_segment_ids_new!=-1],batched_segment_ids_new[batched_segment_ids_new!=-1],dim=0)
    zero = batch_size * num_segments-batched_segment_output.shape[0]
    batched_segment_output = torch.nn.functional.pad(batched_segment_output,[0,0,0,zero],value=0)
  else:
    batched_segment_output = unsorted_segment_fn(batched_data[batched_segment_ids_new!=-1],batched_segment_ids_new[batched_segment_ids_new!=-1],dim=0)
    zero = batch_size * num_segments-batched_segment_output.shape[0]
    batched_segment_output = torch.nn.functional.pad(batched_segment_output,[0,0,0,zero],value=0)
  
  output_shape = batched_segment_output.shape

  batched_segment_output = batched_segment_output.reshape([batch_size, num_segments] + list(output_shape[1:]))

  return batched_segment_output

batched_unsorted_segment_max = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=scatter_max)
batched_unsorted_segment_mean = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=scatter_mean)
batched_unsorted_segment_sum = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=scatter_add)
batched_unsorted_segment_min = functools.partial(
    _batched_unsorted_segment_fn,
    unsorted_segment_fn=scatter_min)

def bilinear_interpolate_torch(im_, index_):
  dtype = torch.cuda.FloatTensor
  dtype_long = torch.cuda.LongTensor
  batch_size,_,_,_ =im_.shape
  out = []
  for i,_ in enumerate(range(batch_size)):
      index = index_[i,...]
      im = im_[i,...]
      x = index[:,0]
      y = index[:,1]
      x0 = torch.floor(x).type(dtype_long)
      x1 = x0 + 1

      y0 = torch.floor(y).type(dtype_long)
      y1 = y0 + 1

      x0 = torch.clamp(x0, 0, im.shape[1]-1)
      x1 = torch.clamp(x1, 0, im.shape[1]-1)
      y0 = torch.clamp(y0, 0, im.shape[0]-1)
      y1 = torch.clamp(y1, 0, im.shape[0]-1)

      Ia = im[ y0, x0 ]
      Ib = im[ y1, x0 ]
      Ic = im[ y0, x1 ]
      Id = im[ y1, x1 ]

      wa = (x1.type(dtype)-x) * (y1.type(dtype)-y)
      wb = (x1.type(dtype)-x) * (y-y0.type(dtype))
      wc = (x-x0.type(dtype)) * (y1.type(dtype)-y)
      wd = (x-x0.type(dtype)) * (y-y0.type(dtype))
      out.append(torch.t((torch.t(Ia)*wa)) + torch.t(torch.t(Ib)*wb) + torch.t(torch.t(Ic)*wc) + torch.t(torch.t(Id)*wd))

  return torch.stack(out,dim=0)