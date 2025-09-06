import open3d as o3d
from Utils import Util
def GetFPFH(organized_pc, voxel_size=0.01):
    organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
    unorganized_pc = Util.organized_pc_to_unorganized_pc(organized_pc=organized_pc_np)
    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))

    radius_normal = voxel_size * 2
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(o3d_pc, o3d.geometry.KDTreeSearchParamHybrid
    (radius=radius_feature, max_nn=100))
    fpfh = pcd_fpfh.data.T
    full_fpfh = np.zeros((unorganized_pc.shape[0], fpfh.shape[1]), dtype=fpfh.dtype)
    full_fpfh[nonzero_indices, :] = fpfh
    full_fpfh_reshaped = full_fpfh.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1]))
    full_fpfh_tensor = torch.tensor(full_fpfh_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
    return full_fpfh_tensor
