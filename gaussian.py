import numpy as np
import open3d as o3d
import os
import copy
import taichi as ti

np_real = lambda x: np.array(x, dtype=np.float64).copy()
np_integer = lambda x: np.array(x, dtype=np.int32).copy()

def act_gaus(x_last_, x_, faces_, gaussians_xyz_, gaussians_scaling_):
    # read vertices and faces form obj
    gmesh = np.zeros(gaussians_xyz_.shape[0])
    file_path = r"C:\ren\code\2d-gaussian-splatting\output\regular\iter_30000"
    gmesh_file = file_path + "/gmesh.npy"
    os.makedirs(file_path, exist_ok=True)
    center_ = np.zeros((faces_.shape[0], 3))
    center_last_ = np.zeros((faces_.shape[0], 3))
    for i in range(faces_.shape[0]):
        for j in range(3):
            center_[i] += x_[faces_[i][j]] / 3
            center_last_[i] += x_last_[faces_[i][j]] / 3
    print("np.max(center_ - center_last_)", np.max(center_ - center_last_))
    ti.init(arch=ti.gpu)
    @ti.kernel
    def create_gmesh(gaussiasns_xyz:ti.types.ndarray(ndim=1), center_last:ti.types.ndarray(ndim=1), gmesh:ti.types.ndarray(ndim=1)):
        for i in range(gaussiasns_xyz.shape[0]):
            min_idx = 0
            min_dis = 0.
            min_dis += (gaussiasns_xyz[i] - center_last[0]).norm()
            for j in range(center_last.shape[0]):
                dis = (gaussiasns_xyz[i] - center_last[j]).norm()
                if dis < min_dis:
                    min_dis = dis
                    min_idx = j
            gmesh[i] = min_idx
    if not os.path.isfile(gmesh_file):
        '''
        for i in range(gaussians_xyz_.shape[0]):
            min_idx = 0
            min_dis = np.linalg.norm(gaussians_xyz_[i] - center_last_[0])
            for j in range(center_last_.shape[0]):
                if np.linalg.norm(gaussians_xyz_[i] - center_last_[j]) < min_dis:
                    min_dis = np.linalg.norm(gaussians_xyz_[i] - center_last_[j])
                    min_idx = j
            gmesh[i] = min_idx
            print("min_dis", min_dis)
        '''
        gaussians_xyz_ti = ti.Vector.ndarray(3, ti.f32, shape=(gaussians_xyz_.shape[0]))
        center_last_ti = ti.Vector.ndarray(3, ti.f32, shape=(center_last_.shape[0]))
        gmesh_ti = ti.ndarray(ti.i32, shape=(gaussians_xyz_.shape[0]))
        gaussians_xyz_ti.from_numpy(gaussians_xyz_)
        center_last_ti.from_numpy(center_last_)
        create_gmesh(gaussians_xyz_ti, center_last_ti, gmesh_ti)
        gmesh = gmesh_ti.to_numpy()
        np.save(gmesh_file, gmesh)
    else:
        gmesh = np.load(gmesh_file)
    num_positons = x_.shape[0]
    num_faces = faces_.shape[0]
    num_gaussians = gaussians_xyz_.shape[0]
    x_last = ti.Vector.ndarray(3, ti.f32, shape=(num_positons))
    x = ti.Vector.ndarray(3, ti.f32, shape=(num_positons))
    faces = ti.Vector.ndarray(3, ti.i32, shape=(num_faces))
    gmesh_ = ti.ndarray(ti.i32, shape=(num_gaussians))
    center = ti.Vector.ndarray(3, ti.f32, shape=(num_faces))
    center_last = ti.Vector.ndarray(3, ti.f32, shape=(num_faces))
    Fs = ti.Matrix.ndarray(3, 3, ti.f32, shape=(num_gaussians))
    gaussians_xyz = ti.Vector.ndarray(3, ti.f32, shape=(num_gaussians))
    gaussians_scaling = ti.Vector.ndarray(3, ti.f32, shape=(num_gaussians))
    x_last.from_numpy(x_last_)
    x.from_numpy(x_)
    faces.from_numpy(faces_)
    gmesh_.from_numpy(gmesh)
    center.from_numpy(center_)
    center_last.from_numpy(center_last_)
    gaussians_xyz.from_numpy(gaussians_xyz_)
    gaussians_scaling.from_numpy(gaussians_scaling_)

    @ti.kernel
    def process_gaussian_kernel(gmesh:ti.types.ndarray(ndim=1), gaussians_xyz:ti.types.ndarray(ndim=1), Fs:ti.types.ndarray(ndim=1),
                                 gaussians_scaling:ti.types.ndarray(ndim=1), center:ti.types.ndarray(ndim=1), center_last:ti.types.ndarray(ndim=1),
                                 x_last:ti.types.ndarray(ndim=1), x:ti.types.ndarray(ndim=1), faces:ti.types.ndarray(ndim=1)):
        for i in range(num_gaussians):
            j = gmesh[i]
            delta_center = (center[j] - center_last[j])
            gaussians_xyz[i] += delta_center
            
            normal_last = (x_last[faces[j][1]] - x_last[faces[j][0]]).cross(x_last[faces[j][2]] - x_last[faces[j][0]])
            normal_last /= normal_last.norm()
            normal = (x[faces[j][1]] - x[faces[j][0]]).cross(x[faces[j][2]] - x[faces[j][0]])
            normal /= normal.norm()
            F1 = ti.Matrix.zero(ti.f32, 3, 3)
            F2 = ti.Matrix.zero(ti.f32, 3, 3)
            for k in range(3):
                F1[k, 0] = x_last[faces[j][1]][k] - x_last[faces[j][0]][k]
                F1[k, 1] = x_last[faces[j][2]][k] - x_last[faces[j][0]][k]
                F1[k, 2] = normal_last[k]
                F2[k, 0] = x[faces[j][1]][k] - x[faces[j][0]][k]
                F2[k, 1] = x[faces[j][2]][k] - x[faces[j][0]][k]
                F2[k, 2] = normal[k]
            F1_inv = F1.inverse()
            F = F2 @ F1_inv
            Fs[i] = F
            
            R, S = ti.polar_decompose(F)
            '''
            rotations[i] = R
            gaussians_scaling[i] = S @ gaussians_scaling[i]
            '''
            

    process_gaussian_kernel(gmesh_, gaussians_xyz, Fs, gaussians_scaling, center, center_last, x_last, x, faces)

    result_xyz = gaussians_xyz.to_numpy()
    result_rotation = Fs.to_numpy()
    result_scaling = gaussians_scaling.to_numpy()
    return result_xyz, result_rotation, result_scaling

if __name__ == "__main__":
    gaussian_file_path = r"C:\ren\code\2d-gaussian-splatting\output\regular\iter_30000"
    gaussian_xyz = np.load(gaussian_file_path + "/new_gaussians_xyz.npy")
    gaussian_scaling = np.load(gaussian_file_path + "/new_gaussians_scaling.npy")
    print("gaussian_xyz", gaussian_xyz.shape)
    print("gaussian_scaling", gaussian_scaling.shape)
    frame_num = 200
    gaussians_xyz_list = []
    gaussians_rotation_list = []
    gaussians_scaling_list = []
    gaussians_xyz_list.append(copy.deepcopy(gaussian_xyz))
    # origin rotation is all identity matrix
    gaussians_rotation_list.append(np.eye(3).reshape(1, 3, 3).repeat(gaussian_xyz.shape[0], axis=0))
    gaussians_scaling_list.append(copy.deepcopy(gaussian_scaling))
    os.makedirs(gaussian_file_path + "/point_cloud", exist_ok=True)
    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(gaussian_xyz)
    o3d.io.write_point_cloud(gaussian_file_path + "/point_cloud/gaussian_xyz0.ply", points)
    file_path = r"C:\Users\bowen ren\Downloads\sim_data"
    mesh_origin = o3d.io.read_triangle_mesh(file_path + "/0.obj")
    position_origin = np_real(mesh_origin.vertices)
    # create a rotation matrix of 2 degree around z axis
    R = np.array([[np.cos(2 * np.pi / 180), -np.sin(2 * np.pi / 180), 0],
                    [np.sin(2 * np.pi / 180), np.cos(2 * np.pi / 180), 0],
                    [0, 0, 1]])
    
    position_list = []
    faces = np_integer(mesh_origin.triangles)
    '''
    for i in range(frame_num):
        position_list.append(copy.deepcopy(position_origin))
        position_origin = (position_origin - position_origin.mean(axis=0)) @ R.T + position_origin.mean(axis=0)
        # save the position and face of each frame to obj file
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(position_origin)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(file_path + "/rot{}.obj".format(i), mesh)
    '''
    for i in range(0, frame_num - 5, 5):
        mesh_last = o3d.io.read_triangle_mesh(file_path + "/rot{}.obj".format(i))
        mesh = o3d.io.read_triangle_mesh(file_path + "/rot{}.obj".format(i + 5))
        position_last = np_real(mesh_last.vertices)
        position = np_real(mesh.vertices)
        # position_last = position_list[i]
        # position = position_list[i + 1]
        gaussians_xyz_new, F, gaussian_scaling_new = act_gaus(position_last, position, faces, gaussian_xyz, gaussian_scaling)
        # print("np.max(gaussians_xyz_new - gaussian_xyz)", np.max(gaussians_xyz_new - gaussian_xyz))
        gaussian_xyz = gaussians_xyz_new
        gaussian_scaling = gaussian_scaling_new
        # save new_gaussians_xyz as obj file of point cloud
        points = o3d.geometry.PointCloud()
        points.points = o3d.utility.Vector3dVector(gaussians_xyz_new)
        o3d.io.write_point_cloud(gaussian_file_path + "/point_cloud/gaussian_xyz{}.ply".format(i + 5), points)
        gaussians_xyz_list.append(copy.deepcopy(gaussians_xyz_new))
        gaussians_rotation_list.append(copy.deepcopy(F))
        gaussians_scaling_list.append(copy.deepcopy(gaussian_scaling_new))

        print("frame " + str(i) + " done")
    np.save(os.path.join(gaussian_file_path, "gaussian_xyz_list_rot.npy"), gaussians_xyz_list)
    np.save(os.path.join(gaussian_file_path, "rotation_list_rot.npy"), gaussians_rotation_list)
    np.save(os.path.join(gaussian_file_path, "gaussian_scaling_list_rot.npy"), gaussians_scaling_list)