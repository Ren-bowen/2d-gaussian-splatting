import open3d as o3d
import numpy as np

try:
    sim_data = np.load("sim_data.npz")
except:
    print("Error when loading data. Run simulation.py first and place the .npy file at the correct place.")
    raise FileNotFoundError
position, faces, frame_num = sim_data["position"], sim_data["faces"], sim_data["frame_num"][0]

def uv(i, j): return i / 16, j / 16
uvs = []
for i in range(16):
    for j in range(16):
        uvs += [uv(i, j), uv(i + 1, j), uv(i + 1, j + 1), uv(i, j + 1), uv(i, j), uv(i + 1, j + 1)]

# 初始化可视化器
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(visible=True)

# 设置渲染选项
o3d_opt = vis.get_render_option()
o3d_opt.mesh_show_back_face = True

# 读取纹理
texture_img = o3d.io.read_image("cloth_texture.jpg")
o3d_uvs = o3d.utility.Vector2dVector(np.array(uvs).copy())

# 创建一个居中的球体
centered_sphere = o3d.geometry.TriangleMesh.create_sphere(0.92, 100)
centered_sphere.compute_vertex_normals()
centered_sphere.paint_uniform_color([0.4, 0.6, 0.1])

def create_o3d(vs):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vs.copy()), o3d.utility.Vector3iVector(faces.copy()))
    mesh.triangle_uvs = o3d_uvs
    mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(3 * len(faces), dtype=np.int32))
    mesh.textures = [texture_img]
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("/home/renbowen/2d-gaussian-splatting/code/code/cloth_textured.ply", mesh, write_ascii=True)
    print("Mesh saved to cloth_textured.ply.")
    return mesh

# 注册一个按键回调，用于按下ESC键退出
def exit_key_callback(vis):
    vis.destroy_window()

vis.register_key_callback(256, exit_key_callback)  # 256是ESC键的ASCII码

# 添加几何体并初始化视角
if frame_num > 0:
    vis.add_geometry(create_o3d(position[0]))
    # vis.add_geometry(centered_sphere)

# 启动交互式窗口
vis.run()  # 这个命令允许你用鼠标拖动视角
# vis.destroy_window()

# 渲染循环
'''
for i in range(1, frame_num, 1):
    vis.clear_geometries()
    vis.add_geometry(create_o3d(position[i]))
    vis.add_geometry(centered_sphere)
    vis.poll_events()
    vis.update_renderer()
'''