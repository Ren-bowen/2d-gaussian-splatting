import open3d as o3d
import numpy as np

try:
	sim_data = np.load("sim_data.npz")
except:
	print("Error when loading data. Run simulation.py first and place the .npy file at the correct place.")
	raise FileNotFoundError
position, faces, frame_num = sim_data["position"], sim_data["faces"], sim_data["frame_num"][0]

view_status = \
'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 1.0, 1.0, 2.0 ],
			"boundingbox_min" : [ -1.0, -1.0, 2.0 ],
			"field_of_view" : 60.0,
			"front" : [ -0.22277517214450518, -0.66601452050614163, 0.71189597635536639 ],
			"lookat" : [ -0.17032013748761263, -0.25260797741584867, 1.0487962702497817 ],
			"up" : [ 0.32475337985372593, 0.63783712514624236, 0.6983545260530345 ],
			"zoom" : 0.90000000000000013
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''

def uv(i, j): return i / 16, j / 16
uvs = []
for i in range(16):
	for j in range(16):
		uvs += [uv(i, j), uv(i + 1, j), uv(i + 1, j + 1), uv(i, j + 1), uv(i, j), uv(i + 1, j + 1)]

vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
o3d_opt = vis.get_render_option()
o3d_opt.mesh_show_back_face = True
texture_img = o3d.io.read_image(str("cloth_texture.jpg"))
o3d_uvs = o3d.utility.Vector2dVector(np.array(uvs).copy())
centered_sphere = o3d.geometry.TriangleMesh.create_sphere(0.92, 100)
centered_sphere.compute_vertex_normals()
centered_sphere.paint_uniform_color([0.4, 0.6, 0.1])
for i in range(frame_num):
    def create_o3d(vs):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vs.copy()), o3d.utility.Vector3iVector(faces.copy()))
        mesh.triangle_uvs = o3d_uvs
        mesh.triangle_material_ids = o3d.utility.IntVector(np.zeros(3 * len(faces), dtype=np.int32))
        mesh.textures = [texture_img]
        mesh.compute_vertex_normals()
        return mesh
    vis.clear_geometries()
    vis.add_geometry(create_o3d(position[i]))
    vis.add_geometry(centered_sphere)
    vis.set_view_status(view_status)
    vis.poll_events()
    vis.update_renderer()
