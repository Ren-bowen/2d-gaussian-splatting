from backend import *
import numpy as np

np_real = lambda x: np.array(x, dtype=np.float64).copy()
np_integer = lambda x: np.array(x, dtype=np.int32).copy()

def create_sim(cell_num=(16, 16), bending_stiffness=0.):
    # In this homework, the area of triangle is fixed to be 0.005.
    dx = 0.1
    cloth_density = 5e-1
    g = 9.81
    h = 1e-3
    frame_num = 1000

    def index(i, j): return i * (cell_num[1] + 1) + j
    vertices = []
    for i in range(cell_num[0] + 1):
        for j in range(cell_num[1] + 1):
            vertices.append([(i - 2 * cell_num[0] / 3) * dx, (j - 2 * cell_num[1] / 3) * dx])
    vertices = np_real(vertices)
    faces = []
    for i in range(cell_num[0]):
        for j in range(cell_num[1]):
            faces += [(index(i, j), index(i + 1, j), index(i + 1, j + 1)),
                (index(i, j + 1), index(i, j), index(i + 1, j + 1))]
    faces = np_integer(faces)
    print("faces")
    sim = Simulator(vertices.T, faces.T, cloth_density, bending_stiffness)
    print("sim")
    sim_pos = sim.position()
    sim_pos[2, :] = 2
    sim.set_position(sim_pos)
    a = np.zeros_like(sim.position())
    a[2, :] = -g
    sim.set_external_acceleration(a)
    return sim, h, frame_num, faces
if __name__ == "__main__":
    sim, h, frame_num, faces = create_sim()
    print("Hello, world!")