import numpy as np
from scipy.optimize import fsolve
import sympy as sp
from phys_engine import time_integrator
import cupy as cp
import time
import simulators
print("dir(simulators): ", dir(simulators))

def magnitude_squared(v):
    return np.dot(v, v)

def overlap(covariance1, covariance2):
    # covariance1, covariance2: 4x4
    # return: float
    # calculate the overlap between two 2D gaussians
    axis_x1 = covariance1[0][:3]
    axis_y1 = covariance1[1][:3]
    axis_x2 = covariance2[0][:3]
    axis_y2 = covariance2[1][:3]
    normal1 = np.cross(axis_x1, axis_y1)
    normal2 = np.cross(axis_x2, axis_y2)
    if (np.linalg.norm(np.cross(normal1, normal2)) < 1e-6):
        # parallel
        return 0
    center1 = covariance1[3][:3]
    center2 = covariance2[3][:3]
    if (np.linalg.norm(center1 - center2) > np.max([np.linalg.norm(axis_x1), np.linalg.norm(axis_y1)]) + np.max([np.linalg.norm(axis_x2), np.linalg.norm(axis_y2)])):
        return 0
    line_direction = np.cross(normal1, normal2)
    A = np.array([normal1, normal2, line_direction])
    b = np.array([np.dot(normal1, center1), np.dot(normal2, center2), 0])
    d = center2 - center1
    line_point = np.linalg.solve(A, b)
    intersection_point1 = np.zeros(3)
    intersection_point2 = np.zeros(3)
    intersection_point3 = np.zeros(3)
    intersection_point4 = np.zeros(3)
    # solve eq: center1 + u * axis_x1 + v * axis_y1 = line_point + t * line_direction, where u^2 + v^2 = 1
    for i in range(3):
        if (np.abs(line_direction[i]) > 1e-6):
            def t(pi):
                return (pi - line_point[i]) / line_direction[i]
            # for j != i, pj = line_point[j] + t(pi) * line_direction[j]
            if (i == 0):
                # p = [p0, p1, p2]
                # t = (p0 - line_point[0]) / line_direction[0]
                # p1 = line_point[1] + t(p0) * line_direction[1]
                # p2 = line_point[2] + t(p0) * line_direction[2]
                # ((p - center1).dot(axis_x1)  / |axis_x1|^2)^2 + ((p - center1).dot(axis_y1) / |axis_y1|^2)^2 = 1
                # a * p0^2 + b * p0 + c = 0
                p0 = sp.symbols('p0')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center1 = sp.Matrix([center1[0], center1[1], center1[2]]).reshape(3, 1)
                axis_x1 = sp.Matrix([axis_x1[0], axis_x1[1], axis_x1[2]]).reshape(3, 1)
                axis_y1 = sp.Matrix([axis_y1[1], axis_y1[1], axis_y1[2]]).reshape(3, 1)

                t_ = (p0 - line_point[0]) / line_direction[0]
                p1 = line_point[1] + t_ * line_direction[1]
                p2 = line_point[2] + t_ * line_direction[2]
                p = sp.Matrix([p0, p1, p2]).reshape(3, 1)

                ellipse_eq = ((p - center1).dot(axis_x1) / axis_x1.norm()**2)**2 + ((p - center1).dot(axis_y1) / axis_y1.norm()**2)**2 - 1

                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p0).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point1[0] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point2[0] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point1[1] = line_point[1] + t(intersection_point1[0]) * line_direction[1]
                intersection_point2[1] = line_point[1] + t(intersection_point2[0]) * line_direction[1]
                intersection_point1[2] = line_point[2] + t(intersection_point1[0]) * line_direction[2]
                intersection_point2[2] = line_point[2] + t(intersection_point2[0]) * line_direction[2]
            elif (i == 1):
                p1 = sp.symbols('p1')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center1 = sp.Matrix([center1[0], center1[1], center1[2]]).reshape(3, 1)
                axis_x1 = sp.Matrix([axis_x1[0], axis_x1[1], axis_x1[2]]).reshape(3, 1)
                axis_y1 = sp.Matrix([axis_y1[1], axis_y1[1], axis_y1[2]]).reshape(3, 1)
                t_ = (p1 - line_point[1]) / line_direction[1]
                p0 = line_point[0] + t_ * line_direction[0]
                p2 = line_point[2] + t_ * line_direction[2]
                p = sp.Matrix([p0, p1, p2])

                ellipse_eq = ((p - center1).dot(axis_x1) / axis_x1.norm()**2)**2 + ((p - center1).dot(axis_y1) / axis_y1.norm()**2)**2 - 1

                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p1).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point1[1] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point2[1] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point1[0] = line_point[0] + t(intersection_point1[1]) * line_direction[0]
                intersection_point2[0] = line_point[0] + t(intersection_point2[1]) * line_direction[0]
                intersection_point1[2] = line_point[2] + t(intersection_point1[1]) * line_direction[2]
                intersection_point2[2] = line_point[2] + t(intersection_point2[1]) * line_direction[2]
            else:
                p2 = sp.symbols('p2')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center1 = sp.Matrix([center1[0], center1[1], center1[2]]).reshape(3, 1)
                axis_x1 = sp.Matrix([axis_x1[0], axis_x1[1], axis_x1[2]]).reshape(3, 1)
                axis_y1 = sp.Matrix([axis_y1[1], axis_y1[1], axis_y1[2]]).reshape(3, 1)
                t_ = (p2 - line_point[2]) / line_direction[2]
                p0 = line_point[0] + t_ * line_direction[0]
                p1 = line_point[1] + t_ * line_direction[1]
                p = sp.Matrix([p0, p1, p2])

                ellipse_eq = ((p - center1).dot(axis_x1) / axis_x1.norm()**2)**2 + ((p - center1).dot(axis_y1) / axis_y1.norm()**2)**2 - 1

                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p2).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point1[2] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point2[2] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point1[0] = line_point[0] + t(intersection_point1[2]) * line_direction[0]
                intersection_point2[0] = line_point[0] + t(intersection_point2[2]) * line_direction[0]
                intersection_point1[1] = line_point[1] + t(intersection_point1[2]) * line_direction[1]
                intersection_point2[1] = line_point[1] + t(intersection_point2[2]) * line_direction[1]
            break

    for i in range(3):
        if (np.abs(line_direction[i]) > 1e-6):
            def t(pi):
                return (pi - line_point[i]) / line_direction[i]
            # for j != i, pj = line_point[j] + t(pi) * line_direction[j]
            if (i == 0):
                # p = [p0, p1, p2]
                # t = (p0 - line_point[0]) / line_direction[0]
                # p1 = line_point[1] + t(p0) * line_direction[1]
                # p2 = line_point[2] + t(p0) * line_direction[2]
                # ((p - center2).dot(axis_x2)  / |axis_x2|^2)^2 + ((p - center2).dot(axis_y2) / |axis_y2|^2)^2 = 1
                # a * p0^2 + b * p0 + c = 0     
                p0 = sp.symbols('p0')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center2 = sp.Matrix([center2[0], center2[1], center2[2]])
                axis_x2 = sp.Matrix([axis_x2[0], axis_x2[1], axis_x2[2]])
                axis_y2 = sp.Matrix([axis_y2[1], axis_y2[1], axis_y2[2]])

                t_ = (p0 - line_point[0]) / line_direction[0]
                p1 = line_point[1] + t_ * line_direction[1]
                p2 = line_point[2] + t_ * line_direction[2]
                p = sp.Matrix([p0, p1, p2])

                ellipse_eq = ((p - center2).dot(axis_x2) / axis_x2.norm()**2)**2 + ((p - center2).dot(axis_y2) / axis_y2.norm()**2)**2 - 1
                
                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p0).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point3[0] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point4[0] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point3[1] = line_point[1] + t(intersection_point3[0]) * line_direction[1]
                intersection_point4[1] = line_point[1] + t(intersection_point4[0]) * line_direction[1]
                intersection_point3[2] = line_point[2] + t(intersection_point3[0]) * line_direction[2]
                intersection_point4[2] = line_point[2] + t(intersection_point4[0]) * line_direction[2]
            elif (i == 1):
                p1 = sp.symbols('p1')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center2 = sp.Matrix([center2[0], center2[1], center2[2]])
                axis_x2 = sp.Matrix([axis_x2[0], axis_x2[1], axis_x2[2]])
                axis_y2 = sp.Matrix([axis_y2[1], axis_y2[1], axis_y2[2]])
                t_ = (p1 - line_point[1]) / line_direction[1]
                p0 = line_point[0] + t_ * line_direction[0]
                p2 = line_point[2] + t_ * line_direction[2]
                p = sp.Matrix([p0, p1, p2])

                ellipse_eq = ((p - center2).dot(axis_x2) / axis_x2.norm()**2)**2 + ((p - center2).dot(axis_y2) / axis_y2.norm()**2)**2 - 1

                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p1).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point3[1] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point4[1] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point3[0] = line_point[0] + t(intersection_point3[1]) * line_direction[0]
                intersection_point4[0] = line_point[0] + t(intersection_point4[1]) * line_direction[0]
                intersection_point3[2] = line_point[2] + t(intersection_point3[1]) * line_direction[2]
                intersection_point4[2] = line_point[2] + t(intersection_point4[1]) * line_direction[2]
            else:
                p2 = sp.symbols('p2')
                line_point = sp.Matrix([line_point[0], line_point[1], line_point[2]])
                line_direction = sp.Matrix([line_direction[0], line_direction[1], line_direction[2]])
                center2 = sp.Matrix([center2[0], center2[1], center2[2]])
                axis_x2 = sp.Matrix([axis_x2[0], axis_x2[1], axis_x2[2]])
                axis_y2 = sp.Matrix([axis_y2[1], axis_y2[1], axis_y2[2]])
                t_ = (p2 - line_point[2]) / line_direction[2]
                p0 = line_point[0] + t_ * line_direction[0]
                p1 = line_point[1] + t_ * line_direction[1]
                p = sp.Matrix([p0, p1, p2])

                ellipse_eq = ((p - center2).dot(axis_x2) / axis_x2.norm()**2)**2 + ((p - center2).dot(axis_y2) / axis_y2.norm()**2)**2 - 1

                ellipse_eq_expanded = sp.expand(ellipse_eq)
                coeffs = sp.Poly(ellipse_eq_expanded, p2).all_coeffs()

                a, b, c = coeffs
                if (b ** 2 - 4 * a * c < 0):
                    return 0
                intersection_point3[2] = (-b + sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point4[2] = (-b - sp.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                intersection_point3[0] = line_point[0] + t(intersection_point3[2]) * line_direction[0]
                intersection_point4[0] = line_point[0] + t(intersection_point4[2]) * line_direction[0]
                intersection_point3[1] = line_point[1] + t(intersection_point3[2]) * line_direction[1]
                intersection_point4[1] = line_point[1] + t(intersection_point4[2]) * line_direction[1]
            break
    # choose the minimum distance of points1, 2, 3, 4
    min_distance = np.inf
    for i in range(4):
        min_distance = min(min_distance, magnitude_squared(intersection_point1 - intersection_point2))
        min_distance = min(min_distance, magnitude_squared(intersection_point1 - intersection_point3))
        min_distance = min(min_distance, magnitude_squared(intersection_point1 - intersection_point4))
        min_distance = min(min_distance, magnitude_squared(intersection_point2 - intersection_point3))
        min_distance = min(min_distance, magnitude_squared(intersection_point2 - intersection_point4))
        min_distance = min(min_distance, magnitude_squared(intersection_point3 - intersection_point4))
    return min_distance

'''
def set_spring(covariance):
    elements = []
    initial_length = []
    stiffness = []
    for i in range(len(covariance)):
        for j in range(i + 1, len(covariance)):
            dens = overlap(covariance[i], covariance[j])
            if (j != i and dens > 0.01):
                elements.append([i, j])
                initial_length.append([np.linalg.norm(covariance[i][3][:3] - covariance[j][3][:3])])
                stiffness.append(dens * 1e10)
                # print("i: ", i, "j: ", j, "overlap: ", dens)
    print("len(elements): ", len(elements))
    return elements, initial_length, stiffness
'''

def set_spring(covariance):
    elements = []
    initial_length = []
    stiffness = []
    volume = []
    springs = [[] for _ in range(len(covariance))]
    for i in range(len(covariance)):
        square = 0
        dens = []
        for j in range(len(covariance)):
            dens.append(np.linalg.norm(covariance[i][3][:3] - covariance[j][3][:3]))
        dens = np.array(dens)
        dens = sorted(dens)
        # if (dens[6] > 0.5):
        #     print("+1")
        #     volume.append(1e-2)
        #     continue
        for j in range(len(covariance)):
            den = np.linalg.norm(covariance[i][3][:3] - covariance[j][3][:3])
            if (j != i and den < dens[6]):
                is_connected = False
                if (j < i):
                    for k in range(len(springs[i])):
                        if(springs[i][k] == j):
                            is_connected = True
                            break
                # print("dens: ", dens)
                if (is_connected):
                    continue
                elements.append([i, j])
                initial_length.append(den)
                stiffness.append(1e5)
                springs[i].append(j)
                springs[j].append(i)
                # print("i: ", i, "j: ", j, "overlap: ", dens)
        for j in range(6):
            square += dens[j]
        square = (square / 6) ** 2
        # volume.append(square * 1e-3)
        volume.append(np.linalg.norm(covariance[i][0][:3]) * np.linalg.norm(covariance[i][1][:3]) * 1e-3)
    spring_volume = np.zeros(len(elements))
    # print("len(volume): ", len(volume))
    # for i in range(len(covariance)):
    #     print("i: ", i, "spring[i]", len(springs[i]))
    for i in range(len(elements)):
        spring_volume[i] = volume[elements[i][0]] / len(springs[elements[i][0]]) + volume[elements[i][1]] / len(springs[elements[i][1]])
        


    print("len(elements): ", len(elements))
    np.save("/home/renbowen/elements.npy", elements)
    np.save("/home/renbowen/initial_length.npy", initial_length)
    return elements, initial_length, stiffness, volume, spring_volume

def simulation(x, covariance):
    # reference: https://github.com/phys-sim-book/solid-sim-tutorial/tree/main/1_mass_spring
    x_history = []
    covariance_history = []
    x = x.detach().cpu().numpy().copy()
    x0 = x.copy()
    covariance = covariance.detach().cpu().numpy().copy()
    np.save("/home/renbowen/covariance.npy", covariance)
    print("covariance.shape: ", covariance.shape)
    for i in range (100): 
        # print("x_: ", covariance[i])
        pass
    print("finish init")
    # np.savetxt('/home/renbowen/covariance.csv', covariance.reshape(covariance.shape[0], -1), delimiter=',', fmt='%.6f', header=','.join([f'Val{i}' for i in range(16)]), comments='')
    start_time = time.time()
    elements, initial_length, stiffness, volume, spring_volume = set_spring(covariance)
    # elements = np.load("/home/renbowen/elements.npy")
    # np.savetxt('/home/renbowen/elements.csv', elements.reshape(-1), delimiter=',', fmt='%.6f', header='Col1', comments='')
    # initial_length = np.load("/home/renbowen/initial_length.npy")
    # np.savetxt('/home/renbowen/initial_length.csv', initial_length, delimiter=',', fmt='%.6f', header='Col1', comments='')
    # stiffness = [1e6] * len(elements)
    print("set_spring time: ", time.time() - start_time)
    #elements = np.load("elements.npy")
    #initial_length = np.load("initial_length.npy")
    #stiffness = np.load("stiffness.npy")
    np.save("/home/renbowen/elements.npy", elements)
    np.save("/home/renbowen/initial_length.npy", initial_length)
    np.save("/home/renbowen/stiffness.npy", stiffness)
    np.save("/home/renbowen/volume.npy", volume)
    np.save("/home/renbowen/spring_volume.npy", spring_volume)
    elements = np.load("/home/renbowen/elements.npy")
    initial_length = np.load("/home/renbowen/initial_length.npy")
    stiffness = np.load("/home/renbowen/stiffness.npy")
    volume = np.load("/home/renbowen/volume.npy")
    spring_volume = np.load("/home/renbowen/spring_volume.npy")
    # hyperparameters
    rho = 5e2  # mass density
    h = 0.02  # time step
    torlerance = 1e-4
    frame_num = 100
    velocity = np.zeros((len(x), 3))
    '''
    rhos = []
    for i in range(len(x)):
        rhos.append(rho * np.linalg.norm(covariance[i][0][:3]) * np.linalg.norm(covariance[i][1][:3]))
    '''
    print("len(x): ", len(x))
    simulator = simulators.MassSpringSimulator3d(
        rho * np.array(volume), # mass of each node
        1,
        1, #initial_stretch,  not used
        np.array(stiffness * spring_volume * 10), #stiffness,
        h, 
        torlerance, 
        1, 
        np.array(initial_length).reshape(-1),
        covariance.reshape(-1), 
        np.array(elements).reshape(-1),  
    )
    print("len(x): ", len(x))

    for i in range(frame_num):
        print("time step: ", i)
        num = 0
        print("len(x): ", len(x))
        for j in range(len(x)):
            if (x0[j][2] < 2.7 and x0[j][2] > 2.3 and x0[j][0] > -0.6 and x0[j][0] < -0.3):
            # if (x0[j][0] > 0.45):
                velocity[j][1] = 1
                num += 1
        print("num_set_v: ", num)
        # if(i == 0):
        # simulator.set_v(velocity.reshape(-1))
        simulator.run()
        velocity = np.array(simulator.get_v()).reshape(-1, 3)
        print("max_v: ", np.max(np.abs(simulator.get_v())))
        # x0 = x.copy()
        x = np.array(simulator.get_x()).reshape(-1, 3)
        covariance = simulator.get_covariance()
        x_history.append(x)
        covariance_history.append(covariance)
    np.save("/home/renbowen/x_history.npy", x_history)
    np.save("/home/renbowen/covariance_history.npy", covariance_history)

    '''
    for i in range(frame_num):
        num = 0
        for j in range(len(x)):
            if (x0[j][2] < 2.7 and x0[j][2] > 2.3 and x0[j][0] > -0.6 and x0[j][0] < -0.3):
                velocity[j][1] = -30
                num += 1
        print("num_set_v: ", num)
        # explicit: calculate forces
        
        f = np.zeros((len(x), 3))
        for j in range(len(elements)):
            m, n = elements[j]
            x1 = x[m]
            x2 = x[n]
            dx = x2 - x1
            l = np.linalg.norm(dx)
            f_spring = -stiffness[j] * (l - initial_length[j]) * dx / l
            f[m] += f_spring
            f[n] -= f_spring
        # calculate acceleration
        a = f / rho
        # update velocity and position
        velocity = velocity + a * h
        x = x + velocity * h
        
        [x_next, v] = time_integrator.step_forward(x, elements, velocity, [rho] * len(x), [length ** 2 for length in initial_length], stiffness, h, 1e-2)
        print("max velocity: ", np.max(v))
        x_history.append(x)
        # shape matching
        start_time = time.time()
    for i in range(len(x)):
        connected_xs = []
        next_connected_xs = []
        for e in range(len(elements)):
            if elements[e][0] == i:
                connected_xs.append(x[elements[e][1]])
                next_connected_xs.append(x_next[elements[e][1]])
            elif elements[e][1] == i:
                connected_xs.append(x[elements[e][0]])
                next_connected_xs.append(x_next[elements[e][0]])

        connected_xs = np.array(connected_xs)
        next_connected_xs = np.array(next_connected_xs)
        center = connected_xs.mean(axis=0)
        next_center = next_connected_xs.mean(axis=0)
        connected_xs -= center
        next_connected_xs -= next_center

        H = np.dot(connected_xs.T, next_connected_xs)
        H_gpu = cp.asarray(H)
        U, S, V = cp.linalg.svd(H_gpu)
        R = cp.dot(U, V).get()  # .get() retrieves the result from GPU to CPU

        covariance[i][:3][:3] = R @ covariance[i][:3][:3]
        print("shape maching time: ", time.time() - start_time)
        covariance_history.append(covariance)
        x = x_next
    '''
    np.save("/home/renbowen/x_history.npy", x_history)
    return x_history

            