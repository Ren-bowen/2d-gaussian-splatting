import numpy as np
from scipy.optimize import fsolve
import sympy as sp

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
    for i in range(len(covariance)):
        for j in range(i + 1, len(covariance)):
            dens = np.linalg.norm(covariance[i][3][:3] - covariance[j][3][:3])
            if (j != i and dens < 0.05):
                print("dens: ", dens)
                elements.append([i, j])
                initial_length.append([dens])
                stiffness.append(1e7)
                # print("i: ", i, "j: ", j, "overlap: ", dens)
    print("len(elements): ", len(elements))
    return elements, initial_length, stiffness

def simulation(x, covariance):
    x_history = []
    x = x.detach().cpu().numpy().copy()
    x0 = x
    covariance = covariance.detach().cpu().numpy().copy()
    print("covariance.shape: ", covariance.shape)
    elements, initial_length, stiffness = set_spring(covariance)
    #elements = np.load("elements.npy")
    #initial_length = np.load("initial_length.npy")
    #stiffness = np.load("stiffness.npy")
    #np.save("elements.npy", elements)
    #np.save("initial_length.npy", initial_length)
    #np.save("stiffness.npy", stiffness)
    # hyperparameters
    rho = 1e2  # mass density
    h = 0.05   # time step
    frame_num = 5
    velocity = np.zeros((len(x), 3))
    for i in range(frame_num):
        num = 0
        for j in range(len(x)):
            if (x0[j][2] < 2.7 and x0[j][2] > 2.3 and x0[j][0] > -0.6 and x0[j][0] < -0.3):
                x[j][1] -= 0.09
                num += 1
        print("num: ", num)
        # calculate forces
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
        x_history.append(x)
    return x_history

            