import sympy as sp

# 定义符号变量
p0 = sp.symbols('p0')
line_point = sp.Matrix([line_point_x, line_point_y, line_point_z])
line_direction = sp.Matrix([line_direction_x, line_direction_y, line_direction_z])
center1 = sp.Matrix([center1_x, center1_y, center1_z])
axis_x1 = sp.Matrix([axis_x1_x, axis_x1_y, axis_x1_z])
axis_y1 = sp.Matrix([axis_y1_x, axis_y1_y, axis_y1_z])

# 定义p1, p2
t = (p0 - line_point[0]) / line_direction[0]
p1 = line_point[1] + t * line_direction[1]
p2 = line_point[2] + t * line_direction[2]
p = sp.Matrix([p0, p1, p2])

# 定义椭圆方程
ellipse_eq = ((p - center1).dot(axis_x1) / axis_x1.norm()**2)**2 + ((p - center1).dot(axis_y1) / axis_y1.norm()**2)**2 - 1

# 展开方程，获得系数a, b, c
ellipse_eq_expanded = sp.expand(ellipse_eq)
coeffs = sp.Poly(ellipse_eq_expanded, p0).all_coeffs()

a, b, c = coeffs
print(f"a = {a}, b = {b}, c = {c}")
