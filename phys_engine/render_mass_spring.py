import igl
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import time
import os
os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"
# 加载数据
try:
    x_history = np.load("/home/renbowen/x_history.npy")
    elements = np.load("/home/renbowen/elements.npy")
    covariance = np.load("/home/renbowen/covariance.npy")
    spring_volume = np.load("/home/renbowen/spring_volume.npy")
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

frame_num = len(x_history)

# 全局变量
is_paused = False
frame_index = 0

def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

def draw_sphere(radius, slices, stacks):
    for i in range(stacks):
        lat0 = np.pi * (-0.5 + float(i) / stacks)
        z0 = radius * np.sin(lat0)
        zr0 = radius * np.cos(lat0)

        lat1 = np.pi * (-0.5 + float(i + 1) / stacks)
        z1 = radius * np.sin(lat1)
        zr1 = radius * np.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * np.pi * float(j) / slices
            x = np.cos(lng)
            y = np.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()

def display():
    global frame_index
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 100, 0, 0, 0, 0, 1, 0)

    x = x_history[frame_index]

    # 绘制球体
    glColor3f(0.4, 0.5, 0.6)  # 设置球体颜色
    for k in range(len(x)):
        radius = np.power(np.linalg.norm(covariance[k][0][:3] * covariance[k][1][:3]), 1/3) * 0.1
        glPushMatrix()
        glTranslatef(x[k][0], x[k][1], x[k][2])
        draw_sphere(radius, 50, 50)
        glPopMatrix()

    # 绘制连线
    glColor3f(0.2, 0.2, 0.2)
    glBegin(GL_LINES)
    for e in elements:
        e0, e1 = e[0], e[1]
        v0, v1 = x[e0], x[e1]
        glVertex3fv(v0)
        glVertex3fv(v1)
    glEnd()

    glutSwapBuffers()

def idle():
    global frame_index, is_paused
    if not is_paused:
        frame_index = (frame_index + 1) % frame_num
        time.sleep(1/30)
    glutPostRedisplay()

def key_pressed(key, x, y):
    global is_paused
    if key == b' ':
        is_paused = not is_paused

if __name__ == "__main__":
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("Libigl Sphere and Lines")
    init()
    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutKeyboardFunc(key_pressed)
    glutMainLoop()
