import glfw
from OpenGL.GL import *
from PIL import Image
from ctypes import c_void_p

import numpy as np
from fealpy import logger

from .kernel import calculate_rotation_matrix
from .coordinate_axes import CoordinateAxes


class OpenGLPlotter:
    def __init__(self, width=800, height=600, title="OpenGL Application"):
        if not glfw.init():
            raise Exception("GLFW cannot be initialized!")

        self.texture = None
        self.dragging = False
        self.last_mouse_pos = (width / 2, height / 2)
        self.first_mouse_use = True

        self.view_angle = 0 # 0 代表 X 轴，1 代表 Y 轴， 2 代表 Z 轴
        self.mode = 2  # 默认同时显示边和面
        self.faceColor = (0.5, 0.7, 0.9, 1.0)  # 浅蓝色
        self.edgeColor = (1.0, 1.0, 1.0, 1.0)  # 白色
        self.bgColor = (0.1, 0.2, 0.3, 1.0)   # 深海军蓝色背景

        self.transform = np.identity(4, dtype=np.float32)
        
        # 设置使用OpenGL核心管线
        #glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        #glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        #glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        #glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        
        # 确保title是字符串类型，然后在这里对其进行编码
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window cannot be created!")
        
        glfw.make_context_current(self.window)
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)

        # 设置视口大小
        glViewport(0, 0, width, height)

        # 顶点着色器源码
        self.vertex_shader_source = """
        #version 460 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoords;
        uniform mat4 transform; //变换矩阵
        //uniform mat4 projection; // 投影矩阵

        out vec2 TexCoords;

        void main()
        {
            gl_Position = transform * vec4(aPos, 1.0);
            //gl_Position = projection * transform * vec4(aPos, 1.0);
            TexCoords = aTexCoords;
        }
        """

        # 片段着色器源码
        self.fragment_shader_source = """
        #version 460 core
        in vec2 TexCoords;

        uniform int mode;  // 0: 显示面，1: 显示边，2: 显示面和边，3：显示纹理
        uniform vec4 faceColor;
        uniform vec4 edgeColor;
        uniform sampler2D textureSampler;
        out vec4 FragColor;

        void main()
        {
            if (mode == 0) {
                FragColor = faceColor;  // 只显示面
            } else if (mode == 1) {
                FragColor = edgeColor;  // 只显示边
            } else if (mode == 2) {
                FragColor = faceColor;  // 同时显示面和边
            } else if (mode == 3) {
                FragColor = texture(textureSampler, TexCoords); // 使用纹理
                //FragColor = vec4(TexCoords, 0.0, 1.0); // 使用纹理
            }
        }
        """

        # 编译着色器
        self.shader_program = self.create_shader_program()

        self.VAO = None
        self.VBO = None
        self.EBO = None

        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)

        glfw.set_scroll_callback(self.window, self.scroll_callback)

        glfw.set_window_size_callback(self.window, self.window_resize_callback)

        self.update_projection_matrix(width, height)

        self.coordinate_axes = CoordinateAxes()

    def update_projection_matrix(self, width, height):
        """
        """
        aspect_ratio = width / height
        fov = np.radians(45)  # Field of view, 45 degrees
        near = 0.1  # Near clipping plane
        far = 100.0  # Far clipping plane

        # Create a perspective projection matrix
        self.projection = np.zeros((4, 4), dtype=np.float32)
        self.projection[0, 0] = 1 / (aspect_ratio * np.tan(fov / 2))
        self.projection[1, 1] = 1 / np.tan(fov / 2)
        self.projection[2, 2] = -(far + near) / (far - near)
        self.projection[2, 3] = -(2 * far * near) / (far - near)
        self.projection[3, 2] = -1

    def load_mesh(self, nodes, cells):
        """
        @brief 加载网格数据，并根据提供的节点数据决定是否包含纹理坐标。
        
        @param nodes: 节点数组，形状可以是 (NN, 3) 或 (NN, 5)。
                      如果是 (NN, 5)，则假设前三个值为顶点坐标，后两个值为纹理坐标。
        @param cells: 单元格节点的索引数组。
        """
        vertices = nodes[cells].reshape(-1, nodes.shape[1])
        self.vertex_count = len(vertices)

        # 创建并绑定VAO
        if self.VAO is None:
            self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        # 创建并绑定VBO
        if self.VBO is None:
            self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # 设置顶点位置属性指针
        stride = vertices.shape[1] * vertices.itemsize
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, c_void_p(0))
        glEnableVertexAttribArray(0)

        # 如果有纹理坐标，设置纹理坐标属性指针
        if nodes.shape[1] == 5:
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, c_void_p(3 * vertices.itemsize))
            glEnableVertexAttribArray(1)

        # 解绑VBO和VAO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def load_texture(self, image_path):
        """
        @brief 加载纹理坐标
        """
        # 加载图片
        image = Image.open(image_path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM) # 将图片上下翻转，因为OpenGL的纹理坐标和图片的默认坐标是反的
        img_data = image.convert("RGBA").tobytes() # 转换图片为RGBA格式，并转换为字节
        # 生成纹理ID
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)

        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # 创建纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        # 解绑纹理
        glBindTexture(GL_TEXTURE_2D, 0)

        # 存储纹理ID
        self.texture = texture

    def compile_shader(self, source, shader_type):
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
            error = glGetShaderInfoLog(shader).decode('utf-8')
            raise Exception(f"Shader compile failure: {error}")
        return shader

    def create_shader_program(self):
        vertex_shader = self.compile_shader(self.vertex_shader_source, GL_VERTEX_SHADER)
        fragment_shader = self.compile_shader(self.fragment_shader_source, GL_FRAGMENT_SHADER)
        shader_program = glCreateProgram()
        glAttachShader(shader_program, vertex_shader)
        glAttachShader(shader_program, fragment_shader)
        glLinkProgram(shader_program)
        if glGetProgramiv(shader_program, GL_LINK_STATUS) != GL_TRUE:
            error = glGetProgramInfoLog(shader_program).decode('utf-8')
            raise Exception(f"Program link failure: {error}")

        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return shader_program

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            
            # 清除颜色缓冲区和深度缓冲区
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(*self.bgColor)

            # 使用着色器程序
            glUseProgram(self.shader_program)


            # 更新着色器的uniform变量
            glUniform4fv(glGetUniformLocation(self.shader_program, "faceColor"), 1, self.faceColor)
            glUniform4fv(glGetUniformLocation(self.shader_program, "edgeColor"), 1, self.edgeColor)
            glUniform1i(glGetUniformLocation(self.shader_program, "mode"), self.mode)

            # 应用变换
            transform_location = glGetUniformLocation(self.shader_program, "transform")
            if transform_location != -1:
                glUniformMatrix4fv(transform_location, 1, GL_FALSE, self.transform)
            else:
                logger.error("Transform location is invalid.")

            """
            # 应用变换
            projection_location = glGetUniformLocation(self.shader_program,
                    "projection")
            if projection_location != -1:
                glUniformMatrix4fv(projection_location, 1, GL_FALSE,
                        self.projection)
            else:
                logger.error("Projection location is invalid.")
            """

            glBindVertexArray(self.VAO)

            # 绑定纹理或设置为显示边和面的模式
            if self.texture is not None and self.mode == 3:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self.texture)
                glUniform1i(glGetUniformLocation(self.shader_program, "textureSampler"), 0)
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            elif self.mode == 3:
                self.mode = 0  # 如果没有加载纹理，则显示边

            # 如果显示模式为2，则需要两遍绘制
            if self.mode == 2:
                # 第一遍，填充面
                glUniform1i(glGetUniformLocation(self.shader_program, "mode"), 0)  # 设置为仅显示面
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

                # 第二遍，绘制边
                glUniform1i(glGetUniformLocation(self.shader_program, "mode"), 1)  # 设置为仅显示边
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # 绘制线框
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # 恢复默认模式

            elif self.mode == 0:
                # 只显示面
                glUniform1i(glGetUniformLocation(self.shader_program, "mode"), 0)
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

            elif self.mode == 1:
                # 只显示边
                glUniform1i(glGetUniformLocation(self.shader_program, "mode"), 1)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)



            # 关闭深度测试，确保坐标轴总是绘制在最前面
            glDisable(GL_DEPTH_TEST)

            # 渲染坐标轴
            self.coordinate_axes.render(self.projection, view_for_axes, np.identity(4))

            # 重新启用深度测试
            glEnable(GL_DEPTH_TEST)

            glfw.swap_buffers(self.window)

        glfw.terminate()

    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        translate_speed = 0.1
        scale_factor = 1.1
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_UP:  # 向上平移
                self.transform[3, 1] += translate_speed
            elif key == glfw.KEY_DOWN:  # 向下平移
                self.transform[3, 1] -= translate_speed
            elif key == glfw.KEY_RIGHT:  # 向右平移
                self.transform[3, 0] += translate_speed
            elif key == glfw.KEY_LEFT:  # 向左平移
                self.transform[3, 0] -= translate_speed
            elif key == glfw.KEY_M:  # 假设我们使用 M 键来切换模式
                self.mode += 1
                if self.mode > 3:  # 超出范围后重置为 0
                    self.mode = 0
            elif key == glfw.KEY_Z:  # 放大
                self.transform[:3, :3] *= scale_factor
            elif key == glfw.KEY_X:  # 缩小
                self.transform[:3, :3] /= scale_factor
            elif key == glfw.KEY_V:
                self.view_angle = (self.view_angle + 1) % 3  # 在0, 1, 2之间循环
                if self.view_angle == 0:  # X轴视角
                    self.transform = np.array([[0, 0, -1, 0],
                                               [0, 1, 0, 0],
                                               [1, 0, 0, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
                elif self.view_angle == 1:  # Y轴视角
                    self.transform = np.array([[1, 0, 0, 0],
                                               [0, 0, 1, 0],
                                               [0, -1, 0, 0],
                                               [0, 0, 0, 1]], dtype=np.float32)
                elif self.view_angle == 2:  # Z轴视角
                    self.transform = np.identity(4, dtype=np.float32)  # 默认视角


    def mouse_button_callback(self, window, button, action, mods):
        """
        """
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                self.dragging = True
                self.first_mouse_use = True
            elif action == glfw.RELEASE:
                self.dragging = False

    def mouse_callback(self, window, xpos, ypos):
        if self.dragging:
            if self.first_mouse_use:
                self.last_mouse_pos = (xpos, ypos)
                self.first_mouse_use = False
                return

            xoffset = xpos - self.last_mouse_pos[0]
            yoffset = self.last_mouse_pos[1] - ypos
            self.last_mouse_pos = (xpos, ypos)

            if xoffset == 0 and yoffset == 0:
                return

            # 计算旋转矩阵，这里仅提供概念代码，具体实现需要根据虚拟轨迹球的逻辑来完成
            rotation_matrix = calculate_rotation_matrix(xoffset, yoffset)
            self.transform = np.dot(rotation_matrix, self.transform)

    def mouse_callback_old(self, window, xpos, ypos):
        print(f"Mouse position: {xpos}, {ypos}")
        if self.first_mouse_use:
            self.last_mouse_pos = (xpos, ypos)
            self.first_mouse_use = False

        xoffset = xpos - self.last_mouse_pos[0]
        yoffset = self.last_mouse_pos[1] - ypos  # 注意这里的y方向与屏幕坐标系相反
        self.last_mouse_pos = (xpos, ypos)

        sensitivity = 0.1
        xoffset *= sensitivity
        yoffset *= sensitivity

        # 生成旋转矩阵
        # 这里简化处理，只根据xoffset和yoffset来做基本的旋转，实际应用中可能需要更复杂的旋转逻辑
        rotation_x = np.array([[1, 0, 0, 0],
                               [0, np.cos(yoffset), -np.sin(yoffset), 0],
                               [0, np.sin(yoffset), np.cos(yoffset), 0],
                               [0, 0, 0, 1]], dtype=np.float32)

        rotation_y = np.array([[np.cos(xoffset), 0, np.sin(xoffset), 0],
                               [0, 1, 0, 0],
                               [-np.sin(xoffset), 0, np.cos(xoffset), 0],
                               [0, 0, 0, 1]], dtype=np.float32)

        self.transform = np.dot(self.transform, rotation_x)
        self.transform = np.dot(self.transform, rotation_y)

        logger.debug("Rotating: X offset {}, Y offset {}".format(xoffset, yoffset))

    def scroll_callback(self, window, xoffset, yoffset):
        """鼠标滚轮回调函数，用于缩放视图。"""
        scale_factor = 1.1  # 缩放系数
        if yoffset < 0:  # 向下滚动，缩小
            scale_factor = 1.0 / scale_factor
        # 更新变换矩阵
        self.transform[:3, :3] *= scale_factor
        logger.debug("Zooming: {}".format("In" if scale_factor > 1 else "Out"))


    def window_resize_callback(self, window, width, height):
        glViewport(0, 0, width, height)
        self.update_projection_matrix(width, height)

def main():
    # 假设nodes和cells是你的网格数据

    """
    # 定义顶点数据和UV坐标
    nodes = np.array([
        [-0.5, -0.5, 0.0,  0.0, 0.0],  # 左下角
        [ 0.5, -0.5, 0.0,  1.0, 0.0],  # 右下角
        [ 0.5,  0.5, 0.0,  1.0, 1.0],  # 右上角
        [-0.5,  0.5, 0.0,  0.0, 1.0]   # 左上角
    ], dtype=np.float32)

    cells = np.array([
        0, 1, 2,
        2, 3, 0
    ], dtype=np.uint32)

    """
    from fealpy.mesh import TriangleMesh

    mesh, U, V = TriangleMesh.from_ellipsoid_surface(80, 800, 
            radius=(4, 2, 1), 
            theta=(np.pi/2, np.pi/2+np.pi/3),
            returnuv=True)
    U = (U - np.min(U))/(np.max(U)-np.min(U))
    V = (V - np.min(V))/(np.max(V)-np.min(V))
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    nodes = np.hstack((node, V.reshape(-1, 1), U.reshape(-1, 1)), dtype=np.float32)
    cells = np.array(cell, dtype=np.uint32)

    plotter = OpenGLPlotter()
    plotter.load_mesh(nodes, cells)
    plotter.load_texture('/home/why/we.jpg')
    plotter.run()

if __name__ == "__main__":
    main()

