import subprocess
import logging
from pyntcloud import PyntCloud
import numpy as np
from scipy.spatial import KDTree
import sys
import os
import re
import open3d as o3d
from PyQt5 import QtCore, QtGui, QtWidgets

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("process.log"),
    logging.StreamHandler()
])
logger = logging.getLogger()

def generate_ply(data_folder_path, output_folder_path):
    """运行 TestRangeImage.exe 来生成 PLY 文件"""
    test_range_image_exe_path = "C:\\Users\\alienware\\Desktop\\TestRangeImage\\TestRangeImage.exe"  # 修改为实际路径
    command = f'"{test_range_image_exe_path}" "{data_folder_path}" "{output_folder_path}"'
    try:
        logger.info(f"运行命令: {command}")
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("TestRangeImage.exe 输出:\n%s", result.stdout.decode())
        if result.stderr:
            logger.error("TestRangeImage.exe 错误:\n%s", result.stderr.decode())
    except subprocess.CalledProcessError as e:
        logger.error("运行 TestRangeImage.exe 时出错:\n返回码: %d\n输出: %s\n错误信息: %s", e.returncode,
                     e.output.decode(), e.stderr.decode())
        raise

def calculate_curvatures(points, tree, roiradius, threshold, erosion_ratio):
    """计算点云的曲率"""
    curvatures = []
    for point in points[['x', 'y', 'z']].values:
        idx = tree.query_ball_point(point, roiradius)
        if len(idx) < 3:
            curvatures.append(0)
            continue
        neighborhood = points.iloc[idx][['x', 'y', 'z']].values
        distances = np.linalg.norm(neighborhood - point, axis=1)
        valid_idx = distances <= (1 - erosion_ratio) * roiradius
        neighborhood = neighborhood[valid_idx]
        if len(neighborhood) < 3:
            curvatures.append(0)
            continue
        covariance = np.cov(neighborhood.T)
        eigvals = np.linalg.eigvalsh(covariance)
        curvature = eigvals[0] / np.sum(eigvals)
        curvatures.append(curvature)
    return curvatures

def process_ply_file(ply_path, output_folder_path, roiradius, threshold, erosion_ratio, density_threshold):
    """处理 PLY 文件，包括着色和生成网格"""
    logger.info(f"处理 PLY 文件: {ply_path}")
    point_cloud = PyntCloud.from_file(ply_path)
    points = point_cloud.points

    logger.info(f"点云数据加载完成，共 {len(points)} 个点")

    tree = KDTree(points[['x', 'y', 'z']].values)

    if roiradius < 0:
        logger.warning("ROI 半径为负数，使用绝对值进行计算")
        roiradius = abs(roiradius)

    curvatures = calculate_curvatures(points, tree, roiradius, threshold, erosion_ratio)

    points['curvature'] = curvatures
    points['red'] = 0
    points['green'] = 0
    points['blue'] = 0

    if roiradius == 0:
        points.loc[points['curvature'] > threshold, ['red', 'green', 'blue']] = [255, 0, 0]
    else:
        var_threshold = float(threshold)
        for i, point in points[['x', 'y', 'z']].iterrows():
            idx = tree.query_ball_point(point.values, roiradius)
            neighborhood_curvatures = points.iloc[idx]['curvature'].values
            if len(neighborhood_curvatures) < 2:
                continue
            curvature_variance = np.var(neighborhood_curvatures)
            if curvature_variance > var_threshold:
                points.loc[idx, ['red', 'green', 'blue']] = [255, 0, 0]

    points['red'] = np.clip(points['red'], 0, 255)
    points['green'] = np.clip(points['green'], 0, 255)
    points['blue'] = np.clip(points['blue'], 0, 255)

    black_points_count = len(points[(points['red'] == 0) & (points['green'] == 0) & (points['blue'] == 0)])
    red_points_count = len(points[(points['red'] == 255) & (points['green'] == 0) & (points['blue'] == 0)])
    logger.info(f"未超过曲率阈值点的个数: {black_points_count}")
    logger.info(f"超过曲率阈值点的个数: {red_points_count}")

    output_ply_path = os.path.join(output_folder_path, os.path.basename(ply_path).replace('.ply', '_colored.ply'))
    logger.info(f"输出文件: {output_ply_path}")
    with open(output_ply_path, 'w') as f:
        f.write(f"ply\n")
        f.write(f"format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write(f"property float x\n")
        f.write(f"property float y\n")
        f.write(f"property float z\n")
        f.write(f"property uchar red\n")
        f.write(f"property uchar green\n")
        f.write(f"property uchar blue\n")
        f.write(f"end_header\n")
        for i, row in points.iterrows():
            line = f"{row['x']} {row['y']} {row['z']} {int(row['red'])} {int(row['green'])} {int(row['blue'])}\n"
            f.write(line)

    def generate_mesh(pcd, output_path, density_threshold):
        """生成网格并应用密度过滤"""
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        densities = np.asarray(densities)
        densities = densities / densities.max() if densities.max() > 0 else densities

        mesh_vertices = np.asarray(mesh.vertices)
        vertex_density = np.zeros(len(mesh_vertices))

        for i, vertex in enumerate(mesh_vertices):
            distances = np.linalg.norm(points[['x', 'y', 'z']].values - vertex, axis=1)
            idx_within_radius = distances <= roiradius
            if np.sum(idx_within_radius) > 0:
                vertex_density[i] = np.mean(densities)

        valid_vertices = vertex_density >= density_threshold
        mesh = mesh.select_by_index(np.where(valid_vertices)[0])

        o3d.io.write_triangle_mesh(output_path, mesh)
        logger.info(f"保存网格文件: {output_path}")

    # 生成未着色的点云网格
    logger.info("生成未着色的点云网格")
    pcd_original = o3d.io.read_point_cloud(ply_path)
    generate_mesh(pcd_original, os.path.join(output_folder_path, os.path.basename(ply_path).replace('.ply', '_original_filtered_mesh.ply')), density_threshold)

    # 处理着色点云生成网格
    logger.info("生成着色点云网格")
    pcd_colored = o3d.io.read_point_cloud(os.path.join(output_folder_path, os.path.basename(ply_path).replace('.ply', '_colored.ply')))
    generate_mesh(pcd_colored, os.path.join(output_folder_path, os.path.basename(ply_path).replace('.ply', '_filtered_mesh.ply')), density_threshold)

def prompt_user_for_input():
    """弹出对话框获取用户输入"""
    app = QtWidgets.QApplication([])
    data_folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选择数据文件夹")
    output_folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "选择输出文件夹")

    if not data_folder_path or not output_folder_path:
        logger.error("未选择数据文件夹或输出文件夹")
        return None

    roiradius, ok1 = QtWidgets.QInputDialog.getDouble(None, "输入ROI半径", "ROI半径：", 0.1, 0.0000, 100.0, 8)
    if not ok1:
        return None

    threshold, ok2 = QtWidgets.QInputDialog.getDouble(None, "输入阈值", "曲率阈值：", 0.1, 0.0001, 1.0, 8)
    if not ok2:
        return None

    erosion_ratio, ok3 = QtWidgets.QInputDialog.getDouble(None, "输入侵蚀比", "侵蚀比：", 0.10, 0.0000, 1.0, 8)
    if not ok3:
        return None

    density_threshold, ok4 = QtWidgets.QInputDialog.getDouble(None, "输入密度阈值", "密度阈值：", 0.1, 0.0000, 1.0, 8)
    if not ok4:
        return None

    return data_folder_path, output_folder_path, roiradius, threshold, erosion_ratio, density_threshold

def process_all_subfolders(root_folder_path, output_folder_path):
    """处理根文件夹下的所有子文件夹"""
    for subdir in os.listdir(root_folder_path):
        subdir_path = os.path.join(root_folder_path, subdir)
        if os.path.isdir(subdir_path):
            # 为每个子文件夹创建一个相同名称的输出文件夹
            output_subfolder = os.path.join(output_folder_path, subdir)
            os.makedirs(output_subfolder, exist_ok=True)

            logger.info(f"处理子文件夹: {subdir_path}")
            generate_ply(subdir_path, output_subfolder)

            # 遍历输出文件夹中的 PLY 文件进行处理
            for filename in os.listdir(output_subfolder):
                if filename.endswith('.ply'):
                    ply_path = os.path.join(output_subfolder, filename)
                    try:
                        process_ply_file(ply_path, output_subfolder, roiradius, threshold, erosion_ratio, density_threshold)
                    except Exception as e:
                        logger.error(f"处理文件 {ply_path} 时出错: {e}")

class PLYViewer(QtWidgets.QWidget):
    def __init__(self,data_folder_path,base_tiff_path, parent=None):
        super(PLYViewer, self).__init__(parent)
        self.initUI()
        self.current_mode = "point_cloud"
        self.current_color_mode = "original"
        self.current_image_index = 0
        self.current_group = None
        self.init_open3d_window()
        self.base_tiff_path = base_tiff_path
        self.folder_path = data_folder_path
        self.load_groups(self.folder_path)

    def initUI(self):
        self.layout = QtWidgets.QHBoxLayout(self)

        self.control_layout = QtWidgets.QVBoxLayout()
        self.control_layout.setContentsMargins(10, 10, 10, 10)

        self.back_button = QtWidgets.QPushButton("返回", self)
        self.back_button.clicked.connect(self.show_all_group_buttons)
        self.control_layout.addWidget(self.back_button)
        self.back_button.setVisible(False)

        self.mode_button = QtWidgets.QPushButton("切换到点云/网格模式", self)
        self.mode_button.clicked.connect(self.toggle_mode)
        self.control_layout.addWidget(self.mode_button)
        self.mode_button.setVisible(False)

        self.color_mode_button = QtWidgets.QPushButton("切换到原始/着色模式", self)
        self.color_mode_button.clicked.connect(self.toggle_color_mode)
        self.control_layout.addWidget(self.color_mode_button)
        self.color_mode_button.setVisible(False)

        self.layout.addLayout(self.control_layout)

        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.VLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout.addWidget(self.separator)

        self.images_layout = QtWidgets.QGridLayout()
        self.images_layout.setContentsMargins(10, 10, 10, 10)
        self.image_labels = [QtWidgets.QLabel(self) for _ in range(8)]
        for i, label in enumerate(self.image_labels):
            self.images_layout.addWidget(label, i // 4, i % 4)

        self.layout.addLayout(self.images_layout)

        self.setLayout(self.layout)

    def init_open3d_window(self):
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window(window_name='PLY Viewer', width=800, height=600, visible=True)

    def load_groups(self,folder_path):
        self.group_buttons = {}
        for folder_name in os.listdir(folder_path):
            full_folder_path = os.path.join(folder_path, folder_name)
            if os.path.isdir(full_folder_path):
                button = QtWidgets.QPushButton(folder_name, self)
                button.clicked.connect(lambda _, path=full_folder_path: self.show_sub_groups(path))
                self.control_layout.addWidget(button)
                self.group_buttons[folder_name] = button

    def show_sub_groups(self, folder_path):
        self.current_group = folder_path
        self.viewer.clear_geometries()

        # Hide main group buttons and show back button and mode/color buttons
        self.show_specific_group_buttons()

        # Get file list and create buttons
        ply_files = [f for f in os.listdir(folder_path) if f.endswith('.ply')]
        groups = {}

        # Group files
        for file_name in ply_files:
            match = re.match(r"^(image_\d+)", file_name)
            if match:
                group_name = match.group(1)
                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append(file_name)

        self.clear_sub_group_buttons()  # Clear previous sub-group buttons
        self.sub_group_buttons = {}
        for index, (group_name, files) in enumerate(groups.items()):
            if len(files) == 4:  # Ensure each group contains four files
                button = QtWidgets.QPushButton(group_name, self)
                button.clicked.connect(lambda _, group=group_name, idx=index: self.load_group_files(group, idx,base_tiff_path=self.base_tiff_path))
                self.control_layout.addWidget(button)
                self.sub_group_buttons[group_name] = button

        # Display the original point cloud by default
        if groups:
            self.load_group_files(next(iter(groups.keys())), 0,base_tiff_path=self.base_tiff_path)

    def load_group_files(self, group_name, index,base_tiff_path):
        self.current_image_index = index  # Set current image index to the index of the clicked button
        self.viewer.clear_geometries()

        # Get current data group folder path
        folder_path = self.current_group
        file_types = {
            "original_point_cloud": r"^{}\.ply$".format(group_name),
            "colored_point_cloud": r"^{}_colored\.ply$".format(group_name),
            "original_mesh": r"^{}_filtered_mesh\.ply$".format(group_name),
            "colored_mesh": r"^{}_original_filtered_mesh\.ply$".format(group_name)
        }

        if self.current_mode == "point_cloud":
            file_pattern = file_types["original_point_cloud"] if self.current_color_mode == "original" else file_types["colored_point_cloud"]
        else:
            file_pattern = file_types["original_mesh"] if self.current_color_mode == "original" else file_types["colored_mesh"]

        ply_file = next((f for f in os.listdir(folder_path) if re.match(file_pattern, f)), None)
        if ply_file:
            ply_file_path = os.path.join(folder_path, ply_file)
            if "mesh" in ply_file:
                mesh = o3d.io.read_triangle_mesh(ply_file_path)
                if not mesh.is_empty():
                    self.viewer.add_geometry(mesh)
                else:
                    print(f"Failed to load mesh: {ply_file_path}")
            else:
                pcd = o3d.io.read_point_cloud(ply_file_path)
                if not pcd.is_empty():
                    self.viewer.add_geometry(pcd)
                else:
                    print(f"Failed to load point cloud: {ply_file_path}")

        self.viewer.poll_events()
        self.viewer.update_renderer()

        self.load_images(folder_path,base_tiff_path)

    def toggle_mode(self):
        if self.current_mode == "point_cloud":
            self.current_mode = "mesh"
            self.mode_button.setText("切换到点云/网格模式")
        else:
            self.current_mode = "point_cloud"
            self.mode_button.setText("切换到点云/网格模式")
        if self.current_group:
            group_name = next(iter(self.sub_group_buttons.keys()))
            self.load_group_files(group_name, self.current_image_index,base_tiff_path=self.base_tiff_path)

    def toggle_color_mode(self):
        if self.current_color_mode == "original":
            self.current_color_mode = "colored"
            self.color_mode_button.setText("切换到原始/着色模式")
        else:
            self.current_color_mode = "original"
            self.color_mode_button.setText("切换到原始/着色模式")
        if self.current_group:
            group_name = next(iter(self.sub_group_buttons.keys()))
            self.load_group_files(group_name, self.current_image_index,base_tiff_path=self.base_tiff_path)

    def load_images(self, folder_path,base_tiff_path):
        subfolder_name = os.path.basename(folder_path)
        tiff_folder_path = os.path.join(base_tiff_path, subfolder_name, "tiff")

        if not os.path.exists(tiff_folder_path):
            print(f"TIFF folder does not exist: {tiff_folder_path}")
            return
        tiff_files = sorted([f for f in os.listdir(tiff_folder_path) if f.endswith('.tif')])
        start_index = self.current_image_index * 8
        for i in range(8):
            if start_index + i < len(tiff_files):
                tiff_file = os.path.join(tiff_folder_path, tiff_files[start_index + i])
                pixmap = QtGui.QPixmap(tiff_file)
                if pixmap.isNull():
                    print(f"Failed to load image: {tiff_file}")
                else:
                    self.image_labels[i].setPixmap(
                        pixmap.scaled(self.image_labels[i].size(), QtCore.Qt.KeepAspectRatio))
                    print(f"Successfully loaded image: {tiff_file}")
            else:
                self.image_labels[i].clear()

    def show_all_group_buttons(self):
        self.clear_sub_group_buttons()  # Clear previous sub-group buttons
        for button in self.group_buttons.values():
            button.setVisible(True)
        self.back_button.setVisible(False)
        self.mode_button.setVisible(False)
        self.color_mode_button.setVisible(False)

    def clear_sub_group_buttons(self):
        if hasattr(self, "sub_group_buttons"):
            for button in self.sub_group_buttons.values():
                button.setVisible(False)
            self.sub_group_buttons.clear()

    def show_specific_group_buttons(self):
        self.clear_sub_group_buttons()  # Clear previous sub-group buttons
        for button in self.group_buttons.values():
            button.setVisible(False)
        self.back_button.setVisible(True)
        self.mode_button.setVisible(True)
        self.color_mode_button.setVisible(True)
        if hasattr(self, "sub_group_buttons"):
            for button in self.sub_group_buttons.values():
                button.setVisible(True)

    def closeEvent(self, event):
        self.viewer.destroy_window()
        event.accept()


if __name__ == "__main__":
    inputs = prompt_user_for_input()
    if inputs:
        data_folder_path, output_folder_path, roiradius, threshold, erosion_ratio, density_threshold = inputs
        process_all_subfolders(data_folder_path,output_folder_path)

    app = QtWidgets.QApplication(sys.argv)
    viewer = PLYViewer(output_folder_path,data_folder_path)
    viewer.resize(1200, 800)
    viewer.show()
    sys.exit(app.exec_())