import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import cv2  # 导入OpenCV库，用于图像处理
from sklearn.utils import check_random_state  # 导入check_random_state函数，用于生成随机数
from src.getpoints import get_points  # 导入get_points函数，用于获取特征点

# 读取数据
def read_data(file_path):
    """
    从文件中读取数据并解析为图像标志、特征点号、x坐标和y坐标。
    :param file_path: 数据文件路径
    :return: 包含图像标志、特征点号、x坐标和y坐标的列表
    """
    data = []  # 定义一个空列表 `data`，用于存储从文件中读取的数据

    with open(file_path, 'r') as f:  # 打开文件，文件路径由 `file_path` 指定，并以只读模式 ('r') 打开
        next(f)  # 跳过文件的第一行
        for line in f:  # 逐行读取文件内容
            if line.strip():  # 检查当前行是否为空（去除空白字符后）
                parts = line.split()  # 将当前行按空白字符分割成多个部分
                img_id = int(parts[0])  # 将第一个部分转换为整数，表示图像 ID
                point_id = int(parts[1])  # 将第二个部分转换为整数，表示点 ID
                x = float(parts[2])  # 将第三个部分转换为浮点数，表示 x 坐标
                y = float(parts[3])  # 将第四个部分转换为浮点数，表示 y 坐标
                data.append((img_id, point_id, x, y))  # 将解析后的数据以元组形式添加到 `data` 列表中
    return data  # 返回包含所有解析数据的列表 `data`

# 提取左右目特征点，并对右图特征点进行坐标系校正
def extract_points(data):
    """
    从数据中提取左右目的特征点。
    :param data: 包含图像标志、特征点号、x坐标和y坐标的列表
    :return: 左目特征点和右目特征点的字典
    """
    left_points = {}  # 创建一个空字典 left_points，用于存储左侧图像中的点
    right_points = {}  # 创建一个空字典 right_points，用于存储右侧图像中的点

    for img_id, point_id, x, y in data:  # 遍历数据中的每一行，数据格式为 (img_id, point_id, x, y)
        if img_id == 0:  # 如果 img_id 为 0，表示该点属于左侧图像
            left_points[point_id] = (x, y)  # 将点 (x, y) 存储在 left_points 字典中，键为 point_id
        elif img_id == 1:  # 如果 img_id 为 1，表示该点属于右侧图像
            right_points[point_id] = (x, y)  # 将点 (x, y) 存储在 right_points 字典中，键为 point_id

    return left_points, right_points  # 返回包含左侧和右侧图像点的字典

# 匹配特征点
def match_points(left_points, right_points):
    """
    匹配左右目的特征点。
    :param left_points: 左目特征点字典
    :param right_points: 右目特征点字典
    :return: 匹配的特征点列表
    """
    matched_points = []  # 初始化一个空列表，用于存储匹配的点对

    for point_id in left_points:  # 遍历左图中的所有点ID
        if point_id in right_points:  # 检查当前点ID是否也存在于右图中
            x_left, y_left = left_points[point_id]  # 如果存在，获取左图中该点的坐标 (x_left, y_left)
            x_right, y_right = right_points[point_id]  # 获取右图中该点的坐标 (x_right, y_right)
            matched_points.append(((x_left, y_left), (x_right, y_right)))  # 将匹配的点对添加到 matched_points 列表中

    return matched_points  # 返回所有匹配的点对

# 生成特征点矩阵
def create_point_matrix(matched_points):
    """
    将匹配的特征点转换为矩阵形式。
    :param matched_points: 匹配的特征点列表
    :return: 左目特征点矩阵和右目特征点矩阵
    """
    n = len(matched_points)  # 获取匹配点的数量
    left_matrix = np.zeros((n, 2))  # 初始化左目特征点矩阵
    right_matrix = np.zeros((n, 2))  # 初始化右目特征点矩阵
    
    for i, ((x_left, y_left), (x_right, y_right)) in enumerate(matched_points):  # 遍历匹配点列表
        left_matrix[i, 0] = x_left  # 将左目x坐标存入矩阵
        left_matrix[i, 1] = y_left  # 将左目y坐标存入矩阵
        right_matrix[i, 0] = x_right  # 将右目x坐标存入矩阵
        right_matrix[i, 1] = y_right  # 将右目y坐标存入矩阵
    
    return left_matrix, right_matrix  # 返回左目和右目特征点矩阵

# 归一化坐标
def normalize_points(points, K):
    """
    将特征点坐标归一化。
    :param points: 特征点矩阵
    :param K: 内参矩阵
    :return: 归一化后的特征点坐标
    """
    n = points.shape[0]  # 获取点的数量
    homogeneous_points = np.hstack((points, np.ones((n, 1))))  # 转换为齐次坐标 [n, 3]
    K_inv = np.linalg.inv(K)  # 内参矩阵的逆
    normalized_points = (K_inv @ homogeneous_points.T).T  # 归一化坐标
    return normalized_points[:, :2]  # 去掉齐次坐标的最后一维

# 计算本质矩阵 E
def compute_essential_matrix(left_norm, right_norm):
    """
    从归一化的特征点计算本质矩阵 E。
    :param left_norm: 左目归一化特征点
    :param right_norm: 右目归一化特征点
    :return: 本质矩阵 E
    """
    A = []  # 初始化矩阵 A
    for (x1, y1), (x2, y2) in zip(left_norm, right_norm):  # 遍历归一化后的特征点
        A.append([x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])  # 构造矩阵 A 的行
    A = np.array(A)  # 将列表转换为NumPy数组
    
    _, _, Vt = np.linalg.svd(A)  # 对 A 进行奇异值分解
    E = Vt[-1].reshape(3, 3)  # 取最小奇异值对应的向量并重塑为3x3矩阵
    
    U, S, Vt = np.linalg.svd(E)  # 对 E 进行奇异值分解
    S = np.diag([1, 1, 0])  # 构造对角矩阵，强制秩为2
    E = U @ S @ Vt  # 重新构造本质矩阵 E
    
    return E  # 返回本质矩阵 E

def ransac_update_num_iters(p, ep, model_points, max_iters):
    """
    更新RANSAC迭代次数。
    :param p: 期望的概率
    :param ep: 异常值比例
    :param model_points: 模型所需的最小点数
    :param max_iters: 最大迭代次数
    :return: 更新后的迭代次数
    """
    p = max(p, 0.0)  # 确保 p 不小于0
    p = min(p, 1.0)  # 确保 p 不大于1
    ep = max(ep, 0.0)  # 确保 ep 不小于0
    ep = min(ep, 1.0)  # 确保 ep 不大于1
    
    num = max(1.0 - p, np.finfo(float).eps)  # 计算分子
    denom = 1.0 - (1.0 - ep) ** model_points  # 计算分母
    if denom < np.finfo(float).eps:  # 如果分母接近0，返回0
        return 0
    
    num = np.log(num)  # 计算对数的分子
    denom = np.log(denom)  # 计算对数的分母
    
    return max_iters if denom >= 0 or -num >= max_iters * (-denom) else int(np.round(num / denom))  # 返回更新后的迭代次数

def find_essential_mat_ransac(left_points, right_points, camera_matrix, prob=0.999, threshold=1, max_iters=1000):
    """
    使用RANSAC算法找到本质矩阵 E。
    :param left_points: 左目特征点
    :param right_points: 右目特征点
    :param camera_matrix: 内参矩阵
    :param prob: 期望的概率
    :param threshold: 阈值
    :param max_iters: 最大迭代次数
    :return: 本质矩阵 E 和内点索引
    """
    left_norm = normalize_points(left_points, camera_matrix)  # 归一化左目特征点
    right_norm = normalize_points(right_points, camera_matrix)  # 归一化右目特征点
    model_points = 8  # 计算本质矩阵所需的最小点数
    best_E = None  # 初始化最佳本质矩阵
    best_inliers = None  # 初始化最佳内点索引
    best_num_inliers = 0  # 初始化最佳内点数量
    
    rng = check_random_state(0)  # 初始化随机数生成器
    
    for _ in range(max_iters):  # 进行最大迭代次数的循环
        indices = rng.choice(len(left_norm), model_points, replace=False)  # 随机选择 model_points 个点
        sample_left = left_norm[indices]  # 获取左目样本点
        sample_right = right_norm[indices]  # 获取右目样本点
        
        E = compute_essential_matrix(sample_left, sample_right)  # 从样本中计算本质矩阵
        
        inliers = []  # 初始化内点列表
        for i in range(len(left_norm)):  # 遍历所有点
            p1 = np.array([left_norm[i][0], left_norm[i][1], 1])  # 左目点齐次坐标
            p2 = np.array([right_norm[i][0], right_norm[i][1], 1])  # 右目点齐次坐标
            error = np.abs(p2.T @ E @ p1)  # 计算误差
            if error < threshold:  # 如果误差小于阈值，则为内点
                inliers.append(i)
        
        if len(inliers) > best_num_inliers:  # 如果当前模型的内点更多，则更新最佳模型
            best_num_inliers = len(inliers)
            best_E = E
            best_inliers = inliers
            
            max_iters = ransac_update_num_iters(prob, 1.0 - (best_num_inliers / len(left_norm)), model_points, max_iters)  # 更新迭代次数
    
    return best_E, best_inliers  # 返回最佳本质矩阵和内点索引

# 分解本质矩阵 E 得到 R 和 t
def decompose_essential_matrix(E):
    """
    分解本质矩阵 E 得到旋转矩阵 R 和平移向量 t。
    :param E: 本质矩阵
    :return: 旋转矩阵列表和平移向量列表
    """
    U, _, Vt = np.linalg.svd(E)  # 对 E 进行奇异值分解
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 旋转矩阵的辅助矩阵
    
    R1 = U @ W @ Vt  # 计算第一个旋转矩阵
    R2 = U @ W.T @ Vt  # 计算第二个旋转矩阵
    t1 = U[:, 2]  # 计算第一个平移向量
    t2 = -U[:, 2]  # 计算第二个平移向量
    
    return [R1, R2], [t1, t2]  # 返回旋转矩阵和平移向量列表

# 计算基础矩阵 F
def compute_fundamental_matrix(E, K):
    """
    计算基础矩阵 F。
    :param E: 本质矩阵
    :param K: 内参矩阵
    :return: 基础矩阵 F
    """
    K_inv = np.linalg.inv(K)  # 计算内参矩阵的逆
    F = K_inv.T @ E @ K_inv  # 计算基础矩阵 F = K^{-T} E K^{-1}
    return F  # 返回基础矩阵 F

# 三角化选择正确的 R 和 t
def triangulate_points(left_points, right_points, R, t, K):
    """
    通过三角化计算三维点。
    :param left_points: 左目特征点
    :param right_points: 右目特征点
    :param R: 旋转矩阵
    :param t: 平移向量
    :param K: 内参矩阵
    :return: 三维点坐标
    """
    n = left_points.shape[0]  # 获取点的数量
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # 左目投影矩阵
    P2 = K @ np.hstack((R, t.reshape(3, 1)))  # 右目投影矩阵
    
    points_3d = np.zeros((n, 3))  # 初始化三维点矩阵
    for i in range(n):  # 遍历所有点
        x1, y1 = left_points[i]  # 获取左目点坐标
        x2, y2 = right_points[i]  # 获取右目点坐标
        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])  # 构造矩阵 A
        _, _, Vt = np.linalg.svd(A)  # 对 A 进行奇异值分解
        X = Vt[-1]  # 取最小奇异值对应的向量
        points_3d[i] = X[:3] / X[3]  # 转换为非齐次坐标
    return points_3d  # 返回三维点坐标

# 计算重投影误差
def compute_reprojection_error(points_3d, left_points, right_points, K, R, t):
    """
    计算重投影误差。
    :param points_3d: 三维点坐标
    :param left_points: 左目特征点
    :param right_points: 右目特征点
    :param K: 内参矩阵
    :param R: 旋转矩阵
    :param t: 平移向量
    :return: 左目和右目的平均重投影误差，以及重投影点
    """
    n = points_3d.shape[0]  # 获取点的数量
    
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # 左目投影矩阵
    P2 = K @ np.hstack((R, t.reshape(3, 1)))  # 右目投影矩阵
    
    homogeneous_points = np.hstack((points_3d, np.ones((n, 1))))  # 转换为齐次坐标
    
    projected_left = (P1 @ homogeneous_points.T).T  # 左目重投影点
    projected_left = projected_left[:, :2] / projected_left[:, 2].reshape(-1, 1)  # 转换为非齐次坐标
    
    projected_right = (P2 @ homogeneous_points.T).T  # 右目重投影点
    projected_right = projected_right[:, :2] / projected_right[:, 2].reshape(-1, 1)  # 转换为非齐次坐标
    
    error_left = np.linalg.norm(projected_left - left_points, axis=1)  # 计算左目重投影误差
    error_right = np.linalg.norm(projected_right - right_points, axis=1)  # 计算右目重投影误差
    
    return np.mean(error_left), np.mean(error_right), projected_left, projected_right  # 返回平均误差和重投影点

def find_valid_R_t(R_list, t_list, left_matrix, right_matrix, K):
    """
    遍历所有可能的 R 和 t 组合，找到有效的旋转矩阵和平移向量。
    :param R_list: 旋转矩阵列表
    :param t_list: 平移向量列表
    :param left_matrix: 左目特征点矩阵
    :param right_matrix: 右目特征点矩阵
    :param K: 内参矩阵
    :return: 有效的旋转矩阵、平移向量和三维点坐标
    """
    for i, R in enumerate(R_list):  # 遍历旋转矩阵列表
        for j, t in enumerate(t_list):  # 遍历平移向量列表
            points_3d = triangulate_points(left_matrix, right_matrix, R, t, K)  # 三角化计算三维点
            if np.all(points_3d[:, 2] > 0):  # 如果所有点的深度为正，则返回有效的 R 和 t
                return R, t, points_3d
    return None, None, None  # 如果没有找到满足条件的 R 和 t

# 可视化
def visualize(points_3d, left_points, right_points, projected_left, projected_right, t):
    """
    可视化重投影后的图像和三维点。
    :param points_3d: 三维点坐标
    :param left_points: 左目特征点 [n, 2]
    :param right_points: 右目特征点 [n, 2]
    :param projected_left: 左目重投影点 [n, 2]
    :param projected_right: 右目重投影点 [n, 2]
    :param t: 平移向量
    """
    plt.figure(figsize=(12, 6))  # 创建一个新的图形窗口，大小为12x6
    
    plt.subplot(2, 2, 1)  # 创建2x2的子图，选择第1个子图
    plt.scatter(left_points[:, 0], left_points[:, 1], c='r', marker='o')  # 绘制左目特征点
    plt.title('Original Left Image')  # 设置子图标题
    
    plt.subplot(2, 2, 2)  # 创建2x2的子图，选择第2个子图
    plt.scatter(right_points[:, 0], right_points[:, 1], c='b', marker='o')  # 绘制右目特征点
    plt.title('Original Right Image')  # 设置子图标题
    
    plt.subplot(2, 2, 3)  # 创建2x2的子图，选择第3个子图
    plt.scatter(projected_left[:, 0], projected_left[:, 1], c='r', marker='o')  # 绘制左目重投影点
    plt.title('Projected Left Image')  # 设置子图标题

    plt.subplot(2, 2, 4)  # 创建2x2的子图，选择第4个子图
    plt.scatter(projected_right[:, 0], projected_right[:, 1], c='b', marker='o')  # 绘制右目重投影点
    plt.title('Projected Right Image')  # 设置子图标题

    plt.suptitle("2D Image")  # 设置整个图形的标题

    plt.figure()  # 创建一个新的图形窗口
    ax = plt.axes(projection='3d')  # 创建3D坐标系

    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='g', marker='o', label='3D Points')  # 绘制三维点

    ax.scatter(0, 0, 0, c='r', marker='^', s=100, label='Left camera')  # 绘制左相机位置
    ax.scatter(t[0], t[1], t[2], c='b', marker='^', s=100, label='Right camera')  # 绘制右相机位置

    ax.legend(loc='lower right')  # 设置图例位置

    plt.show()  # 显示图形

# 主函数
def main():
    """
    主函数，执行整个流程。
    """
    # 获取点坐标（该功能仅能够提取点坐标，但无法进行匹配）
    # left_image = cv2.imread("data/0.JPG",cv2.IMREAD_GRAYSCALE)
    # right_image = cv2.imread("data/1.JPG",cv2.IMREAD_GRAYSCALE)
    # get_points(left_image,right_image,display=True)
    
    # 提取特征点
    data = read_data('data/1.pix')  # 从文件中读取数据
    left_points, right_points = extract_points(data)  # 从数据中提取左右目特征点
    matched_points = match_points(left_points, right_points)  # 匹配左右目特征点
    left_matrix, right_matrix = create_point_matrix(matched_points)  # 将匹配的特征点转换为矩阵形式

    # 引入内参矩阵
    K = np.array([[4705.5337, 0, 3944.07],
                [0, 4705.5337, 2612.675],
                [0, 0, 1]])


    #归一化特征点
    left_norm = normalize_points(left_matrix, K)  # 归一化左目特征点
    right_norm = normalize_points(right_matrix, K)  # 归一化右目特征点

    # 计算本质矩阵 E
    # E,_=cv2.findEssentialMat(left_matrix,right_matrix,K) # OpenCV方法，精度最高
    E = compute_essential_matrix(left_norm,right_norm)  # 应用所有点计算本质矩阵，精度次之
    # E,_=find_essential_mat_ransac(left_matrix,right_matrix,K) # 待完善的RANSAC方法，精度最低
    
    # 计算基础矩阵 F
    F=compute_fundamental_matrix(E,K)

    # 分解本质矩阵 E 得到 R 和 t
    R_list, t_list = decompose_essential_matrix(E)
    
    # 找到合理的 R 和 t 组合
    R,t,points_3d=find_valid_R_t(R_list, t_list, left_matrix, right_matrix, K) 

    # 计算重投影误差
    left_error, right_error, projected_left, projected_right = compute_reprojection_error(points_3d, left_matrix, right_matrix, K, R, t)  # 计算重投影误差
   
    #输出结果
    print(f"左目误差: {left_error}, 右目误差: {right_error}")  # 打印左右目重投影误差
    print(f"旋转矩阵 R: \n{R}")  # 打印旋转矩阵
    print(f"平移向量 t: \n{t}")  # 打印平移向量
    print(f"基础矩阵 F: \n{F}")  # 打印基础矩阵
    visualize(points_3d,left_matrix, right_matrix, projected_left, projected_right,t)  # 可视化重投影后的图像和三维点

if __name__=="__main__":
    main()  # 执行主函数