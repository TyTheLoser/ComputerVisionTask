# 项目概述

本项目实现了从双目图像中提取特征点，计算本质矩阵、基础矩阵、旋转矩阵和平移向量，并通过三角化方法恢复三维点坐标。最后，对计算结果进行了可视化展示。

# 环境配置

## 使用 `pip` 安装依赖

```sh
pip install -r requirements.txt
```

## 使用 Docker 构建和运行容器

```sh
docker build -t my_project .
docker run -it my_project
```

# 项目结构

```
.
├── main.py                  # 主程序，实现特征点提取、矩阵计算、三角化和可视化
├── src
│   └── getpoints.py         # 从图像中提取特征点的函数
├── requirements.txt         # 项目依赖库
├── README.md                # 项目说明文档
└── 1.pix                    # 输入数据文件
```

# 代码功能

## 数据读取与特征点提取

- `read_data(file_path)`: 从文件中读取数据并解析为图像标志、特征点号、x坐标和y坐标。
- `extract_points(data)`: 从数据中提取左右目的特征点。
- `match_points(left_points, right_points)`: 匹配左右目的特征点。
- `create_point_matrix(matched_points)`: 将匹配的特征点转换为矩阵形式。

## 归一化与矩阵计算

- `normalize_points(points, K)`: 将特征点坐标归一化。
- `compute_essential_matrix(left_norm, right_norm)`: 从归一化的特征点计算本质矩阵 $E$。
- `compute_fundamental_matrix(E, K)`: 计算基础矩阵 $F$。

## 旋转矩阵与平移向量

- `decompose_essential_matrix(E)`: 分解本质矩阵 $E$ 得到旋转矩阵 $R$ 和平移向量 $t$。
- `triangulate_points(left_points, right_points, R, t, K)`: 通过三角化计算三维点。
- `find_valid_R_t(R_list, t_list, left_matrix, right_matrix, K)`: 遍历所有可能的 $R$ 和 $t$ 组合，找到有效的旋转矩阵和平移向量。

## 重投影误差计算

- `compute_reprojection_error(points_3d, left_points, right_points, K, R, t)`: 计算重投影误差。

## 可视化

- `visualize(points_3d, left_points, right_points, projected_left, projected_right, t)`: 可视化重投影后的图像和三维点。

# 运行流程

1. **数据读取与特征点提取**：从 `1.pix` 文件中读取数据，提取左右目的特征点并进行匹配。
2. **归一化与矩阵计算**：对特征点进行归一化处理，计算本质矩阵 $E$ 和基础矩阵 $F$。
3. **旋转矩阵与平移向量**：通过分解本质矩阵 $E$ 得到旋转矩阵 $R$ 和平移向量 $t$，并通过三角化方法恢复三维点坐标。
4. **重投影误差计算**：计算重投影误差，评估模型的准确性。
5. **可视化**：展示原始图像、重投影图像以及三维点的空间分布。

# 结果展示

运行 `main.py` 后，程序将输出以下内容：

- 左右目的重投影误差。
- 旋转矩阵 $R$ 和平移向量 $t$。
- 基础矩阵 $F$。
- 可视化结果：包括原始图像、重投影图像以及三维点的空间分布。

# 数学原理

## 本质矩阵 $E$ 的计算

本质矩阵 $E$ 描述了两个相机之间的相对位置和姿态，满足对极几何约束：
$$
\mathbf{x}_2^T E \mathbf{x}_1 = 0
$$
其中，$\mathbf{x}_1$ 和 $\mathbf{x}_2$ 是左右相机中匹配的特征点的归一化坐标（即去除了内参矩阵 $K$ 的影响）：
$$
\mathbf{x} = K^{-1} \mathbf{P}
$$
其中 $\mathbf{P}$ 为图像坐标。

### 对极几何约束的展开

对于一对匹配的归一化特征点 $\mathbf{x}_1 = (x_1, y_1, 1)^T$ 和 $\mathbf{x}_2 = (x_2, y_2, 1)^T$，对极几何约束可以展开为：
$$
x_2 (e_{11} x_1 + e_{12} y_1 + e_{13}) + y_2 (e_{21} x_1 + e_{22} y_1 + e_{23}) + (e_{31} x_1 + e_{32} y_1 + e_{33}) = 0
$$
其中，$E$ 的元素为：
$$
E = \begin{bmatrix}e_{11} & e_{12} & e_{13} \\e_{21} & e_{22} & e_{23} \\e_{31} & e_{32} & e_{33}
\end{bmatrix}
$$
将上述方程整理为关于 $E$ 的线性方程：
$$
x_2 x_1 e_{11} + x_2 y_1 e_{12} + x_2 e_{13} + y_2 x_1 e_{21} + y_2 y_1 e_{22} + y_2 e_{23} + x_1 e_{31} + y_1 e_{32} + e_{33} = 0
$$

### 构造矩阵 $A$

对于每一对匹配的特征点，可以构造一个方程。假设有 $n$ 对匹配点，则可以构造一个 $n \times 9$ 的矩阵 $A$，其中每一行对应一个特征点对的方程：
$$
A = \begin{bmatrix}
x_2^{(1)} x_1^{(1)} & x_2^{(1)} y_1^{(1)} & x_2^{(1)} & y_2^{(1)} x_1^{(1)} & y_2^{(1)} y_1^{(1)} & y_2^{(1)} & x_1^{(1)} & y_1^{(1)} & 1 \\x_2^{(2)} x_1^{(2)} & x_2^{(2)} y_1^{(2)} & x_2^{(2)} & y_2^{(2)} x_1^{(2)} & y_2^{(2)} y_1^{(2)} & y_2^{(2)} & x_1^{(2)} & y_1^{(2)} & 1 \\\vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots \\x_2^{(n)} x_1^{(n)} & x_2^{(n)} y_1^{(n)} & x_2^{(n)} & y_2^{(n)} x_1^{(n)} & y_2^{(n)} y_1^{(n)} & y_2^{(n)} & x_1^{(n)} & y_1^{(n)} & 1
\end{bmatrix}
$$

### 求解本质矩阵 $E$

矩阵 $A$ 的每一行对应一个特征点对的约束方程。为了求解 $E$，我们需要找到满足 $A \mathbf{e} = 0$ 的向量 $\mathbf{e}$，其中 $\mathbf{e}$ 是 $E$ 的扁平化形式：
$$
\mathbf{e} = \begin{bmatrix}
e_{11} & e_{12} & e_{13} & e_{21} & e_{22} & e_{23} & e_{31} & e_{32} & e_{33}
\end{bmatrix}^T
$$
通过奇异值分解（SVD）求解 $A$：
$$
A = U \Sigma V^T
$$
其中，$V^T$ 的最后一列对应最小奇异值，即为 $\mathbf{e}$ 的解。将其重塑为 $3 \times 3$ 矩阵，得到初始的 $E$。

### 强制秩为 2

本质矩阵 $E$ 的秩必须为 2。因此，对初始的 $E$ 进行奇异值分解：
$$
E = U \Sigma V^T
$$
将奇异值矩阵 $\Sigma$ 调整为：
$$
\Sigma = \begin{bmatrix}\sigma_1 & 0 & 0 \\0 & \sigma_2 & 0 \\0 & 0 & 0
\end{bmatrix}
$$
然后重新构造 $E$：
$$
E = U \Sigma V^T
$$

## 基础矩阵 $F$ 的计算

基础矩阵 $F$ 描述了两个相机之间的投影关系，它与本质矩阵 $E$ 密切相关。基础矩阵 $F$ 和本质矩阵 $E$ 的关系由相机的内参矩阵 $K$ 决定。具体关系为：
$$
F = K^{-T} E K^{-1}
$$

## 旋转矩阵 $R$ 和平移向量 $t$

### 平移向量 $t$ 和旋转矩阵 $R$ 的计算方法

平移向量 $t$ 和旋转矩阵 $R$ 是通过分解本质矩阵 $E$ 得到的,本质矩阵 $E$ 可以通过奇异值分解（SVD）分解为：
$$
E = U \Sigma V^T
$$
其中：

- $U$ 和 $V$ 是正交矩阵。
- $\Sigma$ 是对角矩阵，其对角线元素为奇异值。

本质矩阵 $E$ 的奇异值分解满足以下性质：
$$
\Sigma = \text{diag}(\sigma_1, \sigma_2, 0)
$$
其中 $\sigma_1$ 和 $\sigma_2$ 是正奇异值。

### 构造旋转矩阵 $R$ 和平移向量 $t$

从 $E$ 的奇异值分解中，可以构造两个可能的旋转矩阵 $R_1$ 和 $R_2$，以及两个可能的平移向量 $t_1$ 和 $t_2$。具体步骤如下：

   #### 定义辅助矩阵 $W$：
   
$$
W = \begin{bmatrix}0 & -1 & 0 \\1 & 0 & 0 \\0 & 0 & 1\end{bmatrix}
$$
   
   #### 计算旋转矩阵 $R_1$ 和 $R_2$：
   
$$
R_1 = U W V^T
$$
$$
R_2 = U W^T V^T
$$
   
#### 计算平移向量 $t_1$ 和 $t_2$：
   
$$
t_1 = U[:, 2]
$$
$$
t_2 = -U[:, 2]
$$
其中，$U[:, 2]$ 是矩阵 $U$ 的第三列。

### 选择正确的 $R$ 和 $t$

在通过本质矩阵 $E$ 分解得到旋转矩阵 $R$ 和平移向量 $t$ 后，通常会得到四种可能的 $(R, t)$ 组合（即 $R_1$ 和 $R_2$ 分别与 $t_1$ 和 $t_2$ 的组合）。为了选择正确的 $(R, t)$，需要通过三角化方法计算三维点，并进行深度测试。

#### 三角化

三角化是从两个相机的图像点恢复三维空间点的过程。给定两个相机的投影矩阵 $P_1$ 和 $P_2$，以及对应的图像点 $\mathbf{x}_1$ 和 $\mathbf{x}_2$，目标是找到三维点 $\mathbf{X}$，使得：
$$
\mathbf{x}_1 = P_1 \mathbf{X}, \quad \mathbf{x}_2 = P_2 \mathbf{X}
$$

##### 构造线性方程组

对于每个相机，可以写出以下方程：
$$
\mathbf{x}_1 \times (P_1 \mathbf{X}) = 0, \quad \mathbf{x}_2 \times (P_2 \mathbf{X}) = 0
$$
其中 $\times$ 表示叉积。展开后得到：
$$
\begin{bmatrix}x_1 (P_1^{(3)} \mathbf{X}) - (P_1^{(1)} \mathbf{X}) \\y_1 (P_1^{(3)} \mathbf{X}) - (P_1^{(2)} \mathbf{X}) \\x_2 (P_2^{(3)} \mathbf{X}) - (P_2^{(1)} \mathbf{X}) \\y_2 (P_2^{(3)} \mathbf{X}) - (P_2^{(2)} \mathbf{X})
\end{bmatrix} = 0
$$
其中 $P^{(i)}$ 表示投影矩阵的第 $i$ 行。

##### 构造矩阵 $A$

将上述方程整理为矩阵形式 $A \mathbf{X} = 0$，其中：
$$
A = \begin{bmatrix}
x_1 P_1^{(3)} - P_1^{(1)} \\y_1 P_1^{(3)} - P_1^{(2)} \\x_2 P_2^{(3)} - P_2^{(1)} \\y_2 P_2^{(3)} - P_2^{(2)}
\end{bmatrix}
$$

##### 求解三维点 $\mathbf{X}$

通过奇异值分解（SVD）矩阵 $A$：
$$
A = U \Sigma V^T
$$
其中，$V^T$ 的最后一列对应最小奇异值，即为 $\mathbf{X}$ 的解。将其归一化，得到三维点的齐次坐标 $\mathbf{X} = (X, Y, Z, 1)^T$。

#### 深度测试

在得到三维点 $\mathbf{X}$ 后，需要检查其深度（即 $Z$ 坐标）是否为正。如果所有点的深度为正，则该 $(R, t)$ 组合是正确的。

## 代码功能

### 数据读取与特征点提取

- `read_data(file_path)`: 从文件中读取数据并解析为图像标志、特征点号、x坐标和y坐标。
- `extract_points(data)`: 从数据中提取左右目的特征点。
- `match_points(left_points, right_points)`: 匹配左右目的特征点。
- `create_point_matrix(matched_points)`: 将匹配的特征点转换为矩阵形式。

### 归一化与矩阵计算

- `normalize_points(points, K)`: 将特征点坐标归一化。
- `compute_essential_matrix(left_norm, right_norm)`: 从归一化的特征点计算本质矩阵 $E$。
- `compute_fundamental_matrix(E, K)`: 计算基础矩阵 $F$。

### 旋转矩阵与平移向量

- `decompose_essential_matrix(E)`: 分解本质矩阵 $E$ 得到旋转矩阵 $R$ 和平移向量 $t$。
- `triangulate_points(left_points, right_points, R, t, K)`: 通过三角化计算三维点。
- `find_valid_R_t(R_list, t_list, left_matrix, right_matrix, K)`: 遍历所有可能的 $R$ 和 $t$ 组合，找到有效的旋转矩阵和平移向量。

### 重投影误差计算

- `compute_reprojection_error(points_3d, left_points, right_points, K, R, t)`: 计算重投影误差。

### 可视化

- `visualize(points_3d, left_points, right_points, projected_left, projected_right, t)`: 可视化重投影后的图像和三维点。

## 运行流程

1. **数据读取与特征点提取**：从 `1.pix` 文件中读取数据，提取左右目的特征点并进行匹配。
2. **归一化与矩阵计算**：对特征点进行归一化处理，计算本质矩阵 $E$ 和基础矩阵 $F$。
3. **旋转矩阵与平移向量**：通过分解本质矩阵 $E$ 得到旋转矩阵 $R$ 和平移向量 $t$，并通过三角化方法恢复三维点坐标。
4. **重投影误差计算**：计算重投影误差，评估模型的准确性。
5. **可视化**：展示原始图像、重投影图像以及三维点的空间分布。

## 结果展示

运行 `main.py` 后，程序将输出以下内容：

- 左右目的重投影误差。
- 旋转矩阵 $R$ 和平移向量 $t$。
- 基础矩阵 $F$。
- 可视化结果：包括原始图像、重投影图像以及三维点的空间分布。

## 依赖库

- `numpy`: 用于数值计算。
- `matplotlib`: 用于绘图。
- `opencv-python`: 用于图像处理。
- `scikit-learn`: 用于随机数生成。

## 注意事项

- 本项目假设输入的图像数据已经过校正，且特征点已经匹配。
- 内参矩阵 $K$ 需要根据实际相机参数进行调整。
