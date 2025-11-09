# Gaussian Category Prior 预训练使用手册

本文档面向需要复现/调整 Gaussian Category Prior（GCP）预训练流程的研发同学，提供从环境准备、数据组织、预处理、训练到评估与监控的完整中文说明。按照本文流程即可在 Pix3D、ScanNetV2、KITTI、vKITTI2 等数据集上产出类别级高斯模板先验。

---
## 1. 环境准备

### 1.1 创建 Conda 环境
```bash
conda env create -f env/environment.yml
conda activate gcp-prior
```
> `environment.yml` 已包含 PyTorch、TensorBoard、pytorch3d、trimesh 等依赖，可直接在具备 CUDA 12.1 的机器上使用。如需其他 CUDA 版本，请自行在 `environment.yml` 中调整 `pytorch`/`pytorch-cuda` 版本后再创建环境。

### 1.2 代码目录结构
```
gcp_prior/
├─ configs/            # 训练配置（基础 + 各数据集）
├─ data/               # （可选）额外数据存放位置
├─ datasets/           # PyTorch Dataset 封装
├─ evaluation/         # 训练后评估脚本与指标实现
├─ preprocess/         # mesh canonical 化、采样、索引构建
├─ scripts/            # 一键脚本（预处理 / 训练 / 导出）
├─ train/              # 训练主脚本、导出工具
├─ utils/              # 几何、渲染、训练辅助函数
└─ outputs/            # 默认的 checkpoint、日志与先验输出
```

---
## 2. 数据准备

### 2.1 数据集路径约定
假设原始数据已解压至：

| 数据集   | 原始路径                              |
|----------|---------------------------------------|
| Pix3D    | `/media/pc/D/datasets/pix3d`          |
| ScanNetV2| `/media/pc/D/datasets/scannet`        |
| KITTI    | `/media/pc/D/datasets/kitti`          |
| vKITTI2  | `/media/pc/D/datasets/vkitti2`        |

### 2.2 在配置文件中写明原始数据路径
不再通过命令行传参或建立符号链接。请直接在对应数据集的配置文件中填写原始路径，且保持与官方发布的数据目录结构一致，例如在 `configs/pix3d.yaml` 中：

```yaml
paths:
  dataset_root: /media/pc/D/datasets/pix3d        # Pix3D 官方根目录（包含 model/、mask/ 等）
  raw_mesh_dir: model                              # 相对 `dataset_root`，沿用官方的 model/ 子目录
  mask_dir: mask                                   # 同理，沿用官方的 mask/ 子目录
  annotations_json: pix3d.json                     # 使用官方发布的 pix3d.json 注释文件
```

其他数据集同理，分别在 `configs/scannetv2.yaml`、`configs/kitti.yaml`、`configs/vkitti2.yaml` 中填写，务必引用官方提供的目录名称（例如 ScanNet 的 `scans/`、`scannet_frames_25k/`，KITTI 的 `training/` 子目录，vKITTI2 的 `SceneXX/` 树形结构），而非自行重命名。脚本会自动从配置中读取这些字段并在仓库内的 `outputs/preprocessed/<dataset>/` 生成索引文件。

---
## 3. 预处理流程

每个数据集都需要运行对应脚本以完成 canonical 对齐、Occupancy 采样与索引构建。脚本会自动读取配置文件中的 `paths.*` 字段，例如：
```bash
bash scripts/preprocess_pix3d.sh configs/pix3d.yaml
```
默认输出目录由配置文件的 `data.index_file` 推断（即 `outputs/preprocessed/pix3d`）。流程依次执行：
1. `preprocess/canonicalize_meshes.py`：将 mesh 平移至原点、缩放至单位球并统一朝向，输出 canonical mesh 与变换参数。
2. `preprocess/sample_points_occ.py`：从 canonical mesh 表面与体内/体外采样，生成 Occupancy/TSDF 监督点。
3. `preprocess/prepare_pix3d_metadata.py`：把官方 `pix3d.json` 注释整理为摄像机参数文件 `outputs/preprocessed/pix3d/cameras.json`。
4. `preprocess/build_index.py`：整合 mask、相机参数、Occupancy 路径等信息，生成 `index_*.npy`。

其他数据集对应脚本：
```bash
bash scripts/preprocess_scannetv2.sh configs/scannetv2.yaml
bash scripts/preprocess_kitti.sh configs/kitti.yaml
bash scripts/preprocess_vkitti2.sh configs/vkitti2.yaml
```
> 预处理时间依赖于 mesh 数量与采样点数；可在脚本内调整 `--n_surf`、`--n_uniform` 等参数控制精度与耗时。

### 3.1 利用 64 核 CPU 加速预处理

数据预处理阶段包含大量独立的文件级任务，现已在下列脚本中加入多进程开关，可充分利用多核 CPU：

| 脚本 | 新增关键参数 | 说明 |
|------|--------------|------|
| `preprocess/canonicalize_meshes.py` | `--workers` | 将 mesh canonical 化任务拆分到多个进程；默认使用 `os.cpu_count()`，设置为 `1` 即可恢复单进程。 |
| `preprocess/sample_points_occ.py` | `--workers` | 并行采样表面点与 Occupancy 点，建议与磁盘带宽协同调整。 |
| `preprocess/init_gaussians.py` | `--workers`、`--nn_jobs` | `--workers` 控制进程数，`--nn_jobs` 会传递给 `sklearn.NearestNeighbors`，用于控制每个进程内部的线程数（例如设置为 `8`）。 |

建议在正式运行前手动执行一次以验证性能：

```bash
# Canonical 化
python preprocess/canonicalize_meshes.py \
  --src <raw_mesh_dir> \
  --dst <canonical_mesh_dir> \
  --workers 64

# Occupancy 采样
python preprocess/sample_points_occ.py \
  --mesh_dir <canonical_mesh_dir> \
  --out <occ_output_dir> \
  --workers 64

# 高斯初始化（例如 16 个进程，每个进程 8 线程）
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python preprocess/init_gaussians.py \
  --points_dir <occ_output_dir> \
  --out <gaussian_output_dir> \
  --workers 16 \
  --nn_jobs 8
```

> **关于 `BrokenProcessPool` 的提示**：部分第三方几何/线性代数库在 Linux 上采用 `fork` 后会出现段错误或被系统杀死，表现为主进程报错
> `concurrent.futures.process.BrokenProcessPool`。脚本已强制使用 `spawn` 上下文并在异常发生时自动回退至单进程重试；如果仍然遇到该
> 提示，请降低 `--workers` 或调低 `OMP_NUM_THREADS`/`MKL_NUM_THREADS`，并根据终端日志定位具体文件。

> **线程/进程配比建议**：在 2×32 核 Intel Xeon 8336C + 256 GB 内存环境下，可先尝试 `workers × nn_jobs ≈ 64` 的组合，并通过 `htop` 或 `pidstat` 观察 CPU 利用率与上下文切换，必要时调整线程环境变量（`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS`）。

若需要验证/测试索引，可在预处理完成后手动执行：
```bash
python preprocess/build_index.py \
  --mask_dir /media/pc/D/datasets/pix3d/mask \
  --occ_dir outputs/preprocessed/pix3d/occ_npz \
  --cams outputs/preprocessed/pix3d/cameras.json \
  --split val \
  --out outputs/preprocessed/pix3d/index_val.npy
```
> `cameras.json` 由预处理脚本自动从官方 `pix3d.json` 转换生成，可根据需要手动指定不同的 split。
其余数据集同理，将路径替换为配置文件中填写的值。

---
## 4. 配置说明

- `configs/base.yaml`：通用基础配置（优化器、监控开关等），会被其他配置 `include`。
- `configs/model_gcp.yaml`：默认模型规模（高斯数量、类别数等）。
- `configs/{pix3d,scannetv2,kitti,vkitti2}.yaml`：数据集专属配置，指定索引文件、类别列表、预处理输出目录等。

如需自定义：
1. 复制相应 YAML。
2. 修改 `paths.*` 为实际原始数据路径，并根据需要调整 `data.dataset_name`（会用于 checkpoint 命名）。
3. 修改 `data.index_file` 指向新的索引路径。
4. 调整 `train.batch_size`、`model.K`、`monitoring.*` 等字段。

---
## 5. 启动训练

### 5.1 直接使用脚本
每个数据集提供一键训练 + 导出脚本（包含 TensorBoard 配置）：
```bash
bash scripts/train_pix3d.sh configs/pix3d.yaml
bash scripts/train_scannetv2.sh configs/scannetv2.yaml
bash scripts/train_kitti.sh configs/kitti.yaml
bash scripts/train_vkitti2.sh configs/vkitti2.yaml
```
脚本默认：
- 调用 `python train/train_gcp.py --config configs/<dataset>.yaml`
- 训练完成后执行 `python train/export_priors.py`，使用保存的 `gcp_final_<dataset>.pt` 导出高斯模板至 `outputs/priors/<dataset>/cat_XX.npz`
- TensorBoard 日志写入 `outputs/tensorboard/<dataset>/`

### 5.2 手动执行（示例）
```bash
python train/train_gcp.py \
  --config configs/pix3d.yaml \
  --override train.out_dir=outputs/checkpoints/pix3d_run1 \
  --override monitoring.tensorboard.log_dir=outputs/tensorboard/pix3d_run1

python train/export_priors.py \
  --ckpt outputs/checkpoints/pix3d_run1/gcp_final_pix3d.pt \
  --config configs/pix3d.yaml \
  --out outputs/priors/pix3d_run1
```
> 可通过多次 `--override` 在命令行修改 YAML 中的任意字段（点号访问）。当自定义 `train.out_dir` 时，最终 checkpoint 会按照 `gcp_final_<dataset>.pt` 命名。

---
## 6. 训练过程监控（TensorBoard）

### 6.1 启动 TensorBoard
```bash
tensorboard --logdir outputs/tensorboard --port 6006
```
然后在浏览器访问 `http://<server>:6006` 查看曲线与可视化。

### 6.2 关键标量曲线
训练脚本会每 `logging.scalar_interval`（默认 20 iter）记录以下标量：
- `loss/total`
- `loss/silhouette`
- `loss/occ`
- `loss/reg_alpha`
- `loss/reg_scale`
- `metrics/silhouette_iou`
- `metrics/occ_bce`

健康趋势：
- Silhouette/Occ loss 在前期快速下降，后期平滑。
- IoU 稳定上升并保持 >0.65。
- 正则项保持小幅波动，无爆炸。

### 6.3 高斯健康度统计
每 `logging.hist_interval`（默认 100 iter）写入：
- `gaussian/alpha`、`gaussian/scale` 直方图
- `gaussian/alpha_mean`、`gaussian/alpha_min`、`gaussian/alpha_max`
- `gaussian/scale_min`、`gaussian/scale_max`、`gaussian/scale_p95`
- `gaussian/center_movement`

参考阈值：
- `alpha_mean` 维持 0.5–0.8，无大量 α→0 或 α→1。
- `scale_min` 不低于 1e-3，`scale_max` 不超过 0.15。
- `center_movement` 在热身后逐渐 <1e-2。

### 6.4 图像与多视角渲染
每 `logging.image_interval`（默认 200 iter）记录：
- `vis/silhouette_pred`、`vis/silhouette_gt`
- `vis/gaussian_render`
- `vis/overlay`

每 `logging.multiview_interval`（默认 2000 iter）额外渲染验证视角，便于观察模板整体形状是否稳定。

### 6.5 JSON 指标日志
`train/train_gcp.py` 还会在 `train.out_dir/training_metrics.jsonl` 按迭代输出同样的指标，适合离线分析：
```python
import pandas as pd
log = pd.read_json('outputs/checkpoints/pix3d_run1/training_metrics.jsonl', lines=True)
log[['iter', 'silhouette_iou', 'alpha_mean', 'scale_min', 'scale_max']].plot(x='iter')
```

---
## 7. 训练完成后的评估

### 7.1 运行评估脚本
```bash
python evaluation/evaluate_template.py \
  --config configs/pix3d.yaml \
  --checkpoint outputs/checkpoints/pix3d_run1/gcp_final_pix3d.pt \
  --index_file outputs/preprocessed/pix3d/index_val.npy \
  --vis_dir outputs/priors/pix3d_run1/qualitative
```
输出内容：
- `evaluation_report.json`：整体验证指标、各类别评分、阈值判定
- `qualitative/`：预测 vs GT 轮廓叠加图、多视角渲染 PNG
- 终端打印 Chamfer、F-score、Silhouette IoU、Boundary F1、Volume IoU 等汇总

### 7.2 默认成功阈值（可在配置中调整）
- Silhouette mIoU ≥ 0.70，Boundary F1 ≥ 0.60
- Volume IoU ≥ 0.50（128³ voxel）
- Chamfer-L2 ≤ 0.005，F-score@0.02 ≥ 0.60
若多轮训练，请根据 `evaluation_report.json` 选取综合指标最佳的 checkpoint。

### 7.3 常用扩展评估
- 类内/类间 Chamfer 均值方差，用于判断模板泛化能力。
- 先验检索测试：利用 mask 检索最近模板，并评估 silhouette IoU（Top-1 ≥ 0.65、Top-5 ≥ 0.75）。
- Canonical 一致性检查：统计各实例 canonical 变换与模板之间的差异。

---
## 8. 常见问题解答

1. **只需要单类别模板怎么办？**
   - 在配置中将 `model.num_classes` 设置为 1，并在索引构建时筛选目标类别即可。
2. **训练不收敛或损失震荡？**
   - 降低 `optim.lr` 或增大 `train.batch_size`；同时检查 canonical 对齐是否正确。
3. **TensorBoard 图像全黑或 NaN？**
   - 检查渲染参数是否溢出（特别是 scale < 1e-4）；可开启 `logging.debug_dump=true` 以保存异常 batch。
4. **如何启用条件生成器 GCPGenerator？**
   - 在配置中将 `model.mode` 设为 `generator`，并提供 mask encoder 的相关超参（详见 `train/train_gcp.py` 内注释）。

---
## 9. 结果产物

训练完成后，核心产出位于：
- `outputs/checkpoints/<run_name>/gcp_final.pt`：训练后的 GlobalTemplate/GCPGenerator 权重。
- `outputs/priors/<run_name>/cat_XX.npz`：每个类别的高斯模板（包含 `mu`, `scale`, `alpha`）。
- `outputs/tensorboard/<run_name>/`：TensorBoard 日志，便于回溯训练曲线。
- `outputs/priors/<run_name>/qualitative/`：评估阶段的渲染可视化。

以上即 GCP 预训练全流程说明。照此操作即可快速搭建并监控 Gaussian Category Prior 训练任务。
