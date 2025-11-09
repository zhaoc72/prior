# Gaussian Category Prior 使用全指南

> 本文档用中文全面说明 Gaussian Category Prior（GCP）项目从环境搭建、数据准备、预处理、训练、评估到常见问题的全部流程。
> 按照本文执行即可在 Pix3D、ScanNetV2、KITTI、vKITTI2 等数据集上复现类别级高斯模板先验。

---
## 1. 快速上手总览

1. **准备环境**：根据 GPU/CUDA 版本创建 Conda 虚拟环境并安装依赖。
2. **配置数据路径**：在对应的 `configs/*.yaml` 文件写入原始数据的绝对路径。
3. **运行预处理**：执行 `bash scripts/preprocess_<dataset>.sh <config>`，生成 canonical mesh、occupancy 点云、相机标注与索引。
4. **启动训练**：执行 `bash scripts/train_<dataset>.sh <config>` 或直接运行 `python train/train_gcp.py --config <config>`。
5. **导出模板并评估**：训练脚本会自动调用 `train/export_priors.py`；也可手动运行评估脚本获得定量指标与可视化。
6. **监控与排错**：使用 TensorBoard、日志或文档中的故障排查章节定位问题。

---
## 2. 硬件与软件要求

### 2.1 推荐硬件
- **CPU**：≥32 物理核（示例环境：2×Intel Xeon 8336C，共 64 核 128 线程）。
- **内存**：≥128 GB（建议 256 GB，可同时运行多个预处理进程）。
- **GPU**：训练阶段建议使用 NVIDIA RTX/A 系列显卡，显存 ≥24 GB；预处理阶段可仅使用 CPU。
- **存储**：SSD/NVMe，确保有 ≥500 GB 可用空间保存预处理结果与 checkpoint。

### 2.2 软件环境
- 操作系统：Ubuntu 20.04+ / CentOS 7+
- Python：3.10
- CUDA：11.8 或 12.1（与 `env/environment.yml` 中的 PyTorch 版本一致）
- Conda：Miniconda/Anaconda

如需自定义 CUDA 版本，请在创建环境前编辑 `env/environment.yml` 内的 `pytorch` 与 `pytorch-cuda` 包版本。

---
## 3. 创建与验证 Conda 环境

```bash
# 1. 创建环境
git clone <repo-url>
cd prior
conda env create -f env/environment.yml

# 2. 激活
conda activate gcp-prior

# 3. 验证关键依赖
python - <<'PY'
import torch, torch.cuda
import pytorch3d
import trimesh
print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())
print('PyTorch3D', pytorch3d.__version__)
print('Trimesh OK')
PY
```
若 `torch.cuda.is_available()` 返回 `False`，请检查 CUDA 驱动及显卡环境。

---
## 4. 代码目录结构

```
prior/
├─ configs/            # YAML 配置（基础配置、数据集专属配置）
├─ datasets/           # Dataset 封装与数据管道
├─ evaluation/         # 评估脚本与指标
├─ preprocess/         # 预处理脚本（canonical 化、采样、初始化）
├─ scripts/            # bash 脚本，封装常用流程
├─ train/              # 训练、导出、工具脚本
├─ utils/              # 几何、渲染与通用工具函数
├─ env/                # 环境说明文件
└─ outputs/            # 默认输出目录（预处理结果/日志/模型/先验）
```

---
## 5. 配置文件说明

配置体系采用 YAML 文件，可通过 `--config` 指定单一文件，并允许使用 `--override key=value` 在命令行修改任意字段。

常见字段：
- `paths.dataset_root`：数据集根目录（绝对路径）。
- `paths.raw_mesh_dir`、`paths.mask_dir`、`paths.annotations_json`：相对或绝对路径，指向官方发布的 mesh、mask、注释。
- `data.index_file`：预处理后索引文件的输出路径（用于训练）。
- `model.*`：高斯模板参数（数量、初始尺度、类别数）。
- `train.*`：批大小、迭代次数、输出目录、随机种子。
- `monitoring.*`：TensorBoard、JSON 日志、图像渲染频率等。

### 5.1 数据路径示例（Pix3D）
```yaml
paths:
  dataset_root: /media/pc/D/datasets/pix3d
  raw_mesh_dir: model
  mask_dir: mask
  annotations_json: pix3d.json

data:
  index_file: outputs/preprocessed/pix3d/index_train.npy
```
若填写相对路径，脚本会将其拼接到 `dataset_root` 下；如需跨盘路径，请直接写成绝对路径。

---
## 6. 数据准备

### 6.1 原始数据组织建议
| 数据集 | 必须包含的子目录/文件 | 备注 |
|--------|-----------------------|------|
| Pix3D  | `model/`, `mask/`, `pix3d.json` | 官方解压结构，包含 OBJ/PLY 网格与掩码 |
| ScanNetV2 | `scans/`, `scannet_frames_25k/`, `scene0000_00.txt` 等 | 需下载官方全集或至少包含训练/验证场景 |
| KITTI  | `training/`（含 `calib/`, `image_2/`, `label_2/`） | 默认处理 3D 检测训练集 |
| vKITTI2| `SceneXX/`、`vkitti_1.3.1_extrinsicsgt.txt` 等 | 保持官方目录层级 |

确保磁盘对所有路径拥有读写权限，预处理输出默认写入仓库内 `outputs/preprocessed/<dataset>/`。

### 6.2 校验文件完整性
建议在下载后使用 `md5sum`/`sha1sum` 或官方校验脚本确认数据未损坏；部分脚本会在读取时给出数据缺失提示，可按日志补齐。

---
## 7. 预处理流程

### 7.1 总体流程
以 Pix3D 为例：
```bash
bash scripts/preprocess_pix3d.sh configs/pix3d.yaml
```
脚本顺序执行以下步骤，所有中间结果保存在 `outputs/preprocessed/pix3d/`：
1. **canonicalize_meshes.py**：将原始 mesh 平移至原点、归一化尺度、对齐朝向；输出 canonical mesh 与变换矩阵。
2. **sample_points_occ.py**：对 canonical mesh 进行表面/空间点采样，生成 occupancy/TSDF 监督点 `.npz`。
3. **prepare_pix3d_metadata.py**：转换官方注释为摄像机参数 JSON。
4. **build_index.py**：整合 mask、occupancy、摄像机和类别信息，生成训练/验证所需的索引文件。

其他数据集使用相应脚本：
```bash
bash scripts/preprocess_scannetv2.sh configs/scannetv2.yaml
bash scripts/preprocess_kitti.sh configs/kitti.yaml
bash scripts/preprocess_vkitti2.sh configs/vkitti2.yaml
```

### 7.2 手动执行与参数说明
#### canonicalize_meshes.py
```bash
python preprocess/canonicalize_meshes.py \
  --src <raw_mesh_dir> \
  --dst <canonical_mesh_dir> \
  --workers 64
```
- `--workers`：并行进程数，默认 `os.cpu_count()`；设为 1 时串行执行。
- 自动写出 `transforms.json` 保存平移/旋转/缩放矩阵。

#### sample_points_occ.py
```bash
python preprocess/sample_points_occ.py \
  --mesh_dir <canonical_mesh_dir> \
  --out <occ_output_dir> \
  --n_surf 40000 \
  --n_uniform 60000 \
  --workers 64 \
  --executor auto \
  --skip_bad_mesh
```
- `--executor`：`auto`（默认，优先进程池，失败后自动降档至线程池）、`process`、`thread` 可选。
- `--workers`：并发任务数；在 `auto`/`process` 模式下表示进程数，在 `thread` 模式下表示线程数。
- `--no_skip_existing`：重新生成已存在的 `.npz` 文件。
- `--skip_bad_mesh`：遇到损坏网格时跳过并记录到 `failed.txt`。
- 默认在未设置 `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 时强制置为 1，避免每个进程再创建额外线程。
- 遇到 `BrokenProcessPool` 会自动减半进程数重试，最终切换线程池；终端会打印详细日志。

#### init_gaussians.py
```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python preprocess/init_gaussians.py \
  --points_dir <occ_output_dir> \
  --out <gaussian_output_dir> \
  --workers 16 \
  --nn_jobs 8
```
- `--workers`：进程数；`--nn_jobs` 传递给 `sklearn.NearestNeighbors`，决定单个进程内部的线程数。
- 支持断点续跑：若输出文件存在会自动跳过，可加 `--overwrite` 强制重算。

#### build_index.py
```bash
python preprocess/build_index.py \
  --mask_dir <mask_dir> \
  --occ_dir <occ_output_dir> \
  --cams <cameras.json> \
  --out <index_file> \
  --split train \
  --workers 16
```
- `--split` 支持 `train/val/test`，用于过滤注释。
- 索引文件为 `.npy`，内容包含样本路径、类别、camera intrinsics/pose、canonical 变换等。

### 7.3 一键脚本的环境变量
所有预处理脚本都可以通过环境变量覆盖默认并发设置：
- 通用变量：`PREPROCESS_WORKERS`（全局默认）、`PREPROCESS_CANON_WORKERS`、`PREPROCESS_OCC_WORKERS`、`PREPROCESS_INDEX_WORKERS`。
- 数据集专属变量（以 Pix3D 为例）：
  - `PIX3D_CANON_WORKERS`
  - `PIX3D_OCC_WORKERS`
  - `PIX3D_OCC_EXECUTOR`（`auto`/`process`/`thread`）
  - `PIX3D_INDEX_WORKERS`

示例：
```bash
export PREPROCESS_WORKERS=64
export PIX3D_OCC_EXECUTOR=thread
bash scripts/preprocess_pix3d.sh configs/pix3d.yaml
```
环境变量优先级：数据集专属 > 通用 > Python 默认。

### 7.4 监控与排障
- 使用 `htop`/`pidstat` 观察 CPU 利用率；若进程频繁崩溃，可降低 `--workers` 或切换 `--executor thread`。
- `RuntimeWarning: divide by zero` 常见于退化三角面片，通常不会导致流程中断；可在 `preprocess/canonicalize_meshes.py` 中添加自定义滤波策略。
- 若磁盘 I/O 成为瓶颈，可将中间输出目录放置在 NVMe 上或调整 `--workers`。

---
## 8. 训练与导出

### 8.1 使用脚本
```bash
bash scripts/train_pix3d.sh configs/pix3d.yaml
bash scripts/train_scannetv2.sh configs/scannetv2.yaml
bash scripts/train_kitti.sh configs/kitti.yaml
bash scripts/train_vkitti2.sh configs/vkitti2.yaml
```
脚本流程：
1. 调用 `python train/train_gcp.py --config <config>` 启动训练。
2. 训练完成后运行 `python train/export_priors.py` 将 checkpoint 转为每类高斯模板。
3. TensorBoard 日志写入 `outputs/tensorboard/<dataset>/`。

### 8.2 手动命令示例
```bash
python train/train_gcp.py \
  --config configs/pix3d.yaml \
  --override train.out_dir=outputs/checkpoints/pix3d_run1 \
  --override train.max_iter=30000 \
  --override train.batch_size=16

python train/export_priors.py \
  --ckpt outputs/checkpoints/pix3d_run1/gcp_final_pix3d.pt \
  --config configs/pix3d.yaml \
  --out outputs/priors/pix3d_run1
```
- `--resume` 可指定历史 checkpoint 继续训练。
- `train.device` 默认为自动选择 `cuda`，若无 GPU 会回退至 CPU（速度较慢）。
- 支持通过 `--override monitoring.tensorboard.log_dir=...` 等参数修改日志目录。

### 8.3 多机/多卡提示
目前脚本基于单机单卡设计。如需多卡训练，请自行在 `train/train_gcp.py` 中集成 `DistributedDataParallel`，并相应修改配置与启动脚本。

---
## 9. 训练过程监控

### 9.1 启动 TensorBoard
```bash
tensorboard --logdir outputs/tensorboard --port 6006
```
浏览器访问 `http://<server>:6006` 查看曲线与渲染结果。

### 9.2 关键监控项
- **标量**：`loss/total`、`loss/silhouette`、`loss/occ`、`metrics/silhouette_iou`、`metrics/occ_bce` 等。
- **直方图**：`gaussian/alpha`、`gaussian/scale`，用于监控模板权重与尺度。
- **图像**：预测/真值轮廓、渲染效果、遮罩叠加。
- **JSONL 日志**：`train.out_dir/training_metrics.jsonl` 可使用 pandas 读取分析。

健康指标参考：
- IoU 稳定上升并保持在 0.65 以上。
- `alpha_mean` 维持 0.5–0.8，`scale_min` 不低于 1e-3。
- 若出现梯度爆炸，可降低学习率或增大正则权重。

---
## 10. 评估与质检

```bash
python evaluation/evaluate_template.py \
  --config configs/pix3d.yaml \
  --checkpoint outputs/checkpoints/pix3d_run1/gcp_final_pix3d.pt \
  --index_file outputs/preprocessed/pix3d/index_val.npy \
  --vis_dir outputs/priors/pix3d_run1/qualitative
```
输出：
- `evaluation_report.json`：各类指标（Chamfer、F-score、Silhouette IoU 等）。
- `qualitative/`：渲染可视化。

推荐阈值（可在配置中调节）：
- Silhouette mIoU ≥ 0.70
- Boundary F1 ≥ 0.60
- Chamfer-L2 ≤ 0.005

---
## 11. 产出目录约定

```
outputs/
├─ preprocessed/<dataset>/
│  ├─ meshes/            # canonical mesh
│  ├─ occ_npz/           # occupancy 点云
│  ├─ cameras.json       # 采样后的相机参数
│  └─ index_*.npy        # 训练/验证索引
├─ checkpoints/<run_name>/
│  ├─ gcp_final_<dataset>.pt
│  └─ training_metrics.jsonl
├─ priors/<run_name>/
│  ├─ cat_XX.npz         # 每类高斯模板
│  └─ qualitative/       # 评估渲染
└─ tensorboard/<run_name>/
```
可根据项目需求修改 `train.out_dir`、`monitoring.tensorboard.log_dir` 等字段实现自定义布局。

---
## 12. 常见问题排查

1. **`BrokenProcessPool` 反复出现**
   - 降低 `--workers` 或设置 `--executor thread`。
   - 手动导出 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免进程内再开线程。
   - 检查系统日志是否存在 OOM。

2. **`RuntimeWarning: divide by zero`**
   - 出现在 mesh 含有退化三角形时，通常不会中断流程；若影响采样，可在 canonical 阶段先做清理或使用 `--skip_bad_mesh`。

3. **训练损失震荡或 NaN**
   - 检查预处理结果是否缺失。
   - 降低学习率、增大 `train.gradient_clip` 或开启 AMP（若适配）。

4. **显存不足**
   - 减小 `train.batch_size`，或在配置中降低高斯数量 `model.K`。

5. **如何重跑单个步骤？**
   - 删除对应输出文件后重新运行脚本；`sample_points_occ.py` 默认跳过已存在文件，可使用 `--no_skip_existing` 强制覆盖。

6. **如何扩展到新数据集？**
   - 准备原始 mesh 与 mask。
   - 参考 `preprocess/` 下脚本编写新的元数据转换器与索引构建逻辑。
   - 复制一份配置文件，填写路径与类别映射。

---
## 13. FAQ

- **是否必须使用 GPU？**
  - 训练阶段需要 GPU 才能高效运行；预处理可在 CPU 上完成。
- **支持断点续跑吗？**
  - 预处理脚本默认跳过已生成文件；训练脚本可使用 `--resume` 指定 checkpoint。
- **如何自定义日志级别？**
  - 设置环境变量 `LOGLEVEL=INFO/DEBUG`；脚本会统一读取。
- **是否提供 Docker？**
  - 当前仅提供 Conda 环境；如需 Docker，可基于 `env/environment.yml` 自行构建。

---
## 14. 联系方式
如在复现过程中遇到未覆盖的问题，请整理命令、日志与环境信息提交 Issue 或联系维护者。建议附上：
- 配置文件片段
- 系统/驱动版本
- 完整的终端输出与栈追踪

祝顺利构建 Gaussian Category Prior！
