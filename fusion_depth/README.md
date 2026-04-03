# Fusion Depth Training

- `VGGT` 提供多帧聚合特征
- `DA3 Metric Large` 提供逐帧单目特征和单目深度先验
- `fusion head` 融合两路 feature，输出所有帧的 metric depth

## 1. 数据格式

### 非obs数据 - data.py

1. 序列形式

```json
{
  "sequence_id": "drive_0001",
  "frames": [
    {"image": "/abs/path/000000.png", "depth": "/abs/path/000000.npy"},
    {"image": "/abs/path/000001.png", "depth": "/abs/path/000001.npy"}
  ]
}
```

数据集会自动按 `sequence_length` 和 `sequence_stride` 做滑窗。

2. 直接样本形式

```json
{
  "sample_id": "sample_0001",
  "images": ["/abs/path/0.png", "/abs/path/1.png", "/abs/path/2.png", "/abs/path/3.png", "/abs/path/4.png", "/abs/path/5.png"],
  "depths": ["/abs/path/0.npy", "/abs/path/1.npy", "/abs/path/2.npy", "/abs/path/3.npy", "/abs/path/4.npy", "/abs/path/5.npy"]
}
```

支持的 depth 文件格式：

- `.npy`
- `.npz`，默认取 key=`depth`
- 常见图片格式，如 `.png` / `.tif`

如果 PNG 深度以毫米为单位，可将配置中的 `depth_scale` 设为 `0.001`。

### obs数据 - obs_data.py

先筛目标相机，再聚合scene后筛连续帧

## 2. 训练

单卡：

```bash
python train_fusion_depth.py --config configs/fusion_depth_example.yaml
```

多卡：

```bash
torchrun --nproc_per_node=4 train_fusion_depth.py --config configs/fusion_depth_example.yaml
```

## 3. 支持

- 多 GPU DDP
- AMP 混合精度
- 定期保存 checkpoint
- 定期在验证集上评估 `abs_rel / sq_rel / rmse / rmse_log / delta1/2/3`
- 定期保存可视化图

可视化图从左到右依次为：

`RGB | GT | Fusion Pred | DA3 Pred | VGGT Pred`

## 4. 注意

- 当前训练不使用相机参数、ray head 或其他几何分支
- 输入图像会统一 resize 到配置中的 `image_size`
- 输出监督是逐帧 metric depth
