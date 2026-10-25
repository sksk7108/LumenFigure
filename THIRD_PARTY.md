# 第三方代码与模型

本仓库为便于分发，只拷贝了推理实际用到的上游源码，并保留原许可证全文。

## Robust Video Matting

- 来源：https://github.com/PeterL1n/RobustVideoMatting
- 许可证：GPL-3.0（见 `third_party/rvm/LICENSE`）
- 拷贝范围：`model/` 中 MobileNetV3 推理所需文件
- 本地修改：`third_party/rvm/model/model.py` 去掉了 ResNet50 分支，仅保留 MobileNetV3
- 权重：`rvm_mobilenetv3.pth`  
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth

因拷贝了 GPL-3.0 代码，**LumenFigure 整体以 GPL-3.0 发布**。

## Video Depth Anything

- 来源：https://github.com/DepthAnything/Video-Depth-Anything
- 代码许可证：Apache-2.0（见 `third_party/video_depth_anything/LICENSE`）
- 拷贝范围：流式推理所需的 `video_depth_anything/`（`video_depth_stream.py`、`dpt.py`、`dpt_temporal.py`、`dinov2.py`、`dinov2_layers/`、`motion_module/`、`util/`）
- 本地修改：去掉 `video_depth_stream.py` 里未使用的 `utils.util` 导入。LumenFigure 自己维护滑动窗口，并把深度留在 GPU 的网络分辨率上
- 未拷贝：离线 32 帧脚本 `video_depth.py`、`run.py`、benchmark、loss、Gradio
- 默认权重：Video-Depth-Anything-Small（`video_depth_anything_vits.pth`），Apache-2.0  
  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth

**不要把 Base / Large 权重当作默认项**：那些权重是 CC-BY-NC-4.0，且超出 GTX 1660 Ti 6GB。

1660 Ti 上使用官方 **streaming** 路径（逐帧 + 时间缓存），而不是一次送入 32 帧的离线路径。离线 Small 在 518 分辨率下大约需要 7GB 显存。
