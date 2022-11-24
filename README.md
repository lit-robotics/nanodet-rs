# nanodet-rs

Rust implementation of the [nanodet](https://github.com/RangiLyu/nanodet) decoder.

Supports multiple neural network backends via `AsFeatureMatrix` trait. Implementations are provided for the following backends:
- [ncnn-rs](https://github.com/tpoisonooo/rust-ncnn) - Works on most CPUs and has experimental Vulkan GPU support.
- [openvino-rs](https://github.com/intel/openvino-rs) - A very fast framework that utilises Intel integrated CPU graphics.

## Usage

Can be used as library by including in `Cargo.toml`:
```yaml
nanodet-rs = { git = "https://github.com/lit-robotics/nanodet-rs" }
```

For rust-flavored usage see [examples/ncnn_image.rs](examples/ncnn_image.rs), or if you like opencv check [examples/ncnn_opencv.rs](examples/ncnn_opencv.rs)

## Cargo Features

- `ncnn` - [ncnn-rs](https://github.com/tpoisonooo/rust-ncnn) support.
- `openvino` - [openvino-rs](https://github.com/intel/openvino-rs) support.
- `image` - [image](https://crates.io/crates/image) based utility functions.
- `opencv` - [opencv](https://crates.io/crates/opencv) based utility functions.

## Running examples

### `ncnn`

Run pretrained COCO model on an image with rust image pipeline:
```bash
cargo run --example ncnn_image --release --features ncnn,image -- data/coco_test.jpg
```

Run pretrained COCO model on an image with opencv image pipeline:
```bash
cargo run --example ncnn_opencv --release --features ncnn,opencv -- data/coco_test.jpg
```

### `openvino`

Run pretrained COCO model on an image with opencv image pipeline:
```bash
# Initialize openvino environment (check openvino docs)
source /opt/intel/openvino_2022/setupvars.sh
# Run example
cargo run --example openvino_opencv --release --features openvino,opencv -- data/coco_test.jpg
```
