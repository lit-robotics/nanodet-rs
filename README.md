# nanodet-rs

Rust implementation of the [nanodet](https://github.com/RangiLyu/nanodet) decoder.

Supports multiple neural network backends via `AsFeatureMatrix` trait. Implementations are provided for the following backends:
- [ncnn-rs](https://github.com/tpoisonooo/rust-ncnn)

## Usage

Can be used as library by including in `Cargo.toml`:
```yaml
nanodet-rs = { git = "https://github.com/lit-robotics/nanodet-rs" }
```

For rust-flavored usage see [examples/ncnn_image.rs](examples/ncnn_image.rs), or if you like opencv check [examples/ncnn_opencv.rs](examples/ncnn_opencv.rs)

## Cargo Features

- `ncnn` - [ncnn-rs](https://github.com/tpoisonooo/rust-ncnn) support.
- `image` - [image](https://crates.io/crates/image) based utility functions.
- `opencv` - [opencv](https://crates.io/crates/opencv) based utility functions.

## Running examples

Examples are based on `ncnn` backend, which supports the most platforms.

Run pretrained COCO model on an image with rust image pipeline:
```bash
cargo run --example ncnn_image --release --features ncnn,image -- data/coco_test.jpg
```

Run pretrained COCO model on an image with opencv image pipeline:
```bash
cargo run --example ncnn_opencv --release --features ncnn,opencv -- data/coco_test.jpg
```
