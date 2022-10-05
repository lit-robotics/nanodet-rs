# ncnn-nanodet-rs

Rust crate for the [nanodet](https://github.com/RangiLyu/nanodet) object detection model, based on [ncnn](https://github.com/Tencent/ncnn) neural network framework.

## Usage

Can be used as library by including in `Cargo.toml`:
```yaml
ncnn-nanodet-rs = { git = "https://github.com/chemicstry/ncnn-nanodet-rs" }
```

For usage example see [src/bin/image.rs](src/bin/image.rs)

## Running examples

Run pretrained COCO model on an image:
```bash
cargo run --bin image --release -- data/coco_test.jpg
```
