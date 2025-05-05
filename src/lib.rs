mod bbox;
mod decoder;
pub mod image_utils;
mod util;

pub use bbox::*;
pub use decoder::*;
pub use util::*;

/// Trait for reading Nanodet feature matrix (NN output tensor).
///
/// Implement this for your neural network library or use one of the implementations
/// provided by this crate by enabling appropriate feature:
/// - `ncnn`: Implementation for [ncnn_rs::Mat](https://rust-ncnn.github.io/ncnn_rs/struct.Mat.html)
/// - `openvino`: Implementation for [openvino::Blob](https://docs.rs/openvino/latest/openvino/struct.Blob.html)
pub trait AsFeatureMatrix {
    fn row(&self, row: usize) -> &[f32];
}

impl AsFeatureMatrix for Box<dyn AsFeatureMatrix> {
    fn row(&self, row: usize) -> &[f32] {
        (**self).row(row)
    }
}

#[cfg(feature = "ncnn")]
impl AsFeatureMatrix for ncnn_rs::Mat {
    fn row(&self, row: usize) -> &[f32] {
        assert!(row < self.h() as usize, "Row out of range");

        let row_size = self.w() as usize * self.elemsize() as usize;
        unsafe {
            let ptr = self.data() as *const u8;
            core::slice::from_raw_parts(
                ptr.offset(row_size as isize * row as isize) as *const f32,
                row_size,
            )
        }
    }
}

#[cfg(feature = "openvino")]
impl AsFeatureMatrix for openvino::Tensor {
    fn row(&self, row: usize) -> &[f32] {
        let shape = self.get_shape().unwrap();
        let rank = shape.get_rank();
        assert_eq!(rank, 3, "Wrong tensor rank");
        let dims: [_; 3] = shape.get_dimensions().try_into().unwrap();
        let [c, h, w] = dims.map(|s| s as usize);

        assert!(row < h, "Row out of range");
        assert_eq!(c, 1, "Wrong channel count");

        let buffer = self.get_data::<f32>().unwrap();
        let offset = w * row;
        &buffer[offset..offset + w]
    }
}
