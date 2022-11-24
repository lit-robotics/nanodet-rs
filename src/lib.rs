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
pub trait AsFeatureMatrix {
    fn row(&self, row: usize) -> &[f32];
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
