#[cfg(feature = "opencv")]
pub mod opencv;

#[cfg(feature = "image")]
pub mod image;

/// Information about uniform image resize
#[derive(Debug, Clone, Copy)]
pub struct ResizeRoi {
    /// Resized image width (without padding)
    pub width: u32,
    /// Resized image height (without padding)
    pub height: u32,
    /// Horizontal padding on each side (total padding *2)
    pub w_pad: u32,
    /// Vertical padding on each side (total padding *2)
    pub h_pad: u32,
}
