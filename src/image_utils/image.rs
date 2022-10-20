use image::{imageops::FilterType, DynamicImage, GenericImage, Rgb, RgbImage, Rgba};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_text_mut},
    rect::Rect,
};
use rusttype::{Font, Scale};

use crate::Detection;

use super::ResizeRoi;

/// Resizes image keeping the aspect ratio by padding.
///
/// Returns resized image and [ResizeROI] of the original image, which can be used to reverse resize on inferrence data.
pub fn uniform_resize(
    src: &DynamicImage,
    dst_width: u32,
    dst_height: u32,
) -> (RgbImage, ResizeRoi) {
    let mut dst = RgbImage::from_pixel(dst_width, dst_height, Rgb([0, 0, 0]));

    let src_width = src.width();
    let src_height = src.height();
    let src_ratio = src_width as f32 / src_height as f32;
    let dst_ratio = dst_width as f32 / dst_height as f32;

    let (width, height): (u32, u32) = if src_ratio > dst_ratio {
        let scale = dst_width as f32 / src_width as f32;
        (dst_width, (scale * src_height as f32).floor() as _)
    } else {
        let scale = dst_height as f32 / src_height as f32;
        ((scale * src_width as f32).floor() as _, dst_height)
    };

    let tmp = src.resize(width, height, FilterType::Triangle).to_rgb8();

    let w_pad = (dst_width - width) / 2;
    let h_pad = (dst_height - height) / 2;
    dst.copy_from(&tmp, w_pad, h_pad).unwrap();

    (
        dst,
        ResizeRoi {
            width,
            height,
            w_pad,
            h_pad,
        },
    )
}

pub fn draw_detections(
    img: &mut DynamicImage,
    font: &Font,
    eff_roi: ResizeRoi,
    dets: &[Detection],
    labels: &[&str],
) {
    let width = img.width();
    let height = img.height();

    let mut rand = 0x98413548u32;

    for det in dets {
        let bbox = det.bbox;
        let color = Rgba([
            (rand % 255) as u8,
            ((rand >> 8) % 255) as u8,
            ((rand >> 16) % 255) as u8,
            255,
        ]);

        let rect = Rect::at(
            (bbox.xmin.round() as i32 - eff_roi.w_pad as i32) * width as i32 / eff_roi.width as i32,
            (bbox.ymin.round() as i32 - eff_roi.h_pad as i32) * height as i32
                / eff_roi.height as i32,
        )
        .of_size(
            (bbox.xmax.round() - bbox.xmin.round()) as u32 * width / eff_roi.width,
            (bbox.ymax.round() - bbox.ymin.round()) as u32 * height / eff_roi.height,
        );

        draw_hollow_rect_mut(img, rect, color);

        draw_text_mut(
            img,
            color,
            (bbox.xmin.round() as i32 - eff_roi.w_pad as i32) * width as i32 / eff_roi.width as i32
                + 4,
            (bbox.ymin.round() as i32 - eff_roi.h_pad as i32) * height as i32
                / eff_roi.height as i32
                + 4,
            Scale::uniform(16.0),
            font,
            &format!("{} {:.0}%", labels[det.label], det.score * 100.0),
        );

        // XOR shift PRNG
        rand ^= rand << 13;
        rand ^= rand >> 17;
        rand ^= rand << 5;
    }
}
