use opencv::{
    core::{Rect, Size, CV_8UC3},
    imgproc::{resize, INTER_LINEAR},
    prelude::*,
};

use crate::Detection;

use super::ResizeRoi;

/// Resizes image keeping the aspect ratio by padding.
///
/// Returns resized image and [ResizeROI] of the original image, which can be used to reverse resize on inferrence data.
pub fn uniform_resize(src: &Mat, dst_width: i32, dst_height: i32) -> (Mat, ResizeRoi) {
    let dst = Mat::new_rows_cols_with_default(dst_height, dst_width, CV_8UC3, 0.into()).unwrap();

    let src_width = src.cols();
    let src_height = src.rows();
    let src_ratio = src_width as f32 / src_height as f32;
    let dst_ratio = dst_width as f32 / dst_height as f32;

    let (width, height): (i32, i32) = if src_ratio > dst_ratio {
        let scale = dst_width as f32 / src_width as f32;
        (dst_width, (scale * src_height as f32).floor() as _)
    } else {
        let scale = dst_height as f32 / src_height as f32;
        ((scale * src_width as f32).floor() as _, dst_height)
    };

    let mut tmp = Mat::default();
    resize(
        src,
        &mut tmp,
        Size::new(width, height),
        0.0,
        0.0,
        INTER_LINEAR,
    )
    .unwrap();

    let w_pad = (dst_width - width) / 2;
    let h_pad = (dst_height - height) / 2;
    let mut dst_roi = Mat::roi(&dst, Rect::new(w_pad, h_pad, width, height)).unwrap();
    tmp.copy_to(&mut dst_roi).unwrap();

    (
        dst,
        ResizeRoi {
            width: width as u32,
            height: height as u32,
            w_pad: w_pad as u32,
            h_pad: h_pad as u32,
        },
    )
}

pub fn draw_detections(img: &mut Mat, eff_roi: ResizeRoi, dets: &[Detection], labels: &[&str]) {
    let width = img.cols();
    let height = img.rows();

    let mut rand = 0x98413548u32;

    for det in dets {
        let bbox = det.bbox;
        let color = (
            (rand % 255) as f64,
            ((rand >> 8) % 255) as f64,
            ((rand >> 16) % 255) as f64,
        );

        let p1 = opencv::core::Point::new(
            (bbox.xmin.round() as i32 - eff_roi.w_pad as i32) * width / eff_roi.width as i32,
            (bbox.ymin.round() as i32 - eff_roi.h_pad as i32) * height / eff_roi.height as i32,
        );
        let p2 = opencv::core::Point::new(
            (bbox.xmax.round() as i32 - eff_roi.w_pad as i32) * width / eff_roi.width as i32,
            (bbox.ymax.round() as i32 - eff_roi.h_pad as i32) * height / eff_roi.height as i32,
        );
        let rect = opencv::core::Rect::from_points(p1, p2);

        opencv::imgproc::rectangle(img, rect, color.into(), 1, opencv::imgproc::LINE_8, 0).unwrap();

        let p_text = opencv::core::Point::new(
            (bbox.xmin.round() as i32 - eff_roi.w_pad as i32) * width / eff_roi.width as i32,
            (bbox.ymin.round() as i32 - eff_roi.h_pad as i32) * height / eff_roi.height as i32 + 10,
        );

        opencv::imgproc::put_text(
            img,
            &format!("{} {:.0}%", labels[det.label], det.score * 100.0),
            p_text,
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.4,
            color.into(),
            1,
            opencv::imgproc::LINE_8,
            false,
        )
        .unwrap();

        // XOR shift PRNG
        rand ^= rand << 13;
        rand ^= rand >> 17;
        rand ^= rand << 5;
    }
}
