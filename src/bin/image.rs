use ncnn_rs::{MatPixelType, Net};
use opencv::core::{Mat as CvMat, Rect, Size, CV_8UC3};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{resize, INTER_LINEAR};
use opencv::prelude::{MatTraitConst, MatTraitConstManual};

const CLASS_NAMES: &[&'static str] = &[
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

fn main() -> anyhow::Result<()> {
    let nanodet = Nanodet {
        reg_max: 7,
        input_size: (416, 416),
        num_class: 80,
        strides: vec![8, 16, 32, 64],
    };

    let mut img = imread("data/coco_test.jpg", IMREAD_COLOR)?;
    let (resized_img, effective_area) =
        uniform_resize(&img, nanodet.input_size.0, nanodet.input_size.1);
    // imshow("image", &img).unwrap();
    // wait_key(5000).unwrap();

    let mut net = Net::new();
    net.load_param("data/nanodet-plus-m_416.param")?;
    net.load_model("data/nanodet-plus-m_416.bin")?;

    let alloc = ncnn_rs::Allocator::new();
    let mut input = ncnn_rs::Mat::from_pixels(
        resized_img.data_bytes().unwrap(),
        MatPixelType::BGR,
        416,
        416,
        &alloc,
    )?;
    let mut output = ncnn_rs::Mat::new();

    let mean_vals = [183.53, 116.28, 123.675];
    let norm_vals = [0.017429, 0.017507, 0.017125];
    input.substract_mean_normalize(&mean_vals, &norm_vals);

    let mut extractor = net.create_extractor();
    extractor.input("data", &input)?;
    extractor.extract("output", &mut output)?;

    let score_threshold = 0.4;
    let nms_threshold = 0.5;

    let center_priors = nanodet.generate_grid_center_priors();
    let mut results = nanodet.decode_infer(&output, &center_priors, score_threshold);

    // println!("{:#?}", results);

    let mut detections = Vec::new();
    for class in results.iter_mut() {
        nms_filter(class, nms_threshold);
        detections.append(class);
    }

    draw_detections(&mut img, effective_area, &detections);

    imshow("image", &img).unwrap();
    wait_key(0).unwrap();

    // println!("{:#?}", detections);

    Ok(())
}

fn draw_detections(img: &mut CvMat, eff_roi: ROI, dets: &[Detection]) {
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
            (bbox.xmin.round() as i32 - eff_roi.x) * width / eff_roi.width,
            (bbox.ymin.round() as i32 - eff_roi.y) * height / eff_roi.height,
        );
        let p2 = opencv::core::Point::new(
            (bbox.xmax.round() as i32 - eff_roi.x) * width / eff_roi.width,
            (bbox.ymax.round() as i32 - eff_roi.y) * height / eff_roi.height,
        );
        let rect = opencv::core::Rect::from_points(p1, p2);

        opencv::imgproc::rectangle(img, rect, color.into(), 1, opencv::imgproc::LINE_8, 0).unwrap();

        opencv::imgproc::put_text(
            img,
            &format!("{} {:.0}%", CLASS_NAMES[det.label], det.score * 100.0),
            p1,
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

#[derive(Debug, Clone, Copy)]
struct ROI {
    width: i32,
    height: i32,
    x: i32,
    y: i32,
}

/// Resizes image keeping the aspect ratio by padding
fn uniform_resize(src: &CvMat, dst_width: i32, dst_height: i32) -> (CvMat, ROI) {
    let dst = CvMat::new_rows_cols_with_default(dst_height, dst_width, CV_8UC3, 0.into()).unwrap();

    let src_width = src.cols();
    let src_height = src.rows();
    let src_ratio = src_width as f32 / src_height as f32;
    let dst_ratio = dst_width as f32 / dst_height as f32;

    let (tmp_width, tmp_height): (i32, i32) = if src_ratio > dst_ratio {
        let scale = dst_width as f32 / src_width as f32;
        (dst_width, (scale * src_height as f32).floor() as _)
    } else {
        let scale = dst_height as f32 / src_height as f32;
        ((scale * src_width as f32).floor() as _, dst_height)
    };

    let mut tmp = CvMat::default();
    resize(
        src,
        &mut tmp,
        Size::new(tmp_width, tmp_height),
        0.0,
        0.0,
        INTER_LINEAR,
    )
    .unwrap();

    let w_pad = (dst_width - tmp_width) / 2;
    let h_pad = (dst_height - tmp_height) / 2;
    let mut dst_roi = CvMat::roi(&dst, Rect::new(w_pad, h_pad, tmp_width, tmp_height)).unwrap();
    tmp.copy_to(&mut dst_roi).unwrap();

    (
        dst,
        ROI {
            width: tmp_width,
            height: tmp_height,
            x: w_pad,
            y: h_pad,
        },
    )
}

struct CenterPrior {
    x: i32,
    y: i32,
    stride: i32,
}

fn div_ceil(lhs: i32, rhs: i32) -> i32 {
    let d = lhs / rhs;
    let r = lhs % rhs;
    if (r > 0 && rhs > 0) || (r < 0 && rhs < 0) {
        d + 1
    } else {
        d
    }
}

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
    xmin: f32,
    ymin: f32,
    xmax: f32,
    ymax: f32,
}

impl BoundingBox {
    fn intersection(&self, other: &Self) -> Option<Self> {
        let inter = Self {
            xmin: self.xmin.max(other.xmin),
            ymin: self.ymin.max(other.ymin),
            xmax: self.xmax.min(other.xmax),
            ymax: self.ymax.min(other.ymax),
        };

        if (inter.xmax - inter.xmin) <= 1.0 || (inter.ymax - inter.ymin) <= 1.0 {
            None
        } else {
            Some(inter)
        }
    }

    fn area(&self) -> f32 {
        (self.xmax - self.xmin + 1.0).max(0.0) * (self.ymax - self.ymin + 1.0).max(0.0)
    }
}

#[derive(Debug, Clone)]
struct Detection {
    bbox: BoundingBox,
    score: f32,
    label: usize,
}

fn fast_exp(x: f32) -> f32 {
    let i: u32 = ((1 << 23) as f64 * (1.4426950409 * x as f64 + 126.93490512)) as u32;
    f32::from_bits(i)
}

fn activation_softmax(src: &[f32]) -> Vec<f32> {
    let alpha = src.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    let mut denominator = 0.0;

    let mut out: Vec<f32> = src
        .iter()
        .map(|item| {
            let val = fast_exp(item - alpha);
            denominator += val;
            val
        })
        .collect();

    for item in &mut out {
        *item /= denominator;
    }

    out
}

struct Nanodet {
    reg_max: i32,
    /// (width, height)
    input_size: (i32, i32),
    num_class: usize,
    strides: Vec<i32>,
}

impl Nanodet {
    fn dis_pred_to_bbox(&self, dfl_det: &[f32], x: i32, y: i32, stride: i32) -> BoundingBox {
        let ct_x = (x * stride) as f32;
        let ct_y = (y * stride) as f32;

        let dis_pred: Vec<f32> = (0..4)
            .into_iter()
            .map(|i| {
                let start = (i * (self.reg_max + 1)) as usize;
                let end = ((i + 1) * (self.reg_max + 1)) as usize;

                let dis = activation_softmax(&dfl_det[start..end])
                    .iter()
                    .enumerate()
                    .fold(0.0, |dis, (j, dis_after_sm)| dis + j as f32 * dis_after_sm);

                dis * stride as f32
            })
            .collect();

        BoundingBox {
            xmin: (ct_x - dis_pred[0]).max(0.0),
            ymin: (ct_y - dis_pred[1]).max(0.0),
            xmax: (ct_x + dis_pred[2]).min(self.input_size.0 as _),
            ymax: (ct_y + dis_pred[3]).min(self.input_size.1 as _),
        }
    }

    fn decode_infer(
        &self,
        features: &ncnn_rs::Mat,
        center_priors: &[CenterPrior],
        threshold: f32,
    ) -> Vec<Vec<Detection>> {
        let mut results = (0..self.num_class)
            .into_iter()
            .map(|_| Vec::new())
            .collect::<Vec<_>>();

        for (idx, ct) in center_priors.iter().enumerate() {
            let scores = ncnn_get_row(features, idx);

            let (label, score) = scores
                .iter()
                .take(self.num_class)
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap();

            if *score >= threshold {
                let bbox = self.dis_pred_to_bbox(&scores[self.num_class..], ct.x, ct.y, ct.stride);
                results[label].push(Detection {
                    bbox,
                    score: *score,
                    label,
                })
            }
        }

        results
    }

    fn generate_grid_center_priors(&self) -> Vec<CenterPrior> {
        let mut cts = Vec::new();
        for stride in self.strides.iter().copied() {
            let feat_w = div_ceil(self.input_size.0, stride);
            let feat_h = div_ceil(self.input_size.1, stride);
            for y in 0..feat_h {
                for x in 0..feat_w {
                    cts.push(CenterPrior { x, y, stride })
                }
            }
        }
        cts
    }
}

fn ncnn_get_row(mat: &ncnn_rs::Mat, y: usize) -> &[f32] {
    let row_size = mat.get_w() as usize * mat.get_elemsize() as usize;
    unsafe {
        let ptr = mat.get_data() as *const u8;
        core::slice::from_raw_parts(
            ptr.offset(row_size as isize * y as isize) as *const f32,
            row_size,
        )
    }
}

fn nms_filter(detections: &mut Vec<Detection>, threshold: f32) {
    detections.sort_by(|a, b| a.score.total_cmp(&b.score));

    let mut i = 0;
    while let Some(a) = detections.get(i).cloned() {
        let mut j = i + 1;
        while let Some(b) = detections.get(j) {
            if let Some(inter) = a.bbox.intersection(&b.bbox) {
                let inter_area = inter.area();
                let overlap = inter_area / (a.bbox.area() + b.bbox.area() - inter_area);
                if overlap >= threshold {
                    detections.remove(j);
                    continue;
                }
            }

            j += 1;
        }

        i += 1;
    }
}
