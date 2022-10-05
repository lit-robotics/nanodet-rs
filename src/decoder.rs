use crate::{activation_softmax, div_ceil, ncnn_get_f32_row, BoundingBox};

#[derive(Debug, Clone)]
pub struct CenterPrior {
    pub x: i32,
    pub y: i32,
    pub stride: i32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BoundingBox,
    pub score: f32,
    pub label: usize,
}

#[derive(Debug, Clone)]
pub struct NanodetDecoder {
    reg_max: i32,
    /// (width, height)
    input_size: (i32, i32),
    num_classes: usize,
    strides: Vec<i32>,
    center_priors: Vec<CenterPrior>,
}

impl NanodetDecoder {
    pub fn new(
        input_size: (i32, i32),
        reg_max: i32,
        strides: Vec<i32>,
        num_classes: usize,
    ) -> Self {
        let mut nanodet = NanodetDecoder {
            reg_max,
            input_size,
            num_classes,
            strides,
            center_priors: Vec::new(),
        };

        nanodet.generate_grid_center_priors();

        nanodet
    }

    pub fn input_size(&self) -> (i32, i32) {
        self.input_size
    }

    pub fn reg_max(&self) -> i32 {
        self.reg_max
    }

    pub fn strides(&self) -> &[i32] {
        &self.strides
    }

    /// Decodes output matrix of nanodet, returning detections grouped by class
    pub fn decode(&self, features: &ncnn_rs::Mat, threshold: f32) -> Vec<Vec<Detection>> {
        let mut results = (0..self.num_classes)
            .into_iter()
            .map(|_| Vec::new())
            .collect::<Vec<_>>();

        for (idx, ct) in self.center_priors.iter().enumerate() {
            let scores = ncnn_get_f32_row(features, idx);

            let (label, score) = scores
                .iter()
                .take(self.num_classes)
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1))
                .unwrap();

            if *score >= threshold {
                let bbox =
                    self.dis_pred_to_bbox(&scores[self.num_classes..], ct.x, ct.y, ct.stride);
                results[label].push(Detection {
                    bbox,
                    score: *score,
                    label,
                })
            }
        }

        results
    }

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

    /// Generates a list of [CenterPrior] based on input size and strides options.
    fn generate_grid_center_priors(&mut self) {
        self.center_priors.clear();

        for stride in self.strides.iter().copied() {
            let feat_w = div_ceil(self.input_size.0, stride);
            let feat_h = div_ceil(self.input_size.1, stride);
            for y in 0..feat_h {
                for x in 0..feat_w {
                    self.center_priors.push(CenterPrior { x, y, stride })
                }
            }
        }
    }
}

/// Performs non-max suppression filtering
pub fn nms_filter(detections: &mut Vec<Detection>, threshold: f32) {
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
