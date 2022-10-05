#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub xmin: f32,
    pub ymin: f32,
    pub xmax: f32,
    pub ymax: f32,
}

impl BoundingBox {
    /// Returns the intersection of two [BoundingBox] or None if they have no overlap.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
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

    pub fn area(&self) -> f32 {
        (self.xmax - self.xmin + 1.0).max(0.0) * (self.ymax - self.ymin + 1.0).max(0.0)
    }
}
