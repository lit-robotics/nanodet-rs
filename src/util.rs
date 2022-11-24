pub fn div_ceil(lhs: i32, rhs: i32) -> i32 {
    let d = lhs / rhs;
    let r = lhs % rhs;
    if (r > 0 && rhs > 0) || (r < 0 && rhs < 0) {
        d + 1
    } else {
        d
    }
}

pub fn fast_exp(x: f32) -> f32 {
    let i: u32 = ((1 << 23) as f64 * (1.4426950409 * x as f64 + 126.93490512)) as u32;
    f32::from_bits(i)
}

pub fn activation_softmax(src: &[f32]) -> Vec<f32> {
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
