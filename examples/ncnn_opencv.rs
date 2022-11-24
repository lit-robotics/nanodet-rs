use std::io::{BufRead, BufReader};
use std::path::Path;

use ncnn_nanodet_rs::image_utils::opencv::{draw_detections, uniform_resize};
use ncnn_nanodet_rs::{nms_filter, NanodetDecoder};
use ncnn_rs::{MatPixelType, Net};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::prelude::MatTraitConstManual;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "Usage: ./image test_image.jpg");

    let labels = load_labels(Path::new("data/coco_labels.txt"))?;
    let labels_str: Vec<_> = labels.iter().map(String::as_str).collect();
    assert_eq!(labels.len(), 80);

    let input_size = (416, 416);
    let decoder = NanodetDecoder::new(input_size, 7, vec![8, 16, 32, 64], labels.len());

    let mut img = imread(&args[1], IMREAD_COLOR)?;

    // Resize image to square without stretching (adds padding)
    let (resized_img, effective_area) = uniform_resize(&img, input_size.0, input_size.1);

    let mut net = Net::new();
    net.load_param("data/nanodet-plus-m_416.param")?;
    net.load_model("data/nanodet-plus-m_416.bin")?;

    let mut input = ncnn_rs::Mat::from_pixels(
        resized_img.data_bytes().unwrap(),
        MatPixelType::BGR,
        input_size.0,
        input_size.1,
        None,
    )?;
    let mut output = ncnn_rs::Mat::new();

    let mean_vals = [183.53, 116.28, 123.675];
    let norm_vals = [0.017429, 0.017507, 0.017125];
    input.substract_mean_normalize(&mean_vals, &norm_vals);

    // Extractors are consumed and can only be used for single inference.
    // Creating them is cheap, so this step can be performed in a loop for video processing.
    let mut extractor = net.create_extractor();
    extractor.input("data", &input)?;
    extractor.extract("output", &mut output)?;

    let score_threshold = 0.4;
    let nms_threshold = 0.5;

    let mut results = decoder.decode(&output, score_threshold);

    // Perform NMS filtering
    let mut detections = Vec::new();
    for class in results.iter_mut() {
        nms_filter(class, nms_threshold);
        detections.append(class);
    }

    draw_detections(&mut img, effective_area, &detections, &labels_str);

    imshow("image", &img).unwrap();
    wait_key(0).unwrap();

    Ok(())
}

fn load_labels(path: &Path) -> std::io::Result<Vec<String>> {
    let file = std::fs::File::open(path)?;
    Ok(BufReader::new(file)
        .lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.is_empty())
        .collect())
}
