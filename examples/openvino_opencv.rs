use std::io::{BufRead, BufReader};
use std::path::Path;

use nanodet_rs::image_utils::opencv::{draw_detections, uniform_resize};
use nanodet_rs::{nms_filter, NanodetDecoder};
use opencv::core::VecN;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::prelude::MatTraitConst;
use openvino::Core;

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

    let mut core = Core::new(None).unwrap();
    let network = core
        .read_network_from_file(
            "data/nanodet-plus-m_416_openvino.xml",
            "data/nanodet-plus-m_416_openvino.bin",
        )
        .unwrap();

    // Load the network.
    let mut executable_network = core.load_network(&network, "CPU").unwrap();
    let mut infer_request = executable_network.create_infer_request().unwrap();

    // Get the input tensor
    let mut input_blob = infer_request.get_blob("data").unwrap();

    // Write u8 image into f32 input tensor
    let input_data = unsafe { input_blob.buffer_mut_as_type::<f32>().unwrap() };
    for c in 0..3 {
        for h in 0..input_size.1 {
            for w in 0..input_size.0 {
                input_data[(c * input_size.1 * input_size.0 + h * input_size.0 + w) as usize] =
                    f32::from(resized_img.at_2d::<VecN<u8, 3>>(h, w).unwrap()[c as usize]);
            }
        }
    }

    // Execute inference.
    infer_request.infer().unwrap();
    let results = infer_request.get_blob("output").unwrap();

    let score_threshold = 0.4;
    let nms_threshold = 0.5;

    let mut results = decoder.decode(&results, score_threshold);

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
