use std::io::{BufRead, BufReader};
use std::path::Path;

use ab_glyph::FontArc;
use image::io::Reader;
use image::Rgb;
use imageproc::map::map_colors;
use nanodet_rs::image_utils::image::{draw_detections, uniform_resize};
use nanodet_rs::{nms_filter, NanodetDecoder};
use ncnn_rs::{MatPixelType, Net};
use show_image::{create_window, event};

#[show_image::main]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "Usage: ./image test_image.jpg");

    let labels = load_labels(Path::new("data/coco_labels.txt"))?;
    let labels_str: Vec<_> = labels.iter().map(String::as_str).collect();
    assert_eq!(labels.len(), 80);

    let input_size = (416, 416);
    let decoder = NanodetDecoder::new(input_size, 7, vec![8, 16, 32, 64], labels.len());

    let mut img = Reader::open(&args[1])?.decode()?;

    // Resize image to square without stretching (adds padding)
    let (resized_img, effective_area) =
        uniform_resize(&img, input_size.0 as u32, input_size.1 as u32);

    // Hack to map image from RGB to BGR, since that is what nanodet supports
    let resized_img_bgr = map_colors(&resized_img, |Rgb([r, g, b])| Rgb([b, g, r]));

    let mut net = Net::new();
    net.load_param("data/nanodet-plus-m_416.param")?;
    net.load_model("data/nanodet-plus-m_416.bin")?;

    let mut input = ncnn_rs::Mat::from_pixels(
        resized_img_bgr.as_flat_samples().image_slice().unwrap(),
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

    let font = Vec::from(include_bytes!("../data/DejaVuSans.ttf") as &[u8]);
    let font = FontArc::try_from_vec(font).unwrap();
    draw_detections(&mut img, &font, effective_area, &detections, &labels_str);

    // Create a window with default options and display the image.
    let window = create_window("image", Default::default())?;
    window.set_image("image", img)?;

    // Wait for the window to be closed or Escape to be pressed.
    for event in window.event_channel().map_err(|e| e.to_string())? {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if !event.is_synthetic
                && event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                println!("Escape pressed!");
                break;
            }
        }
    }

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
