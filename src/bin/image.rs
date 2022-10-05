use ncnn_nanodet_rs::image_util::{draw_detections, uniform_resize};
use ncnn_nanodet_rs::{nms_filter, NanodetDecoder};
use ncnn_rs::{MatPixelType, Net};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::prelude::MatTraitConstManual;

// Classnames for the pretrained COCO model in `data/`
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
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 2, "Usage: ./image test_image.jpg");

    let input_size = (416, 416);
    let decoder = NanodetDecoder::new(input_size, 7, vec![8, 16, 32, 64], CLASS_NAMES.len());

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

    draw_detections(&mut img, effective_area, &detections, CLASS_NAMES);

    imshow("image", &img).unwrap();
    wait_key(0).unwrap();

    Ok(())
}
