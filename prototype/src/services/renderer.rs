use std::{fs::File, io::BufWriter, path::Path};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

pub fn save_jpeg(img: &RgbImage, out_path: &Path, quality: u8) -> anyhow::Result<()> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = File::create(out_path)?;
    let mut writer = BufWriter::new(file);
    let mut enc = JpegEncoder::new_with_quality(&mut writer, quality);
    enc.encode_image(&DynamicImage::ImageRgb8(img.clone()))?;
    Ok(())
}

/// Draws Count-Anything style red dots (230, 40, 35) with crisp white outer ring (255, 255, 255)
/// and optional subtle 2x2 grid lines onto `src_path`, saving JPEG to `out_path`.
pub fn render_annotated_jpeg(
    src_path: &Path,
    points: &[Point2D],
    grid_boxes: Option<&[[i32; 4]]>,
    out_path: &Path,
) -> anyhow::Result<()> {
    let mut img = image::open(src_path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let w_i = w as i32;
    let h_i = h as i32;

    if let Some(boxes) = grid_boxes {
        for &[x0, y0, x1, y1] in boxes {
            for x in x0.max(0)..=x1.min(w_i - 1) {
                for &yy in &[y0, y1] {
                    if yy >= 0 && yy < h_i {
                        blend_px(&mut img, x as u32, yy as u32, [255, 255, 255], 0.40);
                    }
                }
            }
            for y in y0.max(0)..=y1.min(h_i - 1) {
                for &xx in &[x0, x1] {
                    if xx >= 0 && xx < w_i {
                        blend_px(&mut img, xx as u32, y as u32, [255, 255, 255], 0.40);
                    }
                }
            }
        }
    }

    let min_dim = w.min(h) as f64;
    let r_inner = (min_dim / 180.0).round().max(4.0);
    let ring_w = (r_inner / 2.0).floor().max(1.5);
    let r_outer = r_inner + ring_w;
    let pad = (r_outer.ceil() as i32) + 1;

    for pt in points {
        let cx = pt.x.clamp(0.0, (w - 1) as f64);
        let cy = pt.y.clamp(0.0, (h - 1) as f64);
        let x_start = ((cx as i32) - pad).max(0);
        let x_end = ((cx as i32) + pad).min(w_i - 1);
        let y_start = ((cy as i32) - pad).max(0);
        let y_end = ((cy as i32) + pad).min(h_i - 1);

        for py in y_start..=y_end {
            for px in x_start..=x_end {
                let dist = ((px as f64 - cx).powi(2) + (py as f64 - cy).powi(2)).sqrt();
                if dist <= r_inner {
                    blend_px(&mut img, px as u32, py as u32, [230, 40, 35], 0.92);
                } else if dist <= r_outer {
                    img.put_pixel(px as u32, py as u32, Rgb([255, 255, 255]));
                }
            }
        }
    }

    save_jpeg(&img, out_path, 92)
}

fn blend_px(img: &mut RgbImage, x: u32, y: u32, fg: [u8; 3], alpha: f32) {
    let bg = img.get_pixel(x, y).0;
    let out = [
        (fg[0] as f32 * alpha + bg[0] as f32 * (1.0 - alpha)).round() as u8,
        (fg[1] as f32 * alpha + bg[1] as f32 * (1.0 - alpha)).round() as u8,
        (fg[2] as f32 * alpha + bg[2] as f32 * (1.0 - alpha)).round() as u8,
    ];
    img.put_pixel(x, y, Rgb(out));
}
