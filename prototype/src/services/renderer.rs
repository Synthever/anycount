use image::{codecs::jpeg::JpegEncoder, DynamicImage, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufWriter, path::Path};

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

/// 5x7 bitmap glyphs (bit 4 = left-most column) used by the server-side counter
/// badge so the JPEG matches the SPA's `<query>: <count>` overlay without
/// pulling in a font-rendering dependency.
fn glyph(c: char) -> [u8; 7] {
    match c {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b10101, 0b01010,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
        ],
        '6' => [
            0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100,
        ],
        ':' => [
            0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000,
        ],
        '.' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110,
        ],
        '-' => [
            0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
        ],
        '/' => [
            0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000,
        ],
        '+' => [
            0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000,
        ],
        '(' => [
            0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010,
        ],
        ')' => [
            0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000,
        ],
        '_' => [
            0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111,
        ],
        ' ' => [0; 7],
        _ => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100,
        ],
    }
}

fn fill_rect(img: &mut RgbImage, x0: i32, y0: i32, w: i32, h: i32, color: [u8; 3], alpha: f32) {
    let (iw, ih) = (img.width() as i32, img.height() as i32);
    for y in y0.max(0)..(y0 + h).min(ih) {
        for x in x0.max(0)..(x0 + w).min(iw) {
            blend_px(img, x as u32, y as u32, color, alpha);
        }
    }
}

/// Draws the same top-left counter badge as the web canvas: dark translucent
/// box, white bitmap text `"<query>: <count>"`.
fn draw_counter_badge(img: &mut RgbImage, query: &str, count: usize) {
    let text = format!("{query}: {count}").to_uppercase();
    let min_dim = img.width().min(img.height()) as f64;
    let font_px = ((min_dim / 36.0).round() as i32).max(14);
    let scale = (font_px / 7).max(1);
    let char_h = 7 * scale;
    let pad_x = ((font_px as f64) * 0.7).round() as i32;
    let pad_y = ((font_px as f64) * 0.45).round() as i32;
    let text_w = ((text.chars().count() as i32) * 6 - 1).max(0) * scale;
    let box_w = text_w + pad_x * 2;
    let box_h = char_h + pad_y * 2;

    let bx = 12.min((img.width() as i32 - box_w - 1).max(0));
    let by = 12.min((img.height() as i32 - box_h - 1).max(0));

    fill_rect(img, bx, by, box_w, box_h, [10, 10, 12], 0.82);

    let mut cx = bx + pad_x;
    let cy = by + pad_y;
    for ch in text.chars() {
        let rows = glyph(ch);
        for (row, bits) in rows.iter().enumerate() {
            for col in 0..5 {
                if bits & (1 << (4 - col)) != 0 {
                    fill_rect(
                        img,
                        cx + col * scale,
                        cy + (row as i32) * scale,
                        scale,
                        scale,
                        [255, 255, 255],
                        1.0,
                    );
                }
            }
        }
        cx += 6 * scale;
    }
}

/// Draws Count-Anything style red dots (230, 40, 35) with a crisp 2 px white
/// outer ring, optional subtle 2x2 grid lines and a `<query>: <count>` badge
/// onto `src_path`, saving JPEG to `out_path`.
pub fn render_annotated_jpeg(
    src_path: &Path,
    points: &[Point2D],
    grid_boxes: Option<&[[i32; 4]]>,
    out_path: &Path,
    query: Option<&str>,
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
    let ring_w = 2.0;
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

    if let Some(q) = query {
        if !q.trim().is_empty() {
            draw_counter_badge(&mut img, q.trim(), points.len());
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
