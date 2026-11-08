use std::{io::Cursor, path::Path};
use image::{codecs::jpeg::JpegEncoder, DynamicImage};

use crate::services::renderer::Point2D;

#[derive(Clone, Debug)]
pub struct TileSpec {
    pub tile_id: String,
    pub crop_box: [u32; 4], // [x0, y0, x1, y1]
    pub core_box: [f64; 4], // [x0, y0, x1, y1]
    pub seam_tol_px: f64,
    pub jpeg_bytes: Vec<u8>,
    pub tile_w: u32,
    pub tile_h: u32,
}

#[derive(Clone, Debug)]
pub struct TiledPoint {
    pub x: f64,
    pub y: f64,
    pub tile_id: String,
}

pub fn split_2x2(
    img_path: &Path,
    overlap_ratio: f64,
) -> anyhow::Result<(u32, u32, Vec<[i32; 4]>, Vec<TileSpec>)> {
    let img = image::open(img_path)?.to_rgb8();
    let (w, h) = img.dimensions();
    let rows = 2usize;
    let cols = 2usize;
    let cell_w = w as f64 / cols as f64;
    let cell_h = h as f64 / rows as f64;
    let pad_x = cell_w * overlap_ratio;
    let pad_y = cell_h * overlap_ratio;
    let seam_tol = cell_w.min(cell_h) * 0.04;

    let mut core_boxes = Vec::with_capacity(4);
    let mut tiles = Vec::with_capacity(4);

    for r in 0..rows {
        for c in 0..cols {
            let core_x0 = c as f64 * cell_w;
            let core_y0 = r as f64 * cell_h;
            let core_x1 = (c + 1) as f64 * cell_w;
            let core_y1 = (r + 1) as f64 * cell_h;
            core_boxes.push([
                core_x0 as i32,
                core_y0 as i32,
                core_x1 as i32,
                core_y1 as i32,
            ]);

            let crop_x0 = if c > 0 { (core_x0 - pad_x).round().max(0.0) as u32 } else { 0 };
            let crop_y0 = if r > 0 { (core_y0 - pad_y).round().max(0.0) as u32 } else { 0 };
            let crop_x1 = if c < cols - 1 {
                (core_x1 + pad_x).round().min(w as f64) as u32
            } else {
                w
            };
            let crop_y1 = if r < rows - 1 {
                (core_y1 + pad_y).round().min(h as f64) as u32
            } else {
                h
            };

            let tw = crop_x1.saturating_sub(crop_x0).max(1);
            let th = crop_y1.saturating_sub(crop_y0).max(1);
            let sub = image::imageops::crop_imm(&img, crop_x0, crop_y0, tw, th).to_image();

            let mut jpeg_bytes = Vec::new();
            let mut cursor = Cursor::new(&mut jpeg_bytes);
            let mut enc = JpegEncoder::new_with_quality(&mut cursor, 90);
            enc.encode_image(&DynamicImage::ImageRgb8(sub))?;

            tiles.push(TileSpec {
                tile_id: format!("r{r}_c{c}"),
                crop_box: [crop_x0, crop_y0, crop_x1, crop_y1],
                core_box: [core_x0, core_y0, core_x1, core_y1],
                seam_tol_px: seam_tol,
                jpeg_bytes,
                tile_w: tw,
                tile_h: th,
            });
        }
    }

    Ok((w, h, core_boxes, tiles))
}

pub fn adaptive_point_nms(points: &[TiledPoint], img_w: u32, img_h: u32) -> Vec<Point2D> {
    let min_dim = img_w.min(img_h) as f64;
    let nms_radius = if points.len() >= 4 {
        let mut nn_dists = Vec::new();
        for (i, p1) in points.iter().enumerate() {
            let mut dmin = f64::INFINITY;
            for (j, p2) in points.iter().enumerate() {
                if i != j {
                    let d = (p1.x - p2.x).hypot(p1.y - p2.y);
                    if d < dmin {
                        dmin = d;
                    }
                }
            }
            if dmin.is_finite() && dmin > 1.0 {
                nn_dists.push(dmin);
            }
        }
        nn_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_nn = if nn_dists.is_empty() {
            20.0
        } else {
            nn_dists[nn_dists.len() / 2]
        };
        (median_nn * 0.55).clamp(8.0, min_dim * 0.06)
    } else {
        (min_dim * 0.04).max(8.0)
    };

    let mut kept: Vec<TiledPoint> = Vec::new();
    for pt in points {
        let mut merged = false;
        for k in &mut kept {
            let dist = (pt.x - k.x).hypot(pt.y - k.y);
            let factor = if pt.tile_id != k.tile_id { 1.35 } else { 0.75 };
            if dist < nms_radius * factor {
                k.x = (k.x + pt.x) / 2.0;
                k.y = (k.y + pt.y) / 2.0;
                merged = true;
                break;
            }
        }
        if !merged {
            kept.push(pt.clone());
        }
    }

    kept.into_iter()
        .map(|p| Point2D {
            x: (p.x * 100.0).round() / 100.0,
            y: (p.y * 100.0).round() / 100.0,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use tempfile::TempDir;

    #[test]
    fn split_2x2_and_seam_nms_deduplicates_border_pair() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().join("grid.jpg");
        RgbImage::from_pixel(640, 480, Rgb([20, 30, 40])).save(&p).unwrap();

        let (w, h, cores, tiles) = split_2x2(&p, 0.08).unwrap();
        assert_eq!((w, h), (640, 480));
        assert_eq!(cores.len(), 4);
        assert_eq!(tiles.len(), 4);
        assert!(tiles[0].tile_w > 320 && tiles[0].tile_h > 240);

        let raw = vec![
            TiledPoint { x: 100.0, y: 100.0, tile_id: "r0_c0".into() },
            TiledPoint { x: 319.0, y: 120.0, tile_id: "r0_c0".into() },
            TiledPoint { x: 322.0, y: 121.0, tile_id: "r0_c1".into() }, // duplicate on vertical seam
            TiledPoint { x: 500.0, y: 380.0, tile_id: "r1_c1".into() },
        ];
        let dedup = adaptive_point_nms(&raw, 640, 480);
        assert_eq!(dedup.len(), 3);
    }
}
