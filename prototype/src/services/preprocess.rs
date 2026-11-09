use std::path::Path;

use image::RgbImage;

use crate::services::renderer::save_jpeg;

/// RGB negative: every channel becomes `255 - value`.
pub fn invert_rgb(img: &RgbImage) -> RgbImage {
    let mut out = img.clone();
    image::imageops::invert(&mut out);
    out
}

/// Writes the RGB-negative derivative of `src` to `dst` (JPEG q90) unless it
/// already exists. Images are immutable, so the derivative is cached per image.
pub fn ensure_inverted_jpeg(src: &Path, dst: &Path) -> anyhow::Result<()> {
    if dst.exists() {
        return Ok(());
    }
    let img = image::open(src)?.to_rgb8();
    save_jpeg(&invert_rgb(&img), dst, 90)
}

#[cfg(test)]
mod tests {
    use super::invert_rgb;
    use image::{Rgb, RgbImage};

    #[test]
    fn inverts_every_channel_of_every_pixel() {
        let mut img = RgbImage::new(2, 1);
        img.put_pixel(0, 0, Rgb([0, 10, 255]));
        img.put_pixel(1, 0, Rgb([255, 200, 128]));

        let out = invert_rgb(&img);

        assert_eq!(out.get_pixel(0, 0), &Rgb([255, 245, 0]));
        assert_eq!(out.get_pixel(1, 0), &Rgb([0, 55, 127]));
    }
}
