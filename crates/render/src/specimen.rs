//! Characters set in a font program on their own, no page around them: the reference shapes a reading of glyphs
//! by their shape compares against (`document_lab.glyphs`).

use pdf::geometry::Point;
use pdf::{Error, Result};
use tiny_skia::{FillRule, Paint, Pixmap, Transform};

/// `text` set in the font program `program` ([`pdf::shape::character`]: each character by its Unicode `cmap`, each
/// advancing by its glyph's advance) at `size` pixels an em, the first glyph's origin on the baseline at `origin`
/// (pixels, y down), on a canvas `width` by `height` pixels: each pixel's ink, 0 (none) to 255 (covered), rows
/// top-down, anti-aliased as a page's glyphs are (but placed where stated: not moved onto `MuPDF`'s glyph grid).
///
/// # Errors
///
/// Refuses an empty canvas or one of more than 2^28 pixels, and what [`pdf::shape::character`] refuses.
pub fn specimen(
    program: &[u8],
    text: &str,
    size: f64,
    origin: Point,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    if u64::from(width) * u64::from(height) > 1 << 28 {
        return Err(Error::invalid("a specimen canvas of more than 2^28 pixels"));
    }
    let mut pixmap =
        Pixmap::new(width, height).ok_or_else(|| Error::invalid("an empty specimen canvas"))?;
    let mut paint = Paint {
        anti_alias: true,
        ..Paint::default()
    };
    paint.set_color(tiny_skia::Color::BLACK);
    let mut x = origin.x;
    for c in text.chars() {
        let (outline, advance) = pdf::shape::character(program, c)?;
        let place = |p: Point| Point {
            x: x + p.x * size,
            y: origin.y - p.y * size,
        };
        if let Some(path) = crate::build(&outline, &place) {
            pixmap.fill_path(
                &path,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                None,
            );
        }
        x += advance * size;
    }
    Ok(pixmap.pixels().iter().map(|p| p.alpha()).collect())
}
