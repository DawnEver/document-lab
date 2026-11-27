//! document-lab's own renderer: what a page draws, from `pdf`'s content interpreter, as 8-bit pixels.
//!
//! [`render`] interprets a page ([`pdf::File::interpret`]) and paints its events in order onto a white (or
//! transparent) canvas: paths filled and stroked as the graphics state says (width, caps, joins, miter limit,
//! dashes, the nonzero or even-odd rule, a stroke under a matrix that scales unevenly stroked in user space), images
//! placed by their matrix in every color space the reader converts ([`image`]), each mark clipped by the chain of
//! clipping paths in force ([`clip`]) and composited with its constant alpha and blend mode, anti-aliased by
//! `tiny-skia`. The pixel grid is the page's: pixel `(i, j)` covers page space `[i, i + 1] x [j, j + 1]` divided by
//! the scale, and a region is drawn on the pixels it touches -- so a region's pixels are the whole page's there, but
//! for the anti-aliasing of a curve's edge, which differs by a few coverage steps where the region's edge cuts it.
//!
//! What is not drawn yet is never silently missing. Text is drawn from stage 2 (font outlines): [`Options::text`]
//! must be false, and true is refused by name. What else in the region the renderer cannot draw as the file states
//! it is counted by kind in [`Raster::undrawn`] -- a shading or a pattern (stage 3), a soft mask (drawn without it),
//! a text clip (approximated by its glyphs' boxes), an image in a codec not decoded here (JPX, JBIG2) or that does not
//! decode -- for the caller to refuse or accept. Hidden optional content and what the colorant `None` paints mark
//! nothing and are not drawn.
//!
//! Layers: [`codec`] (image codecs to samples) -> [`image`] (an image's samples to pixels) and [`clip`] (clipping
//! paths to masks) -> this module (the page painted).

pub mod clip;
pub mod codec;
pub mod image;

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use pdf::color::Color;
use pdf::geometry::{Matrix, Point, Rect};
use pdf::interpret::{
    Blend, Compositing, Event, ImageEvent, Layer, PathEvent, Segment, ShadingEvent, Stroke,
};
use pdf::{Error, File, Page, Result};
use tiny_skia::{
    BlendMode, FillRule, FilterQuality, LineCap, LineJoin, Paint, PathBuilder, Pixmap, SpreadMode,
    StrokeDash, Transform,
};

use crate::clip::{Clips, Left};

/// What to draw and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// The part of the page drawn, in page space (points, y down, origin at the top-left as displayed); None: the
    /// whole page.
    pub region: Option<Rect>,
    /// Pixels per point: the resolution over 72 dpi.
    pub scale: f64,
    /// Whether text is drawn. Only false is accepted until stage 2 draws glyphs from their font outlines.
    pub text: bool,
    /// Whether each annotation's appearance is drawn after the page's content.
    pub annotations: bool,
    /// Whether what nothing covers stays transparent (RGBA out) rather than white (RGB out).
    pub alpha: bool,
}

/// A region of a page as pixels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Raster {
    /// Pixels across.
    pub width: u32,
    /// Pixels down.
    pub height: u32,
    /// Bytes a pixel: 3 (RGB, on white) or 4 (RGBA, not premultiplied).
    pub channels: u8,
    /// The pixels, rows top-down.
    pub data: Vec<u8>,
    /// Each kind of mark meeting the region that is not drawn as the file states it, with how many: see the
    /// module. Empty when the region is drawn whole.
    pub undrawn: BTreeMap<String, usize>,
}

/// The `options.region` of `page` drawn at `options.scale`.
///
/// # Errors
///
/// Refuses text (`options.text`: stage 2), a scale that is not positive and finite, a region outside the page or
/// of more than 2^28 pixels, and what [`File::interpret`] refuses.
pub fn render(file: &File, page: &Page, options: &Options) -> Result<Raster> {
    if options.text {
        return Err(Error::unsupported(
            "drawing text is stage 2 of the renderer (glyphs from font outlines); render with text off",
        ));
    }
    let (width, height) = page.size();
    let whole = Rect {
        x0: 0.0,
        y0: 0.0,
        x1: width,
        y1: height,
    };
    let region = options.region.unwrap_or(whole).intersect(whole);
    if !(options.scale.is_finite() && options.scale > 0.0) {
        return Err(Error::invalid(format!(
            "a scale of {} pixels a point",
            options.scale
        )));
    }
    let grid = Grid::of(region, options.scale)
        .ok_or_else(|| Error::invalid("the region holds no pixel of the page"))?;
    let content = file.interpret(page, options.annotations)?;
    let mut canvas = Canvas {
        file,
        pixmap: Pixmap::new(grid.width, grid.height)
            .ok_or_else(|| Error::invalid("the region is too large to draw"))?,
        grid,
        clips: Clips::default(),
        images: HashMap::new(),
        undrawn: BTreeMap::new(),
    };
    if !options.alpha {
        canvas.pixmap.fill(tiny_skia::Color::WHITE);
    }
    for event in &content.events {
        canvas.event(event);
    }
    Ok(canvas.finish(options.alpha))
}

/// The pixels a region covers: how page space maps onto them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Grid {
    pub(crate) scale: f64,
    /// The page-space pixel column and row of the canvas's top-left pixel.
    pub(crate) left: f64,
    pub(crate) top: f64,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

impl Grid {
    /// The whole pixels `region` touches at `scale` (an edge within a thousandth of a pixel of a pixel's edge is on
    /// it, so a region drawn at whole pixels gets exactly those), or None when it touches none.
    fn of(region: Rect, scale: f64) -> Option<Self> {
        const SNAP: f64 = 1e-3;
        let (left, top) = (
            (region.x0 * scale + SNAP).floor(),
            (region.y0 * scale + SNAP).floor(),
        );
        let (right, bottom) = (
            (region.x1 * scale - SNAP).ceil(),
            (region.y1 * scale - SNAP).ceil(),
        );
        let (width, height) = (right - left, bottom - top);
        if !(width >= 1.0 && height >= 1.0 && width * height <= f64::from(1u32 << 28)) {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "whole, positive and bounded above"
        )]
        Some(Self {
            scale,
            left,
            top,
            width: width as u32,
            height: height as u32,
        })
    }

    /// The matrix from page space to the canvas's pixels.
    fn matrix(&self) -> Matrix {
        Matrix([self.scale, 0.0, 0.0, self.scale, -self.left, -self.top])
    }

    /// The canvas's pixels, as a box in pixel space.
    pub(crate) fn bounds(&self) -> Rect {
        Rect {
            x0: 0.0,
            y0: 0.0,
            x1: f64::from(self.width),
            y1: f64::from(self.height),
        }
    }

    /// `p` (page space) in pixel space.
    pub(crate) fn point(&self, p: Point) -> Point {
        self.matrix().apply(p)
    }

    /// `r` (page space) in pixel space.
    pub(crate) fn rect(&self, r: Rect) -> Rect {
        Rect::spanning(
            self.point(Point { x: r.x0, y: r.y0 }),
            self.point(Point { x: r.x1, y: r.y1 }),
        )
    }
}

/// A page being painted.
struct Canvas<'a> {
    file: &'a File,
    pixmap: Pixmap,
    grid: Grid,
    clips: Clips,
    /// Each image `XObject` decoded so far (not a stencil mask: its pixels are the fill color's), by its number.
    images: HashMap<u32, Rc<std::result::Result<image::Decoded, String>>>,
    undrawn: BTreeMap<String, usize>,
}

impl Canvas<'_> {
    fn event(&mut self, event: &Event) {
        match event {
            Event::Path(path) if shows(&path.layer) => self.path(path),
            Event::Image(image) if shows(&image.layer) => self.image(image),
            Event::Shading(shading) if shows(&shading.layer) => self.shading(shading),
            _ => {} // hidden, inkless, or text (refused unless off)
        }
    }

    /// `what` meeting the region was not drawn as the file states it.
    fn undrawn(&mut self, what: &str) {
        *self.undrawn.entry(what.to_string()).or_default() += 1;
    }

    /// Whether `bbox` (page space) meets the canvas.
    fn meets(&self, bbox: Rect) -> bool {
        self.grid.rect(bbox).meets(self.grid.bounds())
    }

    /// The compositing a mark is drawn with, its soft mask (not drawn) counted.
    fn composited(&mut self, compositing: Compositing) -> BlendMode {
        if compositing.soft_mask {
            self.undrawn("soft mask (stage 3; drawn without it)");
        }
        blend_mode(compositing.blend)
    }

    fn path(&mut self, path: &PathEvent) {
        let Some(bounds) = bounds(&path.segments) else {
            return;
        };
        let reach = path
            .stroke
            .as_ref()
            .map_or(0.0, |s| s.width * s.miter_limit.max(1.0) + s.width);
        let reach_box = Rect {
            x0: bounds.x0 - reach,
            y0: bounds.y0 - reach,
            x1: bounds.x1 + reach,
            y1: bounds.y1 + reach,
        };
        if !self.meets(reach_box) {
            return;
        }
        let mask = match self
            .clips
            .mask(path.clip.as_ref(), &self.grid, &mut self.undrawn)
        {
            Left::Nothing => return, // clipped away whole
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let blend = self.composited(path.compositing);
        if let Some(fill) = &path.fill
            && let Some(rgb) = self.paint_color(&fill.color, "fill")
        {
            let Some(outline) = build(&path.segments, &|p| self.grid.point(p)) else {
                return;
            };
            let paint = paint(rgb, path.compositing.fill_alpha, blend);
            let rule = if fill.even_odd {
                FillRule::EvenOdd
            } else {
                FillRule::Winding
            };
            self.pixmap.fill_path(
                &outline,
                &paint,
                rule,
                Transform::identity(),
                mask.as_deref(),
            );
        }
        if let Some(stroke) = &path.stroke
            && let Some(rgb) = self.paint_color(&stroke.color, "stroke")
        {
            let paint = paint(rgb, path.compositing.stroke_alpha, blend);
            self.stroke(&path.segments, stroke, &paint, mask.as_deref());
        }
    }

    /// The RGB `color` paints in, when it is one color the reader converts; else None, counted.
    fn paint_color(&mut self, color: &Color, what: &str) -> Option<[f64; 3]> {
        match (color.pattern, color.rgb) {
            (Some(_), _) => {
                self.undrawn(&format!("pattern {what} (stage 3)"));
                None
            }
            (None, Some(rgb)) => Some(rgb),
            (None, None) => {
                self.undrawn(&format!(
                    "{what} in a {} color with no RGB",
                    String::from_utf8_lossy(&color.space)
                ));
                None
            }
        }
    }

    /// `segments` stroked as `stroke` says: in pixel space when the matrix it was painted under scales evenly (a
    /// width is then the same every way), else in user space through that matrix.
    fn stroke(
        &mut self,
        segments: &[Segment],
        stroke: &Stroke,
        paint: &Paint,
        mask: Option<&tiny_skia::Mask>,
    ) {
        let page_to_pixels = self.grid.matrix();
        let (outline, transform, scale) = if let Some(inverse) = user_space(stroke.ctm) {
            let to_pixels = stroke.ctm.then(page_to_pixels);
            let Some(outline) = build(segments, &|p| inverse.apply(p)) else {
                return;
            };
            (outline, transform(to_pixels), 1.0 / stroke.ctm.scale())
        } else {
            let Some(outline) = build(segments, &|p| self.grid.point(p)) else {
                return;
            };
            (outline, Transform::identity(), self.grid.scale)
        };
        // A line thinner than a tenth of a pixel is drawn one pixel wide, the thinnest a device shows (8.4.3.2).
        let pixels = stroke.width * self.grid.scale;
        let width = if pixels < 0.1 {
            0.0
        } else {
            stroke.width * scale
        };
        let style = tiny_skia::Stroke {
            #[expect(clippy::cast_possible_truncation, reason = "a pixel width")]
            width: width as f32,
            #[expect(clippy::cast_possible_truncation, reason = "a ratio")]
            miter_limit: stroke.miter_limit.max(1.0) as f32,
            line_cap: match stroke.cap {
                1 => LineCap::Round,
                2 => LineCap::Square,
                _ => LineCap::Butt,
            },
            line_join: match stroke.join {
                1 => LineJoin::Round,
                2 => LineJoin::Bevel,
                _ => LineJoin::Miter,
            },
            dash: dash(&stroke.dashes, stroke.dash_phase, scale),
        };
        self.pixmap
            .stroke_path(&outline, paint, &style, transform, mask);
    }

    fn image(&mut self, event: &ImageEvent) {
        if !self.meets(event.bbox) {
            return;
        }
        let mask = match self
            .clips
            .mask(event.clip.as_ref(), &self.grid, &mut self.undrawn)
        {
            Left::Nothing => return, // clipped away whole
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let decoded = if let Some(known) = event.object.and_then(|n| self.images.get(&n)) {
            Rc::clone(known)
        } else {
            let decoded = Rc::new(image::decode(self.file, event));
            if let (Some(n), Ok(d)) = (event.object, decoded.as_ref())
                && !d.stencil
            {
                self.images.insert(n, Rc::clone(&decoded));
            }
            decoded
        };
        let decoded = match decoded.as_ref() {
            Ok(decoded) => decoded,
            Err(why) => {
                self.undrawn(why);
                return;
            }
        };
        for note in &decoded.notes {
            self.undrawn(note);
        }
        let to_pixels = event.matrix.then(self.grid.matrix());
        let pixels = match image::place(decoded, to_pixels) {
            Ok(pixels) => pixels,
            Err(why) => {
                self.undrawn(&why);
                return;
            }
        };
        let Some(pixmap) = pixels.pixmap(decoded) else {
            return;
        };
        let blend = self.composited(event.compositing);
        let mut paint = Paint {
            blend_mode: blend,
            anti_alias: !pixels.grid_fitted,
            ..Paint::default()
        };
        #[expect(clippy::cast_possible_truncation, reason = "an alpha")]
        let opacity = event.compositing.fill_alpha as f32;
        paint.shader = tiny_skia::Pattern::new(
            pixmap,
            SpreadMode::Pad,
            if pixels.smooth {
                FilterQuality::Bilinear
            } else {
                FilterQuality::Nearest
            },
            opacity,
            Transform::identity(),
        );
        #[expect(clippy::cast_precision_loss, reason = "a pixel count")]
        let Some(rect) =
            tiny_skia::Rect::from_xywh(0.0, 0.0, pixmap.width() as f32, pixmap.height() as f32)
        else {
            return;
        };
        self.pixmap
            .fill_rect(rect, &paint, transform(pixels.matrix), mask.as_deref());
    }

    fn shading(&mut self, shading: &ShadingEvent) {
        if self.meets(shading.bbox) {
            self.undrawn("shading (stage 3)");
        }
    }

    fn finish(self, alpha: bool) -> Raster {
        let (width, height) = (self.pixmap.width(), self.pixmap.height());
        let (channels, data) = if alpha {
            (4, self.pixmap.take_demultiplied())
        } else {
            // On white every pixel is opaque: premultiplied is as it is.
            let rgb = self.pixmap.data().as_chunks::<4>().0.iter();
            (3, rgb.flat_map(|&[r, g, b, _]| [r, g, b]).collect())
        };
        Raster {
            width,
            height,
            channels,
            data,
            undrawn: self.undrawn,
        }
    }
}

/// Whether what is drawn in `layer` marks the page in its default view.
fn shows(layer: &Layer) -> bool {
    !layer.hidden && !layer.inkless
}

/// The inverse of `ctm` when a stroke under it is to be drawn in user space: when it scales unevenly (or skews), so
/// a width is not the same every way; None when it scales evenly, or cannot be inverted.
fn user_space(ctm: Matrix) -> Option<Matrix> {
    let [xx, xy, yx, yy, tx, ty] = ctm.0;
    let (across, down) = (xx.hypot(xy), yx.hypot(yy));
    let even = (across - down).abs() <= 1e-6 * across.max(down)
        && (xx * yx + xy * yy).abs() <= 1e-6 * across * down;
    let det = xx * yy - xy * yx;
    if even || det.abs() < 1e-12 {
        return None;
    }
    Some(Matrix([
        yy / det,
        -xy / det,
        -yx / det,
        xx / det,
        (yx * ty - yy * tx) / det,
        (xy * tx - xx * ty) / det,
    ]))
}

/// A path of `segments`, each point mapped by `map`; None when it draws nothing.
pub(crate) fn build(segments: &[Segment], map: &dyn Fn(Point) -> Point) -> Option<tiny_skia::Path> {
    #[expect(clippy::cast_possible_truncation, reason = "pixel coordinates")]
    let f = |p: Point| {
        let q = map(p);
        (q.x as f32, q.y as f32)
    };
    let mut builder = PathBuilder::new();
    for segment in segments {
        match *segment {
            Segment::MoveTo(p) => {
                let (x, y) = f(p);
                builder.move_to(x, y);
            }
            Segment::LineTo(p) => {
                let (x, y) = f(p);
                builder.line_to(x, y);
            }
            Segment::CurveTo(c1, c2, p) => {
                let ((x1, y1), (x2, y2), (x, y)) = (f(c1), f(c2), f(p));
                builder.cubic_to(x1, y1, x2, y2, x, y);
            }
            Segment::Close => builder.close(),
        }
    }
    builder.finish()
}

/// The box `segments`' points lie in, or None when there are none.
fn bounds(segments: &[Segment]) -> Option<Rect> {
    let points: Vec<Point> = segments
        .iter()
        .flat_map(|s| match *s {
            Segment::MoveTo(p) | Segment::LineTo(p) => vec![p],
            Segment::CurveTo(a, b, p) => vec![a, b, p],
            Segment::Close => vec![],
        })
        .collect();
    Rect::around(&points)
}

/// A dash pattern (page space) scaled to the space it is stroked in; None for a solid line. An odd count of lengths
/// repeats (8.4.3.6); lengths that are all zero, or any negative, draw a solid line.
fn dash(dashes: &[f64], phase: f64, scale: f64) -> Option<StrokeDash> {
    if dashes.is_empty() || dashes.iter().any(|d| *d < 0.0) || dashes.iter().all(|d| *d == 0.0) {
        return None;
    }
    let repeat = if dashes.len() % 2 == 1 { 2 } else { 1 };
    #[expect(clippy::cast_possible_truncation, reason = "pixel lengths")]
    let lengths = dashes
        .iter()
        .cycle()
        .take(dashes.len() * repeat)
        .map(|d| (d * scale) as f32)
        .collect();
    #[expect(clippy::cast_possible_truncation, reason = "a pixel length")]
    StrokeDash::new(lengths, (phase * scale) as f32)
}

/// A solid paint of `rgb` (0..=1) at `alpha`, blended by `blend`.
fn paint(rgb: [f64; 3], alpha: f64, blend: BlendMode) -> Paint<'static> {
    let mut paint = Paint {
        anti_alias: true,
        blend_mode: blend,
        ..Paint::default()
    };
    #[expect(clippy::cast_possible_truncation, reason = "color components")]
    paint.set_color(
        tiny_skia::Color::from_rgba(
            rgb[0] as f32,
            rgb[1] as f32,
            rgb[2] as f32,
            alpha.clamp(0.0, 1.0) as f32,
        )
        .unwrap_or(tiny_skia::Color::BLACK),
    );
    paint
}

/// `tiny-skia`'s blend mode for `blend`.
fn blend_mode(blend: Blend) -> BlendMode {
    match blend {
        Blend::Normal => BlendMode::SourceOver,
        Blend::Multiply => BlendMode::Multiply,
        Blend::Screen => BlendMode::Screen,
        Blend::Overlay => BlendMode::Overlay,
        Blend::Darken => BlendMode::Darken,
        Blend::Lighten => BlendMode::Lighten,
        Blend::ColorDodge => BlendMode::ColorDodge,
        Blend::ColorBurn => BlendMode::ColorBurn,
        Blend::HardLight => BlendMode::HardLight,
        Blend::SoftLight => BlendMode::SoftLight,
        Blend::Difference => BlendMode::Difference,
        Blend::Exclusion => BlendMode::Exclusion,
        Blend::Hue => BlendMode::Hue,
        Blend::Saturation => BlendMode::Saturation,
        Blend::Color => BlendMode::Color,
        Blend::Luminosity => BlendMode::Luminosity,
    }
}

/// `m` as `tiny-skia`'s transform.
fn transform(matrix: Matrix) -> Transform {
    #[expect(clippy::cast_possible_truncation, reason = "pixel space")]
    let [sx, ky, kx, sy, tx, ty] = matrix.0.map(|v| v as f32);
    Transform::from_row(sx, ky, kx, sy, tx, ty)
}
