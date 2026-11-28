//! document-lab's own renderer: what a page draws, from `pdf`'s content interpreter, as 8-bit pixels.
//!
//! [`render`] interprets a page for drawing ([`pdf::File::drawing`]) and paints its events in order onto a white (or
//! transparent) canvas: paths filled and stroked as the graphics state says (width, caps, joins, miter limit,
//! dashes, the nonzero or even-odd rule, a stroke under a matrix that scales unevenly stroked in user space), images
//! placed by their matrix in every color space the reader converts ([`image`]), text as its glyphs' outlines
//! ([`pdf::shape`]) placed by each glyph's text rendering matrix and filled, stroked or both as its render mode says
//! (a Type 3 glyph by its procedure's own events), each mark clipped by the chain of clipping paths in force
//! ([`clip`]: a text clip by its glyphs' outlines) and composited with its constant alpha, blend mode and soft mask,
//! anti-aliased by `tiny-skia`. A fill, stroke or glyph in a pattern paints the pattern (`paint`: a shading
//! pattern's colors, a tiling pattern's cell repeated), and `sh` paints a shading over the clip (`shade`). A
//! transparency group's marks are composited together, then their result onto what lies below, and a soft mask's
//! group is drawn to give it (`transparency`). The pixel grid is the page's: pixel `(i, j)` covers page space `[i, i + 1] x [j, j + 1]` divided by
//! the scale, and a region is drawn on the pixels it touches -- so a region's pixels are the whole page's there, but
//! for the anti-aliasing of a curve's edge, which differs by a few coverage steps where the region's edge cuts it.
//!
//! What is not drawn is never silently missing. What in the region the renderer cannot draw as the file states it
//! is counted by kind in [`Raster::undrawn`] -- a glyph whose shape cannot be read (a font that embeds no program,
//! with no substitute and no fallback to be had, by name), a text clip holding such a glyph (by its box), an image whose data does not
//! decode, or decodes only past what a strict decoder refuses ([`codec`]), a shading, pattern or soft mask that does
//! not read (by why), patterns and soft masks nested too deep, an operation the interpreter skipped (its warning, by
//! why: a mark not drawn, or a state the marks after it meeting the region are drawn in, `pdf::interpret::Reach`)
//! -- for the caller to refuse or accept. Hidden optional content and what the colorant `None` paints mark nothing and
//! are not drawn.
//!
//! Layers: [`codec`] (image codecs to samples) -> [`image`] (an image's samples to pixels), [`clip`] (clipping
//! paths to masks) and `shade` (a shading's colors to pixels) -> this module (the page painted), `paint` (what a
//! color paints with) and `transparency` (groups and soft masks). Apart from a page, [`specimen`] sets characters
//! of a font program on their own.

pub mod clip;
pub mod codec;
pub mod image;
mod paint;
mod shade;
pub mod specimen;
mod transparency;

use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use pdf::geometry::{Matrix, Point, Rect, Segment};
use pdf::interpret::{
    Blend, Clip, Content, Event, Group, ImageEvent, Layer, PathEvent, Reach, ShadingEvent,
    SoftMask, Stroke, TextEvent,
};
use pdf::object::Object;
use pdf::shape::Shape;
use pdf::{Error, File, Page, Result};
use tiny_skia::{
    BlendMode, FillRule, FilterQuality, IntRect, LineCap, LineJoin, Mask, Paint, PathBuilder,
    Pixmap, PixmapPaint, SpreadMode, StrokeDash, Transform,
};

use crate::clip::{Clips, Left};
use crate::paint::Source;

/// What to draw and how.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// The part of the page drawn, in page space (points, y down, origin at the top-left as displayed); None: the
    /// whole page.
    pub region: Option<Rect>,
    /// Pixels per point: the resolution over 72 dpi.
    pub scale: f64,
    /// Whether text is drawn: false leaves every glyph out (a page's line art and images alone).
    pub text: bool,
    /// Whether each annotation's appearance is drawn after the page's content.
    pub annotations: bool,
    /// Whether what nothing covers stays transparent (RGBA out) rather than white (RGB out).
    pub alpha: bool,
    /// How a JPEG 2000 image's dictionary is read where ISO 32000 and `MuPDF` part (the standard's by default).
    pub jpx: image::Jpx,
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
/// Refuses a scale that is not positive and finite, a region outside the page or of more than 2^28 pixels, and what
/// [`File::drawing`] refuses.
pub fn render(file: &File, page: &Page, options: &Options) -> Result<Raster> {
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
    let grid = Grid::of(region, options.scale, page.to_page_space())
        .ok_or_else(|| Error::invalid("the region holds no pixel of the page"))?;
    // Without text, no Type 3 glyph's procedure is run either.
    let content = if options.text {
        file.drawing(page, options.annotations, &|m| grid.glyph_placed(m))?
    } else {
        file.interpret(page, options.annotations)?
    };
    let mut pixmap = Pixmap::new(grid.width, grid.height)
        .ok_or_else(|| Error::invalid("the region is too large to draw"))?;
    if !options.alpha {
        pixmap.fill(tiny_skia::Color::WHITE);
    }
    let mut canvas = Canvas {
        file,
        jpx: options.jpx,
        pixmap,
        grid,
        clips: Clips::default(),
        masks: HashMap::new(),
        knockout: None,
        kept: Kept::default(),
    };
    canvas.content(&content);
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
    /// The page's matrix from user space to page space.
    user: Matrix,
    /// The device pixels (the page's) what is drawn here may paint, as `MuPDF` bounds them -- the page's, a group's
    /// or a tile's canvas; a knockout element's window keeps its group's: where its image painter starts stepping
    /// ([`image::sampled`]).
    pub(crate) scissor: Rect,
}

impl Grid {
    /// The whole pixels `region` touches at `scale` (an edge within a thousandth of a pixel of a pixel's edge is on
    /// it, so a region drawn at whole pixels gets exactly those), or None when it touches none; `user` is the page's
    /// matrix from user space to page space.
    fn of(region: Rect, scale: f64, user: Matrix) -> Option<Self> {
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
            user,
            scissor: Rect {
                x0: left,
                y0: top,
                x1: right,
                y1: bottom,
            },
        })
    }

    /// A canvas of `width` x `height` pixels whose page space is its pixel space: a tiling pattern's tile.
    fn tile(width: u32, height: u32) -> Self {
        Self {
            scale: 1.0,
            left: 0.0,
            top: 0.0,
            width,
            height,
            user: Matrix::IDENTITY,
            scissor: Rect {
                x0: 0.0,
                y0: 0.0,
                x1: f64::from(width),
                y1: f64::from(height),
            },
        }
    }

    /// The grid of the pixels `area` of this canvas, a canvas of its own (a group's, a tile's): the same page space,
    /// on fewer pixels.
    fn sub(&self, area: IntRect) -> Self {
        let window = self.window(area);
        Self {
            scissor: window.device_box(),
            ..window
        }
    }

    /// The grid of the pixels `area` of this canvas drawn apart and put back (a knockout element's): as [`Self::sub`],
    /// but bounded as this canvas is.
    pub(crate) fn window(&self, area: IntRect) -> Self {
        Self {
            left: self.left + f64::from(area.x()),
            top: self.top + f64::from(area.y()),
            width: area.width(),
            height: area.height(),
            ..*self
        }
    }

    /// The device pixels (the page's) the canvas covers.
    pub(crate) fn device_box(&self) -> Rect {
        Rect {
            x0: self.left,
            y0: self.top,
            x1: self.left + f64::from(self.width),
            y1: self.top + f64::from(self.height),
        }
    }

    /// The page space the canvas's pixels cover.
    fn page_box(&self) -> Rect {
        Rect {
            x0: self.left / self.scale,
            y0: self.top / self.scale,
            x1: (self.left + f64::from(self.width)) / self.scale,
            y1: (self.top + f64::from(self.height)) / self.scale,
        }
    }

    /// The whole pixels of the canvas `r` (page space) touches; None when it touches none.
    fn pixels(&self, r: Rect) -> Option<IntRect> {
        let b = self.rect(r).intersect(self.bounds());
        let (x0, y0) = (b.x0.floor(), b.y0.floor());
        let (x1, y1) = (b.x1.ceil(), b.y1.ceil());
        if !(x0 < x1 && y0 < y1) {
            return None;
        }
        #[expect(clippy::cast_possible_truncation, reason = "within the canvas")]
        IntRect::from_ltrb(x0 as i32, y0 as i32, x1 as i32, y1 as i32)
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

    /// A glyph's text rendering `matrix` (to page space) with its origin moved onto the pixel grid `MuPDF` places
    /// glyphs on (its glyph cache's, measured 2026-11-27 by moving a glyph in sixteenths of a pixel): across, to the
    /// nearest quarter of a pixel for a glyph under 24 pixels an em, half for one under 48, whole from 48; down, to
    /// the nearest whole pixel.
    pub(crate) fn glyph_placed(&self, matrix: Matrix) -> Matrix {
        let [xx, xy, yx, yy, tx, ty] = matrix.0;
        let size = (xx * yy - xy * yx).abs().sqrt() * self.scale;
        let steps = if size >= 48.0 {
            1.0
        } else if size >= 24.0 {
            2.0
        } else {
            4.0
        };
        let snap = |v: f64, steps: f64| (v * steps + 0.5).floor() / steps;
        let (across, down) = self.device(Point { x: tx, y: ty });
        let (dx, dy) = (snap(across, steps) - across, snap(down, 1.0) - down);
        Matrix([xx, xy, yx, yy, tx + dx / self.scale, ty + dy / self.scale])
    }

    /// Where `MuPDF` reckons the page-space point `p` falls on the canvas: from user space through the page's matrix
    /// scaled, in single precision as it computes -- an origin on a half pixel (a baseline at 90 points, 150 dpi:
    /// 30 points up a 120-point page) rounds as its float sum falls, not as the exact one would.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "MuPDF's single precision, on purpose"
    )]
    fn device(&self, point: Point) -> (f64, f64) {
        let [xx, xy, yx, yy, tx, ty] = self.user.0;
        // The page's matrix is a quarter turn with a flip: its inverse is exact.
        let det = xx * yy - xy * yx;
        let (px, py) = (point.x - tx, point.y - ty);
        let (ux, uy) = ((yy * px - yx * py) / det, (xx * py - xy * px) / det);
        let scale = self.scale as f32;
        let scaled = self.user.0.map(|v| v as f32 * scale);
        let (ux, uy) = (ux as f32, uy as f32);
        (
            f64::from(ux * scaled[0] + uy * scaled[2] + scaled[4]) - self.left,
            f64::from(ux * scaled[1] + uy * scaled[3] + scaled[5]) - self.top,
        )
    }

    /// `r` (page space) in pixel space.
    pub(crate) fn rect(&self, r: Rect) -> Rect {
        Rect::spanning(
            self.point(Point { x: r.x0, y: r.y0 }),
            self.point(Point { x: r.x1, y: r.y1 }),
        )
    }
}

/// A page being painted -- or a transparency group, a soft mask's group, a tiling pattern's tile, each on a canvas
/// of its own.
struct Canvas<'a> {
    file: &'a File,
    /// How a JPEG 2000 image's dictionary is read.
    jpx: image::Jpx,
    pixmap: Pixmap,
    grid: Grid,
    clips: Clips,
    /// Each soft mask made on this canvas, by what names it; None: it could not be made (counted).
    masks: HashMap<transparency::MaskKey, Option<Rc<Mask>>>,
    /// In a knockout group: the group's backdrop, which each of its marks is composited onto.
    knockout: Option<Rc<Pixmap>>,
    kept: Kept,
}

/// What every canvas of a page shares: what was read once, and what was not drawn.
#[derive(Default)]
struct Kept {
    /// Each image `XObject` decoded so far (not a stencil mask: its pixels are the fill color's), by its number.
    images: HashMap<u32, Rc<std::result::Result<image::Decoded, String>>>,
    undrawn: BTreeMap<String, usize>,
    /// How deep the canvas being drawn lies in patterns' cells and soft masks' groups.
    nesting: usize,
}

impl<'a> Canvas<'a> {
    /// A canvas of `pixmap` on `grid`, sharing what this one keeps until [`Self::adopt`] hands it back.
    fn child(&mut self, grid: Grid, pixmap: Pixmap) -> Canvas<'a> {
        Canvas {
            file: self.file,
            jpx: self.jpx,
            pixmap,
            grid,
            clips: Clips::default(),
            masks: HashMap::new(),
            knockout: None,
            kept: std::mem::take(&mut self.kept),
        }
    }

    /// `child`'s pixels, what it kept handed back.
    fn adopt(&mut self, child: Canvas<'a>) -> Pixmap {
        self.kept = child.kept;
        child.pixmap
    }

    /// What `content` draws, drawn; each of its warnings that may change what is drawn here counted in undrawn, by
    /// why: one whose mark is not drawn, unless its box is known and misses the canvas; one whose state the marks
    /// after it are drawn in, when one of those may paint here.
    fn content(&mut self, content: &Content) {
        self.draw(&content.events, None);
        // The last event that may paint here: sought once, when a warning reaches the events after it.
        let mut last: Option<Option<usize>> = None;
        for warning in &content.warnings {
            let meets = match warning.reach {
                Reach::Unseen => false,
                Reach::Anywhere => true,
                Reach::Within(r) => self.meets(r),
                Reach::From(n) => last
                    .get_or_insert_with(|| {
                        content
                            .events
                            .iter()
                            .rposition(|e| self.extent(e).is_some_and(|r| self.meets(r)))
                    })
                    .is_some_and(|last| n <= last),
            };
            if meets {
                self.undrawn(&warning.why);
            }
        }
    }

    /// `events` drawn in order; a run of them in a transparency group directly within `within` (None: the page)
    /// drawn as that group.
    fn draw(&mut self, events: &[Event], within: Option<&Rc<Group>>) {
        let mut i = 0;
        while i < events.len() {
            let event = &events[i];
            let Some(group) = transparency::child_of(event.compositing().group.as_ref(), within)
            else {
                let alpha = transparency::alpha_of(event);
                // What it may paint bounds it in a knockout group alone.
                let extent = self.knockout.as_ref().and_then(|_| self.extent(event));
                self.element(alpha, extent, &|c: &mut Canvas<'a>| c.event(event));
                i += 1;
                continue;
            };
            let run = events[i..]
                .iter()
                .take_while(|e| transparency::holds(e.compositing().group.as_ref(), &group))
                .count();
            let marks = &events[i..i + run];
            self.element(
                group.compositing.fill_alpha,
                Some(group.bbox),
                &|c: &mut Canvas<'a>| c.group(&group, marks),
            );
            i += run;
        }
    }

    fn event(&mut self, event: &Event) {
        match event {
            Event::Path(path) if shows(&path.layer) => self.path(path),
            Event::Image(image) if shows(&image.layer) => self.image(image),
            Event::Shading(shading) if shows(&shading.layer) => self.shading(shading),
            Event::Text(text) if shows(&text.layer) => self.text(text),
            _ => {} // hidden or inkless
        }
    }

    /// The box (page space) of what `event` may paint, as far as anti-aliasing reaches aside -- a glyph whose shape is
    /// missing by its box, where it is counted; None when it paints nothing (hidden, inkless, a Type 3 glyph's text,
    /// whose procedure paints).
    fn extent(&self, event: &Event) -> Option<Rect> {
        match event {
            Event::Path(path) if shows(&path.layer) => {
                let reach = path.stroke.as_ref().map_or(0.0, |s| self.reach(s));
                bounds(&path.segments).map(|b| b.grown(reach))
            }
            Event::Image(image) if shows(&image.layer) => Some(image.bbox),
            Event::Shading(shading) if shows(&shading.layer) => Some(shading.bbox),
            Event::Text(text) if shows(&text.layer) => {
                let points: Vec<Point> = text
                    .glyphs
                    .iter()
                    .filter_map(|glyph| match text.font.shape(glyph.code) {
                        Shape::Outline(outline) => {
                            let matrix = self.grid.glyph_placed(glyph.matrix);
                            Some(
                                points(&outline)
                                    .into_iter()
                                    .map(|p| matrix.apply(p))
                                    .collect(),
                            )
                        }
                        Shape::Missing(_) => Some(vec![
                            Point {
                                x: glyph.bbox.x0,
                                y: glyph.bbox.y0,
                            },
                            Point {
                                x: glyph.bbox.x1,
                                y: glyph.bbox.y1,
                            },
                        ]),
                        Shape::Procedure => None,
                    })
                    .flatten()
                    .collect();
                let strokes = matches!(text.render_mode, 1 | 2 | 5 | 6);
                let reach = if strokes {
                    self.reach(&text.stroke)
                } else {
                    0.0
                };
                Rect::around(&points).map(|b| b.grown(reach))
            }
            _ => None,
        }
    }

    /// `what` meeting the region was not drawn as the file states it.
    fn undrawn(&mut self, what: &str) {
        *self.kept.undrawn.entry(what.to_string()).or_default() += 1;
    }

    /// Whether `bbox` (page space) meets the canvas.
    fn meets(&self, bbox: Rect) -> bool {
        self.grid.rect(bbox).meets(self.grid.bounds())
    }

    /// How far past its path (page space) `stroke` may paint pixels: as far as its caps and joins reach
    /// ([`Stroke::reach`]), and a pixel more -- a line under a tenth of a pixel is drawn one pixel wide, and
    /// anti-aliasing shades a pixel the paint only nears.
    fn reach(&self, stroke: &Stroke) -> f64 {
        stroke.reach() + 1.0 / self.grid.scale
    }

    /// What of the canvas a mark shows on: what the chain of clips `clip` leaves, times the soft mask `soft` (one
    /// that cannot be made is counted, and the mark drawn without it).
    fn coverage(&mut self, clip: Option<&Rc<Clip>>, soft: Option<&Rc<SoftMask>>) -> Left {
        let left = self.clips.mask(clip, &self.grid, &mut self.kept.undrawn);
        let Some(soft) = soft else {
            return left;
        };
        let Some(mask) = self.soft_mask(soft) else {
            return left;
        };
        match left {
            Left::Nothing => Left::Nothing,
            Left::All => Left::Mask(mask),
            Left::Mask(clip) => Left::Mask(Rc::new(transparency::product(&clip, &mask))),
        }
    }

    fn path(&mut self, path: &PathEvent) {
        let Some(bounds) = bounds(&path.segments) else {
            return;
        };
        let reach = path.stroke.as_ref().map_or(0.0, |s| self.reach(s));
        let reach_box = bounds.grown(reach);
        if !self.meets(reach_box) {
            return;
        }
        let compositing = &path.compositing;
        let mask = match self.coverage(path.clip.as_ref(), compositing.soft_mask.as_ref()) {
            Left::Nothing => return, // clipped away whole
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let blend = blend_mode(compositing.blend);
        if let Some(fill) = &path.fill
            && let Some(source) = self.source(&fill.color, "fill", bounds)
        {
            let mut paint = source.paint(compositing.fill_alpha, blend);
            // A pattern fills its path as a clip it is painted through: a rectangle square to the canvas on the
            // whole pixels it touches, as a clip of one is.
            let pixels = matches!(source, Source::Pixels { .. })
                .then(|| clip::whole_pixels(&path.segments, &self.grid))
                .flatten();
            #[expect(clippy::cast_possible_truncation, reason = "pixel coordinates")]
            if let Some(r) = pixels
                && let Some(r) =
                    tiny_skia::Rect::from_ltrb(r.x0 as f32, r.y0 as f32, r.x1 as f32, r.y1 as f32)
            {
                paint.anti_alias = false;
                self.pixmap
                    .fill_rect(r, &paint, Transform::identity(), mask.as_deref());
                return self.stroked(path, reach_box, blend, mask.as_deref());
            }
            let Some(outline) = build(&path.segments, &|p| self.grid.point(p)) else {
                return;
            };
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
        self.stroked(path, reach_box, blend, mask.as_deref());
    }

    /// `path`'s stroke, when it has one, drawn within `reach_box` (page space).
    fn stroked(
        &mut self,
        path: &PathEvent,
        reach_box: Rect,
        blend: BlendMode,
        mask: Option<&Mask>,
    ) {
        if let Some(stroke) = &path.stroke
            && let Some(source) = self.source(&stroke.color, "stroke", reach_box)
        {
            let paint = source.paint(path.compositing.stroke_alpha, blend);
            self.stroke(&path.segments, stroke, &paint, mask);
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
        let compositing = &event.compositing;
        let mask = match self.coverage(event.clip.as_ref(), compositing.soft_mask.as_ref()) {
            Left::Nothing => return, // clipped away whole
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let decoded = if let Some(known) = event.object.and_then(|n| self.kept.images.get(&n)) {
            Rc::clone(known)
        } else {
            let decoded = Rc::new(image::decode(self.file, event, self.jpx));
            if let (Some(n), Ok(d)) = (event.object, decoded.as_ref())
                && !d.stencil
            {
                self.kept.images.insert(n, Rc::clone(&decoded));
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
        self.placed(event, decoded, &pixels, mask.as_deref());
    }

    /// The image `event` places, `decoded`, drawn as `pixels` says through `mask` (its clip and soft mask): square to
    /// the page, sampled -- nearest or interpolated -- as `MuPDF` samples it ([`image::sampled`]), else interpolated
    /// by `tiny-skia`; a stencil mask painting a
    /// pattern as its coverage, the pattern painted through it.
    fn placed(
        &mut self,
        event: &ImageEvent,
        decoded: &image::Decoded,
        pixels: &image::Pixels,
        mask: Option<&Mask>,
    ) {
        let Some(pixmap) = pixels.pixmap(decoded) else {
            return;
        };
        let compositing = &event.compositing;
        let blend = blend_mode(compositing.blend);
        #[expect(clippy::cast_possible_truncation, reason = "an alpha")]
        let opacity = compositing.fill_alpha as f32;
        // Square to the page, its pixels sampled as MuPDF samples them; turned or skewed, interpolated by tiny-skia.
        let nearest = if pixels.smooth && !pixels.grid_fitted {
            None
        } else {
            let scissor = clip::scissor(event.clip.as_ref(), &self.grid);
            let Some(nearest) =
                image::sampled(pixmap, pixels.unit, &self.grid, scissor, pixels.smooth)
            else {
                return;
            };
            Some(nearest)
        };
        let mut paint = Paint {
            blend_mode: blend,
            anti_alias: !pixels.grid_fitted,
            ..Paint::default()
        };
        paint.shader = tiny_skia::Pattern::new(
            pixmap,
            SpreadMode::Pad,
            FilterQuality::Bilinear,
            opacity,
            Transform::identity(),
        );
        #[expect(clippy::cast_precision_loss, reason = "a pixel count")]
        let Some(rect) =
            tiny_skia::Rect::from_xywh(0.0, 0.0, pixmap.width() as f32, pixmap.height() as f32)
        else {
            return;
        };
        // What it covers drawn onto `target` with `paint`'s opacity and blend mode, through `mask`.
        let onto = |target: &mut Pixmap, paint: &Paint, mask: Option<&Mask>| match &nearest {
            Some((sampled, (x, y))) => target.draw_pixmap(
                *x,
                *y,
                sampled.as_ref(),
                &PixmapPaint {
                    opacity,
                    blend_mode: paint.blend_mode,
                    quality: FilterQuality::Nearest,
                },
                Transform::identity(),
                mask,
            ),
            None => target.fill_rect(rect, paint, transform(pixels.matrix), mask),
        };
        if decoded.stencil && event.fill.pattern.is_some() {
            // A stencil painting a pattern: its coverage, then the pattern through it.
            let Some(mut shape) = Pixmap::new(self.grid.width, self.grid.height) else {
                return;
            };
            let shaping = Paint {
                blend_mode: BlendMode::SourceOver,
                ..paint.clone()
            };
            onto(&mut shape, &shaping, None);
            let mut cover = Mask::from_pixmap(shape.as_ref(), tiny_skia::MaskType::Alpha);
            if let Some(clip) = mask {
                cover = transparency::product(&cover, clip);
            }
            let Some(source) = self.source(&event.fill, "stencil mask", event.bbox) else {
                return;
            };
            // The constant alpha is in the coverage already.
            let paint = source.paint(1.0, blend);
            #[expect(clippy::cast_precision_loss, reason = "a pixel count")]
            let canvas = tiny_skia::Rect::from_xywh(
                0.0,
                0.0,
                self.grid.width as f32,
                self.grid.height as f32,
            );
            if let Some(canvas) = canvas {
                self.pixmap
                    .fill_rect(canvas, &paint, Transform::identity(), Some(&cover));
            }
            return;
        }
        onto(&mut self.pixmap, &paint, mask);
    }

    /// The glyphs `text` shows, as its render mode says (9.3.6): 0 and 4 filled, 1 and 5 stroked, 2 and 6 both
    /// (the clip of modes 4-7 is the interpreter's), 3 and 7 not painted. Their outlines are filled together by the
    /// nonzero rule, or one by one when they are composited otherwise than opaquely over what lies below (so two
    /// that overlap show it). A Type 3 glyph is drawn by its procedure's own events; a glyph meeting the region
    /// whose shape is missing is counted.
    fn text(&mut self, text: &TextEvent) {
        let (fills, strokes) = match text.render_mode {
            0 | 4 => (true, false),
            1 | 5 => (false, true),
            2 | 6 => (true, true),
            _ => return,
        };
        let mut glyphs: Vec<Vec<Segment>> = Vec::with_capacity(text.glyphs.len());
        for glyph in &text.glyphs {
            match text.font.shape(glyph.code) {
                Shape::Outline(outline) if !outline.is_empty() => {
                    let matrix = self.grid.glyph_placed(glyph.matrix);
                    glyphs.push(
                        outline
                            .iter()
                            .map(|s| s.mapped(|p| matrix.apply(p)))
                            .collect(),
                    );
                }
                Shape::Missing(why) if self.meets(glyph.bbox) => self.undrawn(&why),
                _ => {} // nothing to draw: a space, a Type 3 glyph (its procedure draws it), one off the region
            }
        }
        let opaque = text.compositing.blend == Blend::Normal
            && (!fills || text.compositing.fill_alpha >= 1.0)
            && (!strokes || text.compositing.stroke_alpha >= 1.0);
        let groups: Vec<Vec<Segment>> = if opaque {
            vec![glyphs.concat()]
        } else {
            glyphs
        };
        for segments in &groups {
            self.glyphs(segments, text, fills, strokes);
        }
    }

    /// Glyph outlines `segments` (page space) painted as `text` says: filled when `fills`, stroked when `strokes`.
    fn glyphs(&mut self, segments: &[Segment], text: &TextEvent, fills: bool, strokes: bool) {
        let Some(around) = bounds(segments) else {
            return;
        };
        let reach = if strokes {
            self.reach(&text.stroke)
        } else {
            0.0
        };
        let reach_box = around.grown(reach);
        if !self.meets(reach_box) {
            return;
        }
        let compositing = &text.compositing;
        let mask = match self.coverage(text.clip.as_ref(), compositing.soft_mask.as_ref()) {
            Left::Nothing => return,
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let blend = blend_mode(compositing.blend);
        if fills && let Some(source) = self.source(&text.fill, "text fill", around) {
            let Some(outline) = build(segments, &|p| self.grid.point(p)) else {
                return;
            };
            let paint = source.paint(compositing.fill_alpha, blend);
            self.pixmap.fill_path(
                &outline,
                &paint,
                FillRule::Winding,
                Transform::identity(),
                mask.as_deref(),
            );
        }
        if strokes && let Some(source) = self.source(&text.stroke.color, "text stroke", reach_box) {
            let paint = source.paint(compositing.stroke_alpha, blend);
            self.stroke(segments, &text.stroke, &paint, mask.as_deref());
        }
    }

    /// `sh`: the shading painted over the clip, within its `/BBox` (its `/Background` ignored, as 8.7.4.5.1 says).
    fn shading(&mut self, event: &ShadingEvent) {
        if !self.meets(event.bbox) {
            return;
        }
        let compositing = &event.compositing;
        let mask = match self.coverage(event.clip.as_ref(), compositing.soft_mask.as_ref()) {
            Left::Nothing => return,
            Left::All => None,
            Left::Mask(mask) => Some(mask),
        };
        let Some(area) = self.grid.pixels(event.bbox) else {
            return;
        };
        let to_pixels = event.matrix.then(self.grid.matrix());
        let pixels = match self.shaded(&event.shading, to_pixels, area, false, event.intent) {
            Ok(pixels) => pixels,
            Err(why) => {
                self.undrawn(&format!("shading not drawn: {why}"));
                return;
            }
        };
        #[expect(clippy::cast_possible_truncation, reason = "an alpha")]
        let paint = PixmapPaint {
            opacity: compositing.fill_alpha as f32,
            blend_mode: blend_mode(compositing.blend),
            quality: FilterQuality::Nearest,
        };
        self.pixmap.draw_pixmap(
            area.x(),
            area.y(),
            pixels.as_ref(),
            &paint,
            Transform::identity(),
            mask.as_deref(),
        );
    }

    /// The shading `object` sampled over `area` through `to_pixels` (see [`shade`]).
    fn shaded(
        &self,
        object: &Object,
        to_pixels: Matrix,
        area: IntRect,
        background: bool,
        intent: pdf::icc::Intent,
    ) -> std::result::Result<Pixmap, String> {
        let shading = self.file.shading(object).map_err(|e| e.to_string())?;
        shade::shade(&shading, to_pixels, area, background, intent)
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
            undrawn: self.kept.undrawn,
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
    Rect::around(&points(segments))
}

/// `segments`' points, their curves' control points too.
pub(crate) fn points(segments: &[Segment]) -> Vec<Point> {
    segments
        .iter()
        .flat_map(|s| match *s {
            Segment::MoveTo(p) | Segment::LineTo(p) => vec![p],
            Segment::CurveTo(a, b, p) => vec![a, b, p],
            Segment::Close => vec![],
        })
        .collect()
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
