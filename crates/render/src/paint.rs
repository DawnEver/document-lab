//! What a color paints with (ISO 32000-2 section 8.7): one RGB color, or a pattern -- a shading pattern's colors
//! ([`crate::shade`], its `/Background` where the shading does not paint), or a tiling pattern's cell repeated.
//!
//! A tiling pattern's cell is drawn by the renderer itself, from the interpreter's events for it
//! ([`pdf::File::cell_drawing`]; an uncolored pattern's in the tint the color states), once, on the canvas's pixel
//! grid where the lattice's first copy lies; each copy meeting what is painted is that drawing moved by whole pixels,
//! as `MuPDF` moves a tile (D58's line, measured 2026-11-27): from the pixel the first copy's box starts on, by the
//! lattice's step through the pattern's matrix reckoned in single precision and truncated -- so where `MuPDF`'s
//! float sum falls a hair short of a pixel, a copy lands a pixel early, a seam it shows and so does this. A pattern
//! filling a rectangle square to the canvas fills the whole pixels it touches, as a clip of it does. Past [`BLITS`]
//! copies, one period of the lattice (`/XStep` by `/YStep`, each copy overlapping it drawn) is drawn into a tile at
//! about the canvas's resolution, its sides whole pixels, at most [`TILE_SIDE`], and repeated across through the
//! pattern's matrix, interpolated. `/TilingType` (how a device may adjust the spacing) is not applied: every type is
//! tiled at its stated steps.

use std::ops::RangeInclusive;
use std::rc::Rc;

use pdf::color::{Color, Pattern, PatternKind};
use pdf::geometry::{Matrix, Point, Rect};
use pdf::object::Object;
use tiny_skia::{
    BlendMode, FilterQuality, IntRect, Paint, Pixmap, PixmapPaint, SpreadMode, Transform,
};

use crate::{Canvas, Grid, transform};

/// The most copies of a tiling pattern's cell placed one by one over what it paints; past it, one period of the
/// lattice is repeated.
pub const BLITS: f64 = 262_144.0;
/// The most pixels a tiling pattern's cell is drawn over.
const TILE_PIXELS: f64 = 16_777_216.0;
/// The most pixels a side of a repeated tile has.
pub const TILE_SIDE: f64 = 2048.0;
/// The most copies of a cell drawn into one tile along each side (a cell many periods wide).
const COPIES: f64 = 16.0;
/// How deep patterns and soft masks may nest (a cell painting a pattern, a mask's group masked) before one is
/// refused as a loop.
pub(crate) const NESTING: usize = 8;

/// What a mark is painted with.
pub(crate) enum Source {
    /// One color, RGB in 0..=1.
    Solid([f64; 3]),
    /// Pixels: placed on the canvas by `transform`, beyond their edge padded or repeated.
    Pixels {
        pixmap: Pixmap,
        transform: Transform,
        spread: SpreadMode,
        quality: FilterQuality,
    },
}

impl Source {
    /// A paint of it at `alpha`, blended by `blend`, its edges anti-aliased.
    pub(crate) fn paint(&self, alpha: f64, blend: BlendMode) -> Paint<'_> {
        let mut paint = Paint {
            anti_alias: true,
            blend_mode: blend,
            ..Paint::default()
        };
        #[expect(clippy::cast_possible_truncation, reason = "an alpha")]
        let alpha = alpha.clamp(0.0, 1.0) as f32;
        match self {
            Self::Solid(rgb) => {
                #[expect(clippy::cast_possible_truncation, reason = "color components")]
                paint.set_color(
                    tiny_skia::Color::from_rgba(rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, alpha)
                        .unwrap_or(tiny_skia::Color::BLACK),
                );
            }
            Self::Pixels {
                pixmap,
                transform,
                spread,
                quality,
            } => {
                paint.shader =
                    tiny_skia::Pattern::new(pixmap.as_ref(), *spread, *quality, alpha, *transform);
            }
        }
        paint
    }

    /// Pixels covering `area` of the canvas, as they are.
    fn layer(pixmap: Pixmap, area: IntRect) -> Self {
        #[expect(clippy::cast_precision_loss, reason = "pixel offsets")]
        Self::Pixels {
            pixmap,
            transform: Transform::from_translate(area.x() as f32, area.y() as f32),
            spread: SpreadMode::Pad,
            quality: FilterQuality::Nearest,
        }
    }
}

impl Canvas<'_> {
    /// What `color` paints `what` (a fill, a stroke, ...) with over `reach` (page space: what is painted lies within
    /// it); None when nothing of it is drawn: off the canvas, or not drawable (counted).
    pub(crate) fn source(&mut self, color: &Color, what: &str, reach: Rect) -> Option<Source> {
        let Some(pattern) = &color.pattern else {
            if let Some(rgb) = color.rgb {
                return Some(Source::Solid(rgb));
            }
            self.undrawn(&format!(
                "{what} in a {} color with no RGB",
                String::from_utf8_lossy(&color.space)
            ));
            return None;
        };
        let area = self.grid.pixels(reach)?;
        let made = match pattern.kind {
            PatternKind::Shading => self.shading_pattern(pattern, area),
            PatternKind::Tiling { uncolored } => {
                self.tiling(pattern, uncolored.then_some(color), area)
            }
        };
        match made {
            Ok(source) => Some(source),
            Err(why) => {
                self.undrawn(&format!("{what} pattern not drawn: {why}"));
                None
            }
        }
    }

    /// A shading pattern over `area`: its shading through the pattern's matrix, its background where the shading
    /// does not paint.
    fn shading_pattern(&mut self, pattern: &Pattern, area: IntRect) -> Result<Source, String> {
        let dict = pattern.object.as_dict().ok_or("not a dictionary")?;
        let shading = self
            .file
            .get(dict, b"Shading")
            .map_err(|e| e.to_string())?
            .ok_or("no /Shading")?;
        if dict.get(b"ExtGState").is_some() {
            self.undrawn("shading pattern's /ExtGState (not applied)");
        }
        let to_pixels = pattern.matrix.then(self.grid.matrix());
        let pixmap = self.shaded(&shading, to_pixels, area, true, pattern.intent)?;
        Ok(Source::layer(pixmap, area))
    }

    /// A tiling pattern over `area`, its cell in `tint` when uncolored (see the module).
    fn tiling(
        &mut self,
        pattern: &Pattern,
        tint: Option<&Color>,
        area: IntRect,
    ) -> Result<Source, String> {
        let dict = pattern.object.as_dict().ok_or("not a stream")?;
        let number = |key: &[u8]| -> Result<f64, String> {
            self.file
                .get(dict, key)
                .map_err(|e| e.to_string())?
                .and_then(|o| o.as_number())
                .filter(|v| v.is_finite() && *v != 0.0)
                .ok_or_else(|| format!("no /{}", String::from_utf8_lossy(key)))
        };
        let (xs, ys) = (number(b"XStep")?.abs(), number(b"YStep")?.abs());
        let bbox = match self.file.get(dict, b"BBox").map_err(|e| e.to_string())? {
            Some(Object::Array(b)) => {
                let v: Vec<f64> = b.iter().filter_map(Object::as_number).collect();
                <[f64; 4]>::try_from(v).map_err(|_| "a /BBox not of four numbers")?
            }
            _ => return Err("no /BBox".to_string()),
        };
        let cell = Rect::spanning(
            Point {
                x: bbox[0],
                y: bbox[1],
            },
            Point {
                x: bbox[2],
                y: bbox[3],
            },
        );
        if tint.is_some_and(|t| t.rgb.is_none()) {
            return Err("an uncolored pattern's tint has no RGB".to_string());
        }
        let to_pixels = pattern.matrix.then(self.grid.matrix());
        let inverse = to_pixels
            .inverse()
            .ok_or("a pattern matrix that maps onto a line")?;
        let corners = [
            (area.left(), area.top()),
            (area.right(), area.top()),
            (area.right(), area.bottom()),
            (area.left(), area.bottom()),
        ]
        .map(|(x, y)| {
            inverse.apply(Point {
                x: f64::from(x),
                y: f64::from(y),
            })
        });
        let reach = Rect::around(&corners).expect("four corners");
        let across = ((reach.x0 - cell.x1) / xs).floor()..=((reach.x1 - cell.x0) / xs).ceil();
        let down = ((reach.y0 - cell.y1) / ys).floor()..=((reach.y1 - cell.y0) / ys).ceil();
        let count = (across.end() - across.start() + 1.0) * (down.end() - down.start() + 1.0);
        if count > BLITS {
            return self.repeated(pattern, tint, cell, (xs, ys), to_pixels);
        }
        let lattice = Lattice {
            cell,
            steps: (xs, ys),
            to_pixels,
            across,
            down,
        };
        self.blitted(pattern, tint, area, &lattice)
    }

    /// A tiling pattern's cell drawn once on the canvas's grid, over the part of its box any copy shows of `area`,
    /// then each copy of `lattice` placed (see the module).
    fn blitted(
        &mut self,
        pattern: &Pattern,
        tint: Option<&Color>,
        area: IntRect,
        lattice: &Lattice,
    ) -> Result<Source, String> {
        let Lattice {
            cell,
            to_pixels,
            across,
            down,
            ..
        } = lattice;
        let step = |i: f64, j: f64| lattice.step(i, j);
        let shifts = [
            step(*across.start(), *down.start()),
            step(*across.end(), *down.start()),
            step(*across.end(), *down.end()),
            step(*across.start(), *down.end()),
        ];
        let reach = Rect::around(&shifts).expect("four corners");
        let placed = Rect::around(&[
            to_pixels.apply(Point {
                x: cell.x0,
                y: cell.y0,
            }),
            to_pixels.apply(Point {
                x: cell.x1,
                y: cell.y0,
            }),
            to_pixels.apply(Point {
                x: cell.x1,
                y: cell.y1,
            }),
            to_pixels.apply(Point {
                x: cell.x0,
                y: cell.y1,
            }),
        ])
        .expect("four corners");
        let shown = Rect {
            x0: placed.x0.max(f64::from(area.left()) - reach.x1),
            y0: placed.y0.max(f64::from(area.top()) - reach.y1),
            x1: placed.x1.min(f64::from(area.right()) - reach.x0),
            y1: placed.y1.min(f64::from(area.bottom()) - reach.y0),
        };
        let mut layer = Pixmap::new(area.width(), area.height()).ok_or("an empty area")?;
        let (tx, ty) = (shown.x0.floor(), shown.y0.floor());
        let (tw, th) = (
            (shown.x1.ceil() - tx).max(0.0),
            (shown.y1.ceil() - ty).max(0.0),
        );
        if tw < 1.0 || th < 1.0 {
            return Ok(Source::layer(layer, area));
        }
        if tw * th > TILE_PIXELS {
            return Err("a tiling cell too large to draw".to_string());
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded"
        )]
        let (tile_x, tile_y, tile_w, tile_h) = (tx as i32, ty as i32, tw as u32, th as u32);
        let region = IntRect::from_xywh(tile_x, tile_y, tile_w, tile_h).ok_or("an empty tile")?;
        let grid = self.grid.sub(region);
        let blank = Pixmap::new(tile_w, tile_h).ok_or("an empty tile")?;
        let tile = self.cells(
            grid,
            blank,
            pattern,
            tint,
            |_, _| pattern.matrix,
            (0.0..=0.0, 0.0..=0.0),
        )?;
        let (left, top) = (placed.x0.floor(), placed.y0.floor());
        let mut j = *down.start();
        while j <= *down.end() {
            let mut i = *across.start();
            while i <= *across.end() {
                let (moved_x, moved_y) = lattice.moved(i, j, (left, top));
                let x = moved_x + tx - f64::from(area.left());
                let y = moved_y + ty - f64::from(area.top());
                let inside = x < f64::from(area.width())
                    && y < f64::from(area.height())
                    && x + tw > 0.0
                    && y + th > 0.0;
                if inside {
                    #[expect(clippy::cast_possible_truncation, reason = "within the area")]
                    layer.draw_pixmap(
                        x as i32,
                        y as i32,
                        tile.as_ref(),
                        &PixmapPaint::default(),
                        Transform::identity(),
                        None,
                    );
                }
                i += 1.0;
            }
            j += 1.0;
        }
        Ok(Source::layer(layer, area))
    }

    /// A tiling pattern of very many cells: one period of the lattice drawn into a tile of whole pixels, repeated
    /// across through the pattern's matrix.
    fn repeated(
        &mut self,
        pattern: &Pattern,
        tint: Option<&Color>,
        cell: Rect,
        (xs, ys): (f64, f64),
        to_pixels: Matrix,
    ) -> Result<Source, String> {
        let [xx, xy, yx, yy, ..] = to_pixels.0;
        let width = (xs * xx.hypot(xy)).round().clamp(1.0, TILE_SIDE);
        let height = (ys * yx.hypot(yy)).round().clamp(1.0, TILE_SIDE);
        let (kx, ky) = (width / xs, height / ys);
        let to_tile = Matrix([kx, 0.0, 0.0, ky, -cell.x0 * kx, -cell.y0 * ky]);
        // The copies of the cell meeting the period [x0, x0 + xs] x [y0, y0 + ys].
        let first = |extent: f64, step: f64| (-(extent / step).ceil() + 1.0).max(-COPIES);
        let across = first(cell.width(), xs)..=0.0;
        let down = first(cell.height(), ys)..=0.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded"
        )]
        let (w, h) = (width as u32, height as u32);
        let blank = Pixmap::new(w, h).ok_or("an empty tile")?;
        let tile = self.cells(
            Grid::tile(w, h),
            blank,
            pattern,
            tint,
            |i, j| Matrix([1.0, 0.0, 0.0, 1.0, i * xs, j * ys]).then(to_tile),
            (across, down),
        )?;
        let to_canvas = to_tile.inverse().expect("a scale").then(to_pixels);
        Ok(Source::Pixels {
            pixmap: tile,
            transform: transform(to_canvas),
            spread: SpreadMode::Repeat,
            quality: FilterQuality::Bilinear,
        })
    }

    /// The cells of `pattern` at each lattice step `(i, j)` of `steps`, each placed by `placed(i, j)` (from pattern
    /// space to `grid`'s page space), drawn onto `pixmap`.
    fn cells(
        &mut self,
        grid: Grid,
        pixmap: Pixmap,
        pattern: &Pattern,
        tint: Option<&Color>,
        placed: impl Fn(f64, f64) -> Matrix,
        (across, down): (std::ops::RangeInclusive<f64>, std::ops::RangeInclusive<f64>),
    ) -> Result<Pixmap, String> {
        if self.kept.nesting >= NESTING {
            return Err("patterns and soft masks nest too deep".to_string());
        }
        let object = Rc::clone(&pattern.object);
        self.kept.nesting += 1;
        let mut canvas = self.child(grid, pixmap);
        let mut failed = None;
        let mut j = *down.start();
        while j <= *down.end() && failed.is_none() {
            let mut i = *across.start();
            while i <= *across.end() {
                let drawn =
                    self.file
                        .cell_drawing(&object, placed(i, j), grid.page_box(), tint, &|m| {
                            grid.glyph_placed(m)
                        });
                match drawn {
                    Ok(content) => canvas.content(&content),
                    Err(e) => {
                        failed = Some(e.to_string());
                        break;
                    }
                }
                i += 1.0;
            }
            j += 1.0;
        }
        let pixmap = self.adopt(canvas);
        self.kept.nesting -= 1;
        failed.map_or(Ok(pixmap), Err)
    }
}

/// Where a tiling pattern's copies lie: its cell (pattern space), its steps, its matrix to the canvas's pixels, and
/// the copies (`i` across, `j` down) that meet what is painted.
struct Lattice {
    cell: Rect,
    steps: (f64, f64),
    to_pixels: Matrix,
    across: RangeInclusive<f64>,
    down: RangeInclusive<f64>,
}

impl Lattice {
    /// How far copy `(i, j)` lies from the first, in the canvas's pixels.
    fn step(&self, i: f64, j: f64) -> Point {
        let [xx, xy, yx, yy, ..] = self.to_pixels.0;
        let (xs, ys) = self.steps;
        Point {
            x: i * xx * xs + j * yx * ys,
            y: i * xy * xs + j * yy * ys,
        }
    }

    /// The whole pixels copy `(i, j)` lies from the first, as `MuPDF` moves a tile: from `corner` (the pixel the
    /// first copy's box starts on), by the step through the matrix reckoned in single precision, truncated.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "MuPDF's single precision and truncation, on purpose"
    )]
    fn moved(&self, i: f64, j: f64, corner: (f64, f64)) -> (f64, f64) {
        let [xx, xy, yx, yy] = [0, 1, 2, 3].map(|k| self.to_pixels.0[k] as f32);
        let (xs, ys) = (self.steps.0 as f32, self.steps.1 as f32);
        let (sx, sy) = (i as f32 * xs, j as f32 * ys);
        let (left, top) = (corner.0 as f32, corner.1 as f32);
        (
            f64::from((left + (sx * xx + sy * yx)) as i32) - corner.0,
            f64::from((top + (sx * xy + sy * yy)) as i32) - corner.1,
        )
    }
}
