//! A shading ([`pdf::shading`]) sampled into pixels: each pixel's color where its sample point falls in the
//! shading's space (ISO 32000-2 section 8.7.4.5).
//!
//! A pixel is sampled at [`SAMPLE`], where `MuPDF` samples it, and painted when the shading paints there: edges are
//! not anti-aliased, as `MuPDF` paints a shading. Axial and radial shadings take `t` there (a radial one the larger
//! `s` whose circle passes through the point, its radius not negative) and look its color up in a table of the
//! function sampled at one entry a pixel along the gradient (256 at the least), interpolated between entries; past
//! an end it is extended or left transparent as `/Extend` says. A function-based shading evaluates its function at
//! the sample point, inside its domain. A mesh's triangles, and a patch mesh's patches cut into triangles a few
//! pixels across (each patch's corner values interpolated over its parameter square), paint the pixels whose sample
//! points they hold, their values interpolated linearly -- a component's or `t`'s, `t` then through the function --
//! and converted to sRGB. What the shading does not paint stays transparent, or, for a shading pattern, its
//! `/Background`; what lies outside its `/BBox` stays transparent.

use pdf::geometry::{Matrix, Point, Rect};
use pdf::icc::Intent;
use pdf::shading::{Kind, Patch, Shading, Vertex};
use tiny_skia::{IntRect, Pixmap};

/// The most triangles a patch is cut into along each side.
const PATCH_STEPS: usize = 64;
/// The longest a patch's triangles may be across, in pixels, before it is cut finer.
const PATCH_PIXELS: f64 = 4.0;
/// Where in its pixel a shading is sampled, from the pixel's top-left corner: one pixel right, on its top edge.
/// `MuPDF` paints a mesh's triangle (and an axial, radial or function-based shading, which it draws as one) on
/// the pixels whose point there the triangle holds -- measured 2026-11-27 on triangles of every orientation, which
/// it matches pixel for pixel (D58's line: where the pixels of a mark fall).
const SAMPLE: (f64, f64) = (1.0, 0.0);

/// What converts a shading's values to sRGB.
type Rgb<'a> = &'a dyn Fn(&[f64]) -> Result<[f64; 3], String>;
/// A triangle's corner in pixels, with its values.
type Corner = (Point, Vec<f64>);

/// `shading` sampled over `area` (pixels of the canvas `to_pixels` maps the shading's space onto), its colors at
/// `intent`; `background` paints what it does not (a shading pattern's).
///
/// # Errors
///
/// Refuses a shading whose space gives no RGB for its colors, a function that yields none, and a space that maps
/// onto a line.
pub(crate) fn shade(
    shading: &Shading,
    to_pixels: Matrix,
    area: IntRect,
    background: bool,
    intent: Intent,
) -> Result<Pixmap, String> {
    let mut out = Pixmap::new(area.width(), area.height()).ok_or("an empty area")?;
    let inverse = to_pixels
        .inverse()
        .ok_or("a shading under a matrix that maps onto a line")?;
    let rgb = |values: &[f64]| -> Result<[f64; 3], String> {
        shading
            .rgb(values, intent)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| {
                format!(
                    "shading in a {} color with no RGB",
                    String::from_utf8_lossy(&shading.space.family())
                )
            })
    };
    if background && let Some(components) = &shading.background {
        let [r, g, b] = shading
            .space
            .color(components.clone(), intent)
            .map_err(|e| e.to_string())?
            .rgb
            .ok_or("a shading pattern's background with no RGB")?;
        #[expect(clippy::cast_possible_truncation, reason = "color components")]
        out.fill(
            tiny_skia::Color::from_rgba(r as f32, g as f32, b as f32, 1.0)
                .unwrap_or(tiny_skia::Color::BLACK),
        );
    }
    let mut canvas = Samples {
        out,
        area,
        inverse,
        bbox: shading.bbox,
    };
    match &shading.kind {
        Kind::Triangles(_) | Kind::Patches(_) => mesh(&mut canvas, shading, to_pixels, &rgb)?,
        _ => gradient(&mut canvas, shading, to_pixels, &rgb)?,
    }
    Ok(canvas.out)
}

/// An axial, radial or function-based shading painted onto `canvas`.
fn gradient(
    canvas: &mut Samples,
    shading: &Shading,
    to_pixels: Matrix,
    rgb: Rgb,
) -> Result<(), String> {
    match &shading.kind {
        Kind::Axial {
            coords,
            domain,
            extend,
        } => {
            let [x0, y0, x1, y1] = *coords;
            let (dx, dy) = (x1 - x0, y1 - y0);
            let length = dx * dx + dy * dy;
            let table = Table::new(domain, length.sqrt() * to_pixels.scale(), rgb)?;
            canvas.each(|p| {
                if length == 0.0 {
                    return None;
                }
                let s = ((p.x - x0) * dx + (p.y - y0) * dy) / length;
                extended(s, *extend).map(|s| table.at(s))
            });
        }
        Kind::Radial {
            coords,
            domain,
            extend,
        } => {
            let extent = (coords[2].abs().max(coords[5].abs())
                + (coords[3] - coords[0]).hypot(coords[4] - coords[1]))
                * to_pixels.scale();
            let table = Table::new(domain, extent, rgb)?;
            canvas.each(|p| radial(coords, *extend, p).map(|s| table.at(s)));
        }
        Kind::Function { domain, matrix } => {
            let to_domain = matrix
                .inverse()
                .ok_or("a function shading whose matrix maps onto a line")?;
            let [x0, x1, y0, y1] = *domain;
            let mut failed = None;
            canvas.each(|p| {
                let q = to_domain.apply(p);
                if !(x0.min(x1) <= q.x
                    && q.x <= x0.max(x1)
                    && y0.min(y1) <= q.y
                    && q.y <= y0.max(y1))
                {
                    return None;
                }
                match rgb(&[q.x, q.y]) {
                    Ok(c) => Some(c),
                    Err(e) => {
                        failed.get_or_insert(e);
                        None
                    }
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
        }
        Kind::Triangles(_) | Kind::Patches(_) => {}
    }
    Ok(())
}

/// A mesh's triangles, or its patches cut into triangles, painted onto `canvas`.
fn mesh(
    canvas: &mut Samples,
    shading: &Shading,
    to_pixels: Matrix,
    rgb: Rgb,
) -> Result<(), String> {
    match &shading.kind {
        Kind::Triangles(triangles) => {
            let lookup = Lookup::new(shading, triangles.iter().flatten(), rgb)?;
            for triangle in triangles {
                canvas.triangle(
                    triangle
                        .each_ref()
                        .map(|v| (to_pixels.apply(v.point), v.values.as_slice())),
                    &lookup,
                )?;
            }
        }
        Kind::Patches(patches) => {
            let corners: Vec<Vertex> = patches
                .iter()
                .flat_map(|p| {
                    p.values.iter().map(|v| Vertex {
                        point: Point::default(),
                        values: v.clone(),
                    })
                })
                .collect();
            let lookup = Lookup::new(shading, corners.iter(), rgb)?;
            for patch in patches {
                for triangle in cut(patch, to_pixels) {
                    canvas.triangle(
                        triangle.each_ref().map(|(p, v)| (*p, v.as_slice())),
                        &lookup,
                    )?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// `s` past the ends of `[0, 1]` held at the end extended there, else None.
fn extended(s: f64, extend: [bool; 2]) -> Option<f64> {
    if s < 0.0 {
        extend[0].then_some(0.0)
    } else if s > 1.0 {
        extend[1].then_some(1.0)
    } else {
        Some(s)
    }
}

/// The `s` of a radial shading at `p`: the largest whose circle passes through it with a radius not negative,
/// within `[0, 1]` or past an end that extends (8.7.4.5.4).
#[expect(
    clippy::many_single_char_names,
    reason = "the quadratic's a, b and k, as the derivation names them"
)]
fn radial(c: &[f64; 6], extend: [bool; 2], p: Point) -> Option<f64> {
    let [x0, y0, r0, x1, y1, r1] = *c;
    let (cx, cy, dr) = (x1 - x0, y1 - y0, r1 - r0);
    let (px, py) = (p.x - x0, p.y - y0);
    let a = cx * cx + cy * cy - dr * dr;
    let b = px * cx + py * cy + r0 * dr;
    let k = px * px + py * py - r0 * r0;
    let fits = |s: f64| {
        r0 + s * dr >= 0.0 && (0.0..=1.0).contains(&s)
            || s < 0.0 && extend[0] && r0 + s * dr >= 0.0
            || s > 1.0 && extend[1] && r0 + s * dr >= 0.0
    };
    let candidates = if a.abs() < 1e-12 {
        if b.abs() < 1e-12 {
            return None;
        }
        [k / (2.0 * b), f64::NAN]
    } else {
        let disc = b * b - a * k;
        if disc < 0.0 {
            return None;
        }
        let root = disc.sqrt();
        let (s1, s2) = ((b + root) / a, (b - root) / a);
        [s1.max(s2), s1.min(s2)]
    };
    candidates
        .into_iter()
        .find(|s| s.is_finite() && fits(*s))
        .map(|s| s.clamp(0.0, 1.0))
}

/// A function of `t` over a domain, sampled into a table of colors.
struct Table {
    colors: Vec<[f64; 3]>,
}

impl Table {
    /// The colors of `t` across `domain` at one entry a pixel of `pixels` (256 at the least, 4096 at the most).
    fn new(domain: &[f64; 2], pixels: f64, rgb: Rgb) -> Result<Self, String> {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "bounded"
        )]
        let n = (pixels.ceil() as usize).clamp(256, 4096);
        #[expect(clippy::cast_precision_loss, reason = "a small count")]
        let colors = (0..n)
            .map(|i| rgb(&[domain[0] + (domain[1] - domain[0]) * i as f64 / (n - 1) as f64]))
            .collect::<Result<_, _>>()?;
        Ok(Self { colors })
    }

    /// The color at `s` in `[0, 1]` across the domain, interpolated between entries.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "an index within the table"
    )]
    #[expect(clippy::many_single_char_names, reason = "an interpolation's terms")]
    fn at(&self, s: f64) -> [f64; 3] {
        let x = s.clamp(0.0, 1.0) * (self.colors.len() - 1) as f64;
        let i = (x.floor() as usize).min(self.colors.len() - 2);
        let f = x - i as f64;
        let (a, b) = (self.colors[i], self.colors[i + 1]);
        [0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * f)
    }
}

/// How a mesh's interpolated values become colors: through a table of `t` when a function gives the color, else
/// each converted.
enum Lookup<'a> {
    Table { table: Table, low: f64, high: f64 },
    Direct(Rgb<'a>),
}

impl<'a> Lookup<'a> {
    fn new<'v>(
        shading: &Shading,
        vertices: impl Iterator<Item = &'v Vertex>,
        rgb: Rgb<'a>,
    ) -> Result<Self, String> {
        if !shading.parametric() {
            return Ok(Self::Direct(rgb));
        }
        let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
        for v in vertices {
            low = low.min(v.values[0]);
            high = high.max(v.values[0]);
        }
        if low > high {
            (low, high) = (0.0, 1.0);
        }
        let table = Table::new(&[low, high], 1024.0, rgb)?;
        Ok(Self::Table { table, low, high })
    }

    fn color(&self, values: &[f64]) -> Result<[f64; 3], String> {
        match self {
            Self::Table { table, low, high } => {
                let span = high - low;
                Ok(table.at(if span > 0.0 {
                    (values[0] - low) / span
                } else {
                    0.0
                }))
            }
            Self::Direct(rgb) => rgb(values),
        }
    }
}

/// The pixels a shading is sampled into.
struct Samples {
    out: Pixmap,
    area: IntRect,
    /// From the canvas's pixels to the shading's space.
    inverse: Matrix,
    /// The shading's `/BBox`, in its space.
    bbox: Option<Rect>,
}

impl Samples {
    /// Each pixel whose center `color` gives a color for (in the shading's space), painted so.
    fn each(&mut self, mut color: impl FnMut(Point) -> Option<[f64; 3]>) {
        let (width, x0, y0) = (self.area.width(), self.area.x(), self.area.y());
        for (i, pixel) in self.out.pixels_mut().iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "a pixel index")]
            let (x, y) = (i as u32 % width, i as u32 / width);
            let center = Point {
                x: f64::from(x) + f64::from(x0) + SAMPLE.0,
                y: f64::from(y) + f64::from(y0) + SAMPLE.1,
            };
            let p = self.inverse.apply(center);
            if self
                .bbox
                .is_some_and(|b| !(b.x0 <= p.x && p.x <= b.x1 && b.y0 <= p.y && p.y <= b.y1))
            {
                continue;
            }
            if let Some(rgb) = color(p) {
                *pixel = opaque(rgb);
            }
        }
    }

    /// The pixels whose centers the triangle `v` (canvas pixels, each with its values) holds, its values
    /// interpolated and looked up.
    #[expect(
        clippy::many_single_char_names,
        reason = "the triangle's corners a, b and c, and the pixel's p"
    )]
    fn triangle(&mut self, v: [(Point, &[f64]); 3], lookup: &Lookup) -> Result<(), String> {
        /// How far outside an edge a sample may lie and be held: rounding's, not geometry's.
        const EDGE: f64 = -1e-9;
        let [(a, va), (b, vb), (c, vc)] = v;
        let area = (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
        if area.abs() < 1e-12 || !area.is_finite() {
            return Ok(());
        }
        let (x0, y0) = (f64::from(self.area.x()), f64::from(self.area.y()));
        let (w, h) = (f64::from(self.area.width()), f64::from(self.area.height()));
        let left = (a.x.min(b.x).min(c.x) - x0 - SAMPLE.0).ceil().max(0.0);
        let right = (a.x.max(b.x).max(c.x) - x0 - SAMPLE.0).floor().min(w - 1.0);
        let top = (a.y.min(b.y).min(c.y) - y0 - SAMPLE.1).ceil().max(0.0);
        let bottom = (a.y.max(b.y).max(c.y) - y0 - SAMPLE.1).floor().min(h - 1.0);
        if left > right || top > bottom {
            return Ok(());
        }
        let width = self.area.width();
        let mut values = vec![0.0; va.len()];
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "within the area"
        )]
        for y in top as u32..=bottom as u32 {
            for x in left as u32..=right as u32 {
                let p = Point {
                    x: f64::from(x) + x0 + SAMPLE.0,
                    y: f64::from(y) + y0 + SAMPLE.1,
                };
                let wa = ((b.x - p.x) * (c.y - p.y) - (c.x - p.x) * (b.y - p.y)) / area;
                let wb = ((c.x - p.x) * (a.y - p.y) - (a.x - p.x) * (c.y - p.y)) / area;
                let wc = 1.0 - wa - wb;
                if wa < EDGE || wb < EDGE || wc < EDGE {
                    continue;
                }
                if let Some(bbox) = self.bbox {
                    let q = self.inverse.apply(p);
                    if !(bbox.x0 <= q.x && q.x <= bbox.x1 && bbox.y0 <= q.y && q.y <= bbox.y1) {
                        continue;
                    }
                }
                for (k, value) in values.iter_mut().enumerate() {
                    *value = wa * va[k] + wb * vb[k] + wc * vc[k];
                }
                let rgb = lookup.color(&values)?;
                self.out.pixels_mut()[(y * width + x) as usize] = opaque(rgb);
            }
        }
        Ok(())
    }
}

/// `rgb` (0..=1) as an opaque pixel.
fn opaque(rgb: [f64; 3]) -> tiny_skia::PremultipliedColorU8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to a byte"
    )]
    let [r, g, b] = rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
    tiny_skia::PremultipliedColorU8::from_rgba(r, g, b, 255).expect("opaque")
}

/// `patch` (in the shading's space) cut into triangles a few pixels across under `to_pixels`, each corner with its
/// values interpolated over the patch's parameter square.
#[expect(
    clippy::many_single_char_names,
    reason = "the patch's control points p, parameters u and v, as 8.7.4.5.8 names them"
)]
fn cut(patch: &Patch, to_pixels: Matrix) -> Vec<[Corner; 3]> {
    let p = patch.points.map(|row| row.map(|q| to_pixels.apply(q)));
    // The control polygon's longest side bounds the patch's size.
    let mut longest: f64 = 0.0;
    for i in 0..4 {
        for j in 0..3 {
            longest = longest
                .max((p[i][j + 1].x - p[i][j].x).hypot(p[i][j + 1].y - p[i][j].y))
                .max((p[j + 1][i].x - p[j][i].x).hypot(p[j + 1][i].y - p[j][i].y));
        }
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "bounded"
    )]
    let steps = ((longest * 3.0 / PATCH_PIXELS).ceil() as usize).clamp(1, PATCH_STEPS);
    let bernstein = |t: f64| {
        let s = 1.0 - t;
        [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t]
    };
    let [c00, c01, c11, c10] = &patch.values;
    #[expect(clippy::cast_precision_loss, reason = "a small count")]
    let at = |i: usize, j: usize| {
        let (u, v) = (i as f64 / steps as f64, j as f64 / steps as f64);
        let (bu, bv) = (bernstein(u), bernstein(v));
        let mut point = Point::default();
        for (a, wu) in bu.iter().enumerate() {
            for (b, wv) in bv.iter().enumerate() {
                point.x += wu * wv * p[a][b].x;
                point.y += wu * wv * p[a][b].y;
            }
        }
        let values = (0..c00.len())
            .map(|k| {
                (1.0 - u) * (1.0 - v) * c00[k]
                    + (1.0 - u) * v * c01[k]
                    + u * v * c11[k]
                    + u * (1.0 - v) * c10[k]
            })
            .collect::<Vec<f64>>();
        (point, values)
    };
    let grid: Vec<Vec<(Point, Vec<f64>)>> = (0..=steps)
        .map(|i| (0..=steps).map(|j| at(i, j)).collect())
        .collect();
    let mut out = Vec::with_capacity(2 * steps * steps);
    for i in 0..steps {
        for j in 0..steps {
            let (a, b, c, d) = (
                &grid[i][j],
                &grid[i + 1][j],
                &grid[i + 1][j + 1],
                &grid[i][j + 1],
            );
            out.push([a.clone(), b.clone(), c.clone()]);
            out.push([a.clone(), c.clone(), d.clone()]);
        }
    }
    out
}
