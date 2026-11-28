//! Shadings (ISO 32000-2 section 8.7.4.5): a color for each point of an area, read from a shading dictionary (types
//! 1-3) or stream (the meshes, types 4-7) into what a renderer samples.
//!
//! - Type 1, function-based: a color for each point of a rectangle of the shading's own space (`/Domain`, mapped by
//!   `/Matrix`), from a function of two inputs.
//! - Types 2 and 3, axial and radial: a parameter `t` across `/Domain`, varying along an axis or between two
//!   circles, each end extended or not (`/Extend`), its color from a function of one input.
//! - Types 4 and 5, triangle meshes (free-form, each vertex with its flag; lattice, `/VerticesPerRow`), and 6 and 7,
//!   patch meshes (Coons, tensor-product): vertices and patches read from the stream's bits through `/Decode`, each
//!   vertex or corner carrying its color's components, or one `t` when a function gives the color. A Coons patch is
//!   read as the tensor-product patch with the same boundary (8.7.4.5.8's interior points), so one patch kind is
//!   sampled. Bits are read as they come, with no byte alignment between vertices or patches, as `MuPDF` reads them.
//!
//! Colors are the space's ([`crate::color`]): a function's outputs, else the components as read, through the
//! shading's `/ColorSpace`. The renderer samples the geometry; reading what the file states is this module's.

use crate::color::Space;
use crate::error::{Error, Result};
use crate::file::File;
use crate::function::Function;
use crate::geometry::{Matrix, Point, Rect};
use crate::icc::Intent;
use crate::object::{Dict, Object};

/// The most vertices a mesh may hold before it is refused: a bound on what one shading may cost.
pub const MAX_VERTICES: usize = 1 << 22;

/// A shading, read.
#[derive(Debug, Clone)]
pub struct Shading {
    /// Its color space (any but `Pattern`).
    pub space: Space,
    /// `/Background`: the components a shading pattern paints where the shading does not (`sh` ignores it).
    pub background: Option<Vec<f64>>,
    /// `/BBox`, in the shading's space: what it may paint, at most.
    pub bbox: Option<Rect>,
    /// Its geometry.
    pub kind: Kind,
    /// Its function: one of as many outputs as the space has components, or one per component; empty when the
    /// values are the components themselves.
    functions: Vec<Function>,
}

/// A shading's geometry.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Type 1: a function of the point, over `domain` (`[x0 x1 y0 y1]`) of the space `matrix` maps to the shading's.
    Function {
        /// `[x0 x1 y0 y1]`.
        domain: [f64; 4],
        /// From the domain's space to the shading's.
        matrix: Matrix,
    },
    /// Type 2: `t` along the axis from `(x0, y0)` to `(x1, y1)` (`coords`).
    Axial {
        /// `[x0 y0 x1 y1]`.
        coords: [f64; 4],
        /// `[t0 t1]`: `t` at each end.
        domain: [f64; 2],
        /// Whether it extends past each end.
        extend: [bool; 2],
    },
    /// Type 3: `t` between the circle `(x0, y0, r0)` and the circle `(x1, y1, r1)` (`coords`).
    Radial {
        /// `[x0 y0 r0 x1 y1 r1]`.
        coords: [f64; 6],
        /// `[t0 t1]`: `t` on each circle.
        domain: [f64; 2],
        /// Whether it extends past each circle.
        extend: [bool; 2],
    },
    /// Types 4 and 5: triangles, each vertex's values interpolated across it.
    Triangles(Vec<[Vertex; 3]>),
    /// Types 6 and 7: tensor-product patches.
    Patches(Vec<Patch>),
}

/// A mesh's vertex: where it is (the shading's space) and its values (the components, or `t`).
#[derive(Debug, Clone, PartialEq)]
pub struct Vertex {
    /// Where.
    pub point: Point,
    /// Its color's components, or its `t` when a function gives the color.
    pub values: Vec<f64>,
}

/// A tensor-product patch (8.7.4.5.8): `points[i][j]` is `p_ij`, the surface `sum B_i(u) B_j(v) p_ij` over the unit
/// square; `values` are its corners' -- at `(u, v)` = `(0, 0)`, `(0, 1)`, `(1, 1)`, `(1, 0)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Patch {
    /// The control points, `[i][j]`.
    pub points: [[Point; 4]; 4],
    /// The corners' values (components or `t`).
    pub values: [Vec<f64>; 4],
}

impl Shading {
    /// Whether a function gives the color (a mesh's values are each one `t`).
    #[must_use]
    pub fn parametric(&self) -> bool {
        !self.functions.is_empty()
    }

    /// The color components of `values`: the function's outputs for them, else themselves.
    ///
    /// # Errors
    ///
    /// Refuses an evaluation that yields no number.
    pub fn components(&self, values: &[f64]) -> Result<Vec<f64>> {
        match self.functions.as_slice() {
            [] => Ok(values.to_vec()),
            [one] => one.evaluate(values),
            each => each.iter().map(|f| Ok(f.evaluate(values)?[0])).collect(),
        }
    }

    /// The RGB of `values` (see [`Self::components`]) at `intent`; None when the space gives none.
    ///
    /// # Errors
    ///
    /// As [`Self::components`], and a space that refuses the components.
    pub fn rgb(&self, values: &[f64], intent: Intent) -> Result<Option<[f64; 3]>> {
        Ok(self.space.color(self.components(values)?, intent)?.rgb)
    }
}

impl File {
    /// The shading `value` is or refers to.
    ///
    /// # Errors
    ///
    /// Refuses a shading of no known `/ShadingType`, one missing what its type needs, a function that does not read
    /// or fit, and a mesh whose data does not decode or holds more than [`MAX_VERTICES`] vertices.
    pub fn shading(&self, value: &Object) -> Result<Shading> {
        let value = self.resolve(value)?;
        let dict = value
            .as_dict()
            .ok_or_else(|| Error::malformed(format!("a shading is a {}", value.kind())))?;
        let kind = self.get(dict, b"ShadingType")?.and_then(|t| t.as_integer());
        let space = match self.get(dict, b"ColorSpace")? {
            Some(cs) => self.color_space(&cs, &Dict::new())?,
            None => return Err(Error::malformed("a shading has no /ColorSpace")),
        };
        if matches!(space, Space::Pattern(_)) {
            return Err(Error::malformed("a shading in the Pattern space"));
        }
        let functions = self.shading_functions(dict, kind, &space)?;
        let background = self.numbers_at(dict, b"Background")?;
        let bbox = self.numbers_at(dict, b"BBox")?.and_then(|b| {
            <[f64; 4]>::try_from(b).ok().map(|[x0, y0, x1, y1]| {
                Rect::spanning(Point { x: x0, y: y0 }, Point { x: x1, y: y1 })
            })
        });
        let extend = || -> Result<[bool; 2]> {
            Ok(match self.get(dict, b"Extend")? {
                Some(Object::Array(e)) => [
                    matches!(e.first(), Some(Object::Bool(true))),
                    matches!(e.get(1), Some(Object::Bool(true))),
                ],
                _ => [false, false],
            })
        };
        let coords = |n: usize, what: &str| -> Result<Vec<f64>> {
            match self.numbers_at(dict, b"Coords")? {
                Some(c) if c.len() == n => Ok(c),
                _ => Err(Error::malformed(format!(
                    "a {what} shading's /Coords are not {n} numbers"
                ))),
            }
        };
        let kind = match kind {
            Some(1) => Kind::Function {
                domain: four(&self.fixed(dict, b"Domain", &[0.0, 1.0, 0.0, 1.0])?),
                matrix: Matrix::from_numbers(&self.fixed(dict, b"Matrix", &Matrix::IDENTITY.0)?)
                    .unwrap_or(Matrix::IDENTITY),
            },
            Some(2) => Kind::Axial {
                coords: four(&coords(4, "axial")?),
                domain: two(&self.fixed(dict, b"Domain", &[0.0, 1.0])?),
                extend: extend()?,
            },
            Some(3) => {
                let c = coords(6, "radial")?;
                Kind::Radial {
                    coords: [c[0], c[1], c[2], c[3], c[4], c[5]],
                    domain: two(&self.fixed(dict, b"Domain", &[0.0, 1.0])?),
                    extend: extend()?,
                }
            }
            Some(kind @ 4..=7) => {
                let values = if functions.is_empty() {
                    space.components()
                } else {
                    1
                };
                self.mesh(&value, kind, values)?
            }
            _ => return Err(Error::malformed("a shading of no known /ShadingType")),
        };
        Ok(Shading {
            space,
            background,
            bbox,
            kind,
            functions,
        })
    }

    /// A shading's `/Function`: one, or one per component, fitting its type's inputs and its space's components;
    /// none for a mesh that states its colors.
    fn shading_functions(
        &self,
        dict: &Dict,
        kind: Option<i64>,
        space: &Space,
    ) -> Result<Vec<Function>> {
        let functions = match self.get(dict, b"Function")? {
            None => Vec::new(),
            Some(Object::Array(each)) => each
                .iter()
                .map(|f| self.function(f))
                .collect::<Result<_>>()?,
            Some(one) => vec![self.function(&one)?],
        };
        let inputs = if kind == Some(1) { 2 } else { 1 };
        let fits = match functions.as_slice() {
            [] => kind.is_some_and(|k| k >= 4),
            [one] => one.inputs() == inputs && one.outputs() == space.components(),
            each => {
                each.len() == space.components()
                    && each
                        .iter()
                        .all(|f| f.inputs() == inputs && f.outputs() == 1)
            }
        };
        if fits {
            Ok(functions)
        } else {
            Err(Error::malformed(format!(
                "a type {} shading's /Function does not fit its {} color components",
                kind.unwrap_or(0),
                space.components()
            )))
        }
    }

    /// The numbers of the array `dict` holds at `key`, if it holds one.
    fn numbers_at(&self, dict: &Dict, key: &[u8]) -> Result<Option<Vec<f64>>> {
        Ok(self.get(dict, key)?.and_then(|o| {
            o.as_array()
                .map(|a| a.iter().filter_map(Object::as_number).collect())
        }))
    }

    /// The numbers at `key`, `default` when absent: as many as `default` holds (any count when it is empty).
    fn fixed(&self, dict: &Dict, key: &[u8], default: &[f64]) -> Result<Vec<f64>> {
        let got = self
            .numbers_at(dict, key)?
            .unwrap_or_else(|| default.to_vec());
        if got.len() == default.len() || default.is_empty() {
            Ok(got)
        } else {
            Err(Error::malformed(format!(
                "a shading's /{} holds {} numbers",
                String::from_utf8_lossy(key),
                got.len()
            )))
        }
    }

    /// A mesh shading's (type `kind`, 4 to 7) vertices or patches, each vertex carrying `values` values.
    fn mesh(&self, value: &Object, kind: i64, values: usize) -> Result<Kind> {
        let Object::Stream(stream) = value else {
            return Err(Error::malformed(format!(
                "a type {kind} shading is not a stream"
            )));
        };
        let dict = &stream.dict;
        let int = |key: &[u8]| -> Result<u32> {
            self.get(dict, key)?
                .and_then(|o| o.as_integer())
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| {
                    Error::malformed(format!(
                        "a mesh shading has no /{}",
                        String::from_utf8_lossy(key)
                    ))
                })
        };
        let decode = self.fixed(dict, b"Decode", &[])?;
        if decode.len() < 4 + 2 * values {
            return Err(Error::malformed(
                "a mesh shading's /Decode is shorter than its values",
            ));
        }
        let format = Format {
            coordinate: int(b"BitsPerCoordinate")?,
            component: int(b"BitsPerComponent")?,
            flag: if kind == 5 { 0 } else { int(b"BitsPerFlag")? },
            values,
            decode,
        };
        format.check()?;
        let data = self.decode(stream)?.into_bytes()?;
        let mut bits = Bits::new(&data);
        Ok(match kind {
            4 => Kind::Triangles(free_form(&mut bits, &format)?),
            5 => {
                let row = int(b"VerticesPerRow")?;
                Kind::Triangles(lattice(&mut bits, &format, row as usize)?)
            }
            _ => Kind::Patches(patches(&mut bits, &format, kind == 7)?),
        })
    }
}

fn four(v: &[f64]) -> [f64; 4] {
    [v[0], v[1], v[2], v[3]]
}

fn two(v: &[f64]) -> [f64; 2] {
    [v[0], v[1]]
}

/// How a mesh's data is laid out.
struct Format {
    coordinate: u32,
    component: u32,
    flag: u32,
    /// How many values a vertex carries.
    values: usize,
    /// `[xmin xmax ymin ymax v1min v1max ...]`.
    decode: Vec<f64>,
}

impl Format {
    fn check(&self) -> Result<()> {
        let ok = [1, 2, 4, 8, 12, 16, 24, 32].contains(&self.coordinate)
            && [1, 2, 4, 8, 12, 16].contains(&self.component)
            && [0, 2, 4, 8].contains(&self.flag);
        if ok {
            Ok(())
        } else {
            Err(Error::malformed(format!(
                "a mesh shading of {} bits a coordinate, {} a component, {} a flag",
                self.coordinate, self.component, self.flag
            )))
        }
    }

    fn vertex(&self, bits: &mut Bits) -> Option<Vertex> {
        let x = bits.sample(self.coordinate, self.decode[0], self.decode[1])?;
        let y = bits.sample(self.coordinate, self.decode[2], self.decode[3])?;
        let values = self.values(bits)?;
        Some(Vertex {
            point: Point { x, y },
            values,
        })
    }

    fn values(&self, bits: &mut Bits) -> Option<Vec<f64>> {
        (0..self.values)
            .map(|i| {
                bits.sample(
                    self.component,
                    self.decode[4 + 2 * i],
                    self.decode[5 + 2 * i],
                )
            })
            .collect()
    }

    fn point(&self, bits: &mut Bits) -> Option<Point> {
        Some(Point {
            x: bits.sample(self.coordinate, self.decode[0], self.decode[1])?,
            y: bits.sample(self.coordinate, self.decode[2], self.decode[3])?,
        })
    }
}

/// A stream's bits, read from the highest of each byte down.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    /// The next `n` bits as a number; None past the end.
    fn read(&mut self, n: u32) -> Option<u64> {
        let end = self.at + n as usize;
        if end > self.data.len() * 8 {
            return None;
        }
        let mut v = 0u64;
        for i in self.at..end {
            v = v << 1 | u64::from(self.data[i / 8] >> (7 - i % 8) & 1);
        }
        self.at = end;
        Some(v)
    }

    /// The next `n` bits mapped onto `[low, high]`.
    #[expect(clippy::cast_precision_loss, reason = "at most 32 bits")]
    fn sample(&mut self, n: u32, low: f64, high: f64) -> Option<f64> {
        let raw = self.read(n)? as f64;
        let max = ((1u64 << n) - 1) as f64;
        Some(low + raw * (high - low) / max)
    }
}

/// Type 4's triangles: a vertex flagged 0 starts one with the two after it; 1 or 2 makes one with the last one's
/// second and third vertices (1) or its first and third (2).
fn free_form(bits: &mut Bits, format: &Format) -> Result<Vec<[Vertex; 3]>> {
    let mut out: Vec<[Vertex; 3]> = Vec::new();
    while let Some(flag) = bits.read(format.flag) {
        let Some(v) = format.vertex(bits) else {
            break;
        };
        let triangle = match (flag, out.last()) {
            (1, Some([_, b, c])) => [b.clone(), c.clone(), v],
            (2, Some([a, _, c])) => [a.clone(), c.clone(), v],
            _ => {
                // A new triangle: two more vertices, their flags not read for meaning.
                let mut next = || bits.read(format.flag).and_then(|_| format.vertex(bits));
                let (Some(b), Some(c)) = (next(), next()) else {
                    break;
                };
                [v, b, c]
            }
        };
        out.push(triangle);
        if out.len() * 3 > MAX_VERTICES {
            return Err(Error::malformed("a mesh shading of too many vertices"));
        }
    }
    Ok(out)
}

/// Type 5's triangles: rows of `per_row` vertices, two triangles between each pair of neighbors in adjacent rows.
fn lattice(bits: &mut Bits, format: &Format, per_row: usize) -> Result<Vec<[Vertex; 3]>> {
    if per_row < 2 {
        return Err(Error::malformed(
            "a lattice mesh of fewer than 2 vertices a row",
        ));
    }
    let mut rows: Vec<Vec<Vertex>> = Vec::new();
    'rows: loop {
        let mut row = Vec::with_capacity(per_row);
        for _ in 0..per_row {
            match format.vertex(bits) {
                Some(v) => row.push(v),
                None => break 'rows,
            }
        }
        rows.push(row);
        if rows.len() * per_row > MAX_VERTICES {
            return Err(Error::malformed("a mesh shading of too many vertices"));
        }
    }
    let mut out = Vec::new();
    for pair in rows.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        for i in 0..per_row - 1 {
            out.push([a[i].clone(), a[i + 1].clone(), b[i].clone()]);
            out.push([a[i + 1].clone(), b[i + 1].clone(), b[i].clone()]);
        }
    }
    Ok(out)
}

/// The order a patch's points come in (8.7.4.5.7-8): the boundary from `p00` round, then (a tensor's) the inside.
const ORDER: [(usize, usize); 16] = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (3, 2),
    (3, 1),
    (3, 0),
    (2, 0),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 1),
];

/// Types 6 and 7's patches: a flag 0 patch states its 12 boundary points (and a tensor's 4 inside) and 4 corners'
/// values; 1, 2 or 3 takes its first side, and two corners, from the last patch's second, third or fourth side.
fn patches(bits: &mut Bits, format: &Format, tensor: bool) -> Result<Vec<Patch>> {
    let mut out: Vec<Patch> = Vec::new();
    while let Some(flag) = bits.read(format.flag) {
        let mut sequence = [Point::default(); 16];
        let mut values: [Vec<f64>; 4] = Default::default();
        let shared = match (flag, out.last()) {
            (1..=3, Some(last)) => {
                #[expect(clippy::cast_possible_truncation, reason = "1 to 3")]
                let side = flag as usize;
                for (k, slot) in sequence.iter_mut().take(4).enumerate() {
                    let (i, j) = ORDER[(3 * side + k) % 12];
                    *slot = last.points[i][j];
                }
                values[0].clone_from(&last.values[side]);
                values[1].clone_from(&last.values[(side + 1) % 4]);
                true
            }
            _ => false,
        };
        let (first, corners) = if shared { (4, 2) } else { (0, 0) };
        let count = if tensor { 16 } else { 12 };
        let mut complete = true;
        for slot in &mut sequence[first..count] {
            match format.point(bits) {
                Some(p) => *slot = p,
                None => complete = false,
            }
        }
        for slot in &mut values[corners..] {
            match format.values(bits) {
                Some(v) => *slot = v,
                None => complete = false,
            }
        }
        if !complete {
            break;
        }
        let mut points = [[Point::default(); 4]; 4];
        for (k, &(i, j)) in ORDER.iter().enumerate().take(count) {
            points[i][j] = sequence[k];
        }
        if !tensor {
            coons_inside(&mut points);
        }
        out.push(Patch { points, values });
        if out.len() * 16 > MAX_VERTICES {
            return Err(Error::malformed("a mesh shading of too many vertices"));
        }
    }
    Ok(out)
}

/// A Coons patch's four inside points, as the tensor-product patch with its boundary has them (8.7.4.5.8).
fn coons_inside(p: &mut [[Point; 4]; 4]) {
    let at = |terms: &[(f64, Point)]| Point {
        x: terms.iter().map(|(w, q)| w * q.x).sum::<f64>() / 9.0,
        y: terms.iter().map(|(w, q)| w * q.y).sum::<f64>() / 9.0,
    };
    let q = *p;
    let inside = |c: (usize, usize),
                  a: [(usize, usize); 2],
                  b: [(usize, usize); 2],
                  d: [(usize, usize); 2],
                  e: (usize, usize)| {
        let g = |(i, j): (usize, usize)| q[i][j];
        at(&[
            (-4.0, g(c)),
            (6.0, g(a[0])),
            (6.0, g(a[1])),
            (-2.0, g(b[0])),
            (-2.0, g(b[1])),
            (3.0, g(d[0])),
            (3.0, g(d[1])),
            (-1.0, g(e)),
        ])
    };
    p[1][1] = inside(
        (0, 0),
        [(0, 1), (1, 0)],
        [(0, 3), (3, 0)],
        [(3, 1), (1, 3)],
        (3, 3),
    );
    p[1][2] = inside(
        (0, 3),
        [(0, 2), (1, 3)],
        [(0, 0), (3, 3)],
        [(3, 2), (1, 0)],
        (3, 0),
    );
    p[2][2] = inside(
        (3, 3),
        [(3, 2), (2, 3)],
        [(3, 0), (0, 3)],
        [(2, 0), (0, 2)],
        (0, 0),
    );
    p[2][1] = inside(
        (3, 0),
        [(3, 1), (2, 0)],
        [(3, 3), (0, 0)],
        [(2, 3), (0, 1)],
        (0, 3),
    );
}
