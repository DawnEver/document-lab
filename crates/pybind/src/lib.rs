//! `document_lab._pdf`: the `pdf` crate, exposed to Python as plain values -- no PDF logic of its own.
//!
//! `open(data, password)` reads a file; its `page(number, curve_samples)` returns what the page draws as tuples,
//! which `document_lab.pdf` turns into the IR, and its `image(object)` an image's samples. `create(path)` writes a
//! PDF of image pages, or of pages copied from a `Reader` (`document_lab.capture`).

use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use pdf::color::Color;
use pdf::geometry::{Point, Rect};
use pdf::interpret::{Event, PathEvent, Segment};
use pdf::writer::Image;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

/// A box as `(x0, y0, x1, y1)`.
type Box4 = (f64, f64, f64, f64);
/// A span: text, font name, size, descriptor flags, font weight, box, render mode, written vertically.
type SpanRow = (String, String, f64, i64, Option<f64>, Box4, i64, bool);
/// A path (one subpath): points, closed, stroke color, line width, dashes, dash phase, fill color, even-odd, and
/// the index of the painted path it belongs to on its page.
type PathRow = (
    Vec<(f64, f64)>,
    bool,
    Option<Vec<f64>>,
    f64,
    Vec<f64>,
    f64,
    Option<Vec<f64>>,
    bool,
    usize,
);
/// An image: box, sample width and height, object number (None inline).
type ImageRow = (Box4, i64, i64, Option<u32>);
/// What a page draws: spans, paths, images, warnings.
type PageRows = (Vec<SpanRow>, Vec<PathRow>, Vec<ImageRow>, Vec<String>);
/// An image's samples: width, height, color space family (None: a stencil mask), components, bits per component,
/// the data after every lossless filter, and the image codec it is still in (None: raw samples).
type SamplesRow = (
    i64,
    i64,
    Option<String>,
    usize,
    i64,
    Vec<u8>,
    Option<String>,
);

/// A PDF opened.
#[pyclass(unsendable)]
struct Reader {
    file: pdf::File,
    pages: Vec<pdf::Page>,
}

/// Open the PDF `data` with `password` (empty: none).
#[pyfunction]
fn open(data: Vec<u8>, password: &[u8]) -> PyResult<Reader> {
    let file = pdf::File::open(data, password).map_err(|e| refused(&e))?;
    let pages = file.pages().map_err(|e| refused(&e))?;
    Ok(Reader { file, pages })
}

#[pymethods]
impl Reader {
    /// What was repaired to read the file, in words.
    #[getter]
    fn repairs(&self) -> Vec<String> {
        self.file.repairs()
    }

    /// Each page's number, width and height as displayed (points).
    #[getter]
    fn pages(&self) -> Vec<(usize, f64, f64)> {
        self.pages
            .iter()
            .map(|p| (p.number, p.size().0, p.size().1))
            .collect()
    }

    /// What page `number` (1-based) draws: (spans, paths, images, warnings); each curve flattened to
    /// `curve_samples` points.
    fn page(&self, number: usize, curve_samples: usize) -> PyResult<PageRows> {
        let page = self.at(number)?;
        let content = self.file.interpret(page).map_err(|e| refused(&e))?;
        let spans = pdf::spans::spans(&content.events)
            .into_iter()
            .map(|s| {
                let font = &s.font;
                (
                    s.text,
                    font.name.clone(),
                    s.size,
                    font.flags,
                    font.weight,
                    box4(s.bbox),
                    s.render_mode,
                    s.vertical,
                )
            })
            .collect();
        let mut paths = Vec::new();
        let mut images = Vec::new();
        let mut painted = 0;
        for event in &content.events {
            match event {
                Event::Path(path) => {
                    paths.extend(subpaths(path, curve_samples.max(1), painted));
                    painted += 1;
                }
                Event::Image(image) => {
                    images.push((box4(image.bbox), image.width, image.height, image.object));
                }
                Event::Text(_) => {}
            }
        }
        Ok((spans, paths, images, content.warnings))
    }

    /// The samples of the image `XObject` `object`.
    fn image(&self, object: u32) -> PyResult<SamplesRow> {
        let samples = self.file.image(object).map_err(|e| refused(&e))?;
        let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
        Ok((
            samples.width,
            samples.height,
            samples.space.as_ref().map(|s| text(&s.family())),
            samples
                .space
                .as_ref()
                .map_or(1, pdf::color::Space::components),
            samples.bits,
            samples.data,
            samples.codec.map(|c| text(&c.name)),
        ))
    }
}

/// A PDF being written, page by page, to a file.
#[pyclass(unsendable)]
struct Writer {
    /// None once closed.
    inner: Option<pdf::writer::Writer<BufWriter<fs::File>>>,
}

/// Start a PDF at `path` (created, or emptied).
#[pyfunction]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 hands a path over owned"
)]
fn create(path: PathBuf) -> PyResult<Writer> {
    let file = fs::File::create(&path)
        .map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let inner = pdf::writer::Writer::new(BufWriter::new(file)).map_err(|e| refused(&e))?;
    Ok(Writer { inner: Some(inner) })
}

fn refused(e: &pdf::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[pymethods]
impl Writer {
    /// How many pages were added.
    #[getter]
    fn pages(&self) -> usize {
        self.inner.as_ref().map_or(0, pdf::writer::Writer::pages)
    }

    /// Add a page `page_width` x `page_height` points showing an image over all of it: `width` x `height` samples
    /// in `space` (`DeviceGray`, `DeviceRGB`, `DeviceCMYK`), `bits` a component, `data` as stored, encoded with
    /// `filter` (None: raw).
    #[expect(
        clippy::too_many_arguments,
        reason = "an image's every stored attribute, each a plain value"
    )]
    #[pyo3(signature = (width, height, space, bits, filter, data, page_width, page_height))]
    fn image_page(
        &mut self,
        width: u32,
        height: u32,
        space: &str,
        bits: u8,
        filter: Option<&str>,
        data: &[u8],
        page_width: f64,
        page_height: f64,
    ) -> PyResult<()> {
        let image = Image {
            width,
            height,
            space: space.as_bytes(),
            bits,
            filter: filter.map(str::as_bytes),
            data,
        };
        self.open()?
            .image_page(&image, (page_width, page_height))
            .map_err(|e| refused(&e))
    }

    /// Add page `number` (1-based) of `reader` as the next page, with everything it draws.
    fn copy_page(&mut self, reader: &Reader, number: usize) -> PyResult<()> {
        let page = reader.at(number)?;
        self.open()?
            .copy_page(&reader.file, page)
            .map_err(|e| refused(&e))
    }

    /// Finish the file: its page tree and cross-reference table; the pages written. Closing twice is refused.
    fn close(&mut self) -> PyResult<usize> {
        let writer = self
            .inner
            .take()
            .ok_or_else(|| PyValueError::new_err("the PDF is closed"))?;
        let pages = writer.pages();
        writer.finish().map_err(|e| refused(&e))?;
        Ok(pages)
    }
}

impl Writer {
    fn open(&mut self) -> PyResult<&mut pdf::writer::Writer<BufWriter<fs::File>>> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("the PDF is closed"))
    }
}

impl Reader {
    /// Page `number`, 1-based.
    fn at(&self, number: usize) -> PyResult<&pdf::Page> {
        number
            .checked_sub(1)
            .and_then(|i| self.pages.get(i))
            .ok_or_else(|| PyValueError::new_err(format!("no page {number}")))
    }
}

fn box4(r: Rect) -> Box4 {
    (r.x0, r.y0, r.x1, r.y1)
}

/// A color as RGB when its space defines it, else as its own components.
fn color(c: &Color) -> Vec<f64> {
    c.rgb
        .map_or_else(|| c.components.clone(), |rgb| rgb.to_vec())
}

/// A painted path as one row per subpath, curves flattened.
fn subpaths(path: &PathEvent, samples: usize, group: usize) -> Vec<PathRow> {
    let stroke = path.stroke.as_ref();
    let row = |points: Vec<(f64, f64)>, closed: bool| -> PathRow {
        (
            points,
            closed,
            stroke.map(|s| color(&s.color)),
            stroke.map_or(0.0, |s| s.width),
            stroke.map_or_else(Vec::new, |s| s.dashes.clone()),
            stroke.map_or(0.0, |s| s.dash_phase),
            path.fill.as_ref().map(|f| color(&f.color)),
            path.fill.as_ref().is_some_and(|f| f.even_odd),
            group,
        )
    };
    let mut out = Vec::new();
    let mut points: Vec<(f64, f64)> = Vec::new();
    let mut closed = false;
    let mut last = Point { x: 0.0, y: 0.0 };
    for segment in &path.segments {
        match *segment {
            Segment::MoveTo(p) => {
                if points.len() > 1 {
                    out.push(row(std::mem::take(&mut points), closed));
                }
                points = vec![(p.x, p.y)];
                closed = false;
                last = p;
            }
            Segment::LineTo(p) => {
                points.push((p.x, p.y));
                last = p;
            }
            Segment::CurveTo(c1, c2, p) => {
                for i in 1..=samples {
                    #[expect(clippy::cast_precision_loss, reason = "a sample count is small")]
                    let t = i as f64 / samples as f64;
                    let u = 1.0 - t;
                    let weights = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
                    let [w0, w1, w2, w3] = weights;
                    points.push((
                        w0 * last.x + w1 * c1.x + w2 * c2.x + w3 * p.x,
                        w0 * last.y + w1 * c1.y + w2 * c2.y + w3 * p.y,
                    ));
                }
                last = p;
            }
            Segment::Close => closed = true,
        }
    }
    if points.len() > 1 {
        out.push(row(points, closed));
    }
    out
}

#[pymodule]
#[pyo3(name = "_pdf")]
fn extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(open, module)?)?;
    module.add_class::<Reader>()?;
    module.add_function(wrap_pyfunction!(create, module)?)?;
    module.add_class::<Writer>()?;
    Ok(())
}
