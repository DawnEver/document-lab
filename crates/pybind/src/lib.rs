//! `document_lab._pdf`: the `pdf` crate, exposed to Python as plain values -- no PDF logic of its own.
//!
//! `open(data, password)` reads a file; its `page(number, curve_samples=, flatness=)` returns what the page draws
//! as tuples and columns, which `document_lab.pdf` turns into the IR, and its `image(object)` an image's samples. `create(path)` writes a
//! PDF of image pages, or of pages copied from a `Reader` (`document_lab.capture`).
//!
//! A refusal raises by its kind: [`Refused`] and its kinds when the reader refuses the input, `ValueError` for a
//! request it cannot honor, `OSError` when writing fails.

use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use pdf::error::Kind;
use pdf::geometry::Rect;
use pdf::interpret::{Event, PathEvent};
use pdf::paths::{Flattening, Polylines};
use pdf::writer::Image;
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

create_exception!(
    document_lab._pdf,
    Refused,
    PyValueError,
    "The reader refuses the input: one of its kinds says why."
);
create_exception!(
    document_lab._pdf,
    NotPdf,
    Refused,
    "The input is no PDF at all."
);
create_exception!(
    document_lab._pdf,
    Locked,
    Refused,
    "The PDF is encrypted and the password given -- none, or a wrong one -- opens no access."
);
create_exception!(
    document_lab._pdf,
    Unsupported,
    Refused,
    "The PDF uses what the reader does not read (a security handler, a filter)."
);
create_exception!(
    document_lab._pdf,
    Malformed,
    Refused,
    "The input is not what a PDF must be."
);

/// A box as `(x0, y0, x1, y1)`.
type Box4 = (f64, f64, f64, f64);
/// A span: text, font name (no subset tag), size, bold, box, render mode, and the number it shares with the spans
/// its words run on into with no space between (None: none).
type SpanRow = (String, String, f64, bool, Box4, i64, Option<usize>);
/// What a page paints as paths, as columns -- `document_lab.pdf` takes each buffer whole with `numpy.frombuffer`,
/// no Python object per point. Buffers are native-endian; an entry per subpath (`pdf::paths::Polylines`) unless
/// said:
/// - points: `float64` (x, y) pairs, every subpath's points one after another, curves flattened;
/// - bounds: `int64`, the pair where each subpath starts, then where the last ends (one entry more);
/// - boxes: `float64` (x0, y0, x1, y1), the bounds of its points;
/// - closed: `uint8`, 1 when closed back to its start;
/// - group: `int64`, the painted path it is a subpath of: its index in the styles;
/// - styles: a [`StyleRow`] per painted path, in drawing order -- what its subpaths share.
type PathColumns<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Vec<StyleRow>,
);
/// How a painted path is painted: stroke color, line width, dashes, dash phase, fill color, even-odd.
type StyleRow = (Option<Vec<f64>>, f64, Vec<f64>, f64, Option<Vec<f64>>, bool);
/// An image: box, sample width and height, object number (None inline).
type ImageRow = (Box4, i64, i64, Option<u32>);
/// What a page draws: spans, paths (as columns), images, warnings.
type PageRows<'py> = (Vec<SpanRow>, PathColumns<'py>, Vec<ImageRow>, Vec<String>);
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

    /// What page `number` (1-based) draws: (spans, paths, images, warnings); each curve flattened into
    /// `curve_samples` segments, or into as few as keep within `flatness` points of it -- one of the two
    /// (`pdf::paths::Flattening`).
    #[pyo3(signature = (number, *, curve_samples = None, flatness = None))]
    fn page<'py>(
        &self,
        py: Python<'py>,
        number: usize,
        curve_samples: Option<usize>,
        flatness: Option<f64>,
    ) -> PyResult<PageRows<'py>> {
        let flattening = match (curve_samples, flatness) {
            (Some(n), None) if n > 0 => Flattening::Samples(n),
            (None, Some(d)) if d > 0.0 => Flattening::Flatness(d),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "a curve is flattened by a curve_samples above 0 or by a flatness above 0, one of the two; \
                     given {curve_samples:?} and {flatness:?}"
                )));
            }
        };
        let page = self.at(number)?;
        let pdf::Content { events, warnings } =
            self.file.interpret(page).map_err(|e| refused(&e))?;
        let spans = pdf::spans::spans(&events)
            .into_iter()
            .map(|s| {
                let (font, bold) = (s.font.name_without_subset().to_string(), s.font.bold());
                (
                    s.text,
                    font,
                    s.size,
                    bold,
                    box4(s.bbox),
                    s.render_mode,
                    s.unspaced,
                )
            })
            .collect();
        let mut lines = Polylines::new();
        let (mut groups, mut styles) = (Vec::new(), Vec::new());
        let mut images = Vec::new();
        // Each event is let go once read: a page's drawing is held as segments or as points, never both at once.
        for event in events {
            match event {
                Event::Path(path) => {
                    lines.add(&path.segments, flattening);
                    groups.resize(lines.len(), styles.len());
                    styles.push(style(&path));
                }
                Event::Image(image) => {
                    images.push((box4(image.bbox), image.width, image.height, image.object));
                }
                Event::Text(_) => {}
            }
        }
        let paths = columns(py, &lines, &groups, styles);
        Ok((spans, paths, images, warnings))
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

/// The Python exception of `e`'s kind.
fn refused(e: &pdf::Error) -> PyErr {
    let message = e.to_string();
    match e.kind {
        Kind::NotPdf => NotPdf::new_err(message),
        Kind::Password => Locked::new_err(message),
        Kind::Unsupported => Unsupported::new_err(message),
        Kind::Malformed => Malformed::new_err(message),
        Kind::Invalid => PyValueError::new_err(message),
        Kind::Io => PyOSError::new_err(message),
    }
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

/// How `path` is painted.
fn style(path: &PathEvent) -> StyleRow {
    let stroke = path.stroke.as_ref();
    (
        stroke.map(|s| s.color.rgb_or_components()),
        stroke.map_or(0.0, |s| s.width),
        stroke.map_or_else(Vec::new, |s| s.dashes.clone()),
        stroke.map_or(0.0, |s| s.dash_phase),
        path.fill.as_ref().map(|f| f.color.rgb_or_components()),
        path.fill.as_ref().is_some_and(|f| f.even_odd),
    )
}

/// `lines` as [`PathColumns`], subpath `i` of the painted path `groups[i]`.
fn columns<'py>(
    py: Python<'py>,
    lines: &Polylines,
    groups: &[usize],
    styles: Vec<StyleRow>,
) -> PathColumns<'py> {
    let boxes = (0..lines.len()).flat_map(|i| {
        let (first, rest) = lines
            .subpath(i)
            .split_first()
            .expect("a subpath has two points or more");
        rest.iter().fold(
            [first.x, first.y, first.x, first.y],
            |[x0, y0, x1, y1], p| [x0.min(p.x), y0.min(p.y), x1.max(p.x), y1.max(p.y)],
        )
    });
    #[expect(
        clippy::cast_possible_wrap,
        reason = "an index into a page's points fits an int64"
    )]
    let index = |i: &usize| *i as i64;
    (
        packed(
            py,
            2 * lines.points.len(),
            lines.points.iter().flat_map(|p| [p.x, p.y]),
        ),
        packed(py, lines.bounds.len(), lines.bounds.iter().map(index)),
        packed(py, 4 * lines.len(), boxes),
        packed(py, lines.len(), lines.closed.iter().map(|c| u8::from(*c))),
        packed(py, groups.len(), groups.iter().map(index)),
        styles,
    )
}

/// A number a column holds.
trait Packed: Copy {
    /// Its native-endian bytes.
    type Bytes: AsRef<[u8]>;
    /// Its size in bytes.
    const SIZE: usize;
    /// Its native-endian bytes.
    fn bytes(self) -> Self::Bytes;
}

impl Packed for f64 {
    type Bytes = [u8; 8];
    const SIZE: usize = 8;
    fn bytes(self) -> [u8; 8] {
        self.to_ne_bytes()
    }
}

impl Packed for i64 {
    type Bytes = [u8; 8];
    const SIZE: usize = 8;
    fn bytes(self) -> [u8; 8] {
        self.to_ne_bytes()
    }
}

impl Packed for u8 {
    type Bytes = [u8; 1];
    const SIZE: usize = 1;
    fn bytes(self) -> [u8; 1] {
        [self]
    }
}

/// The `count` `values` as one `bytes`, each native-endian -- a column `numpy.frombuffer` takes without a copy --
/// written straight into it.
fn packed<T: Packed>(
    py: Python<'_>,
    count: usize,
    values: impl Iterator<Item = T>,
) -> Bound<'_, PyBytes> {
    PyBytes::new_with(py, count * T::SIZE, |out| {
        let mut chunks = out.chunks_exact_mut(T::SIZE);
        for (chunk, value) in chunks.by_ref().zip(values) {
            chunk.copy_from_slice(value.bytes().as_ref());
        }
        assert!(chunks.next().is_none(), "fewer values than counted");
        Ok(())
    })
    .expect("filling a new bytes object cannot fail")
}

#[pymodule]
#[pyo3(name = "_pdf")]
fn extension(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(open, module)?)?;
    module.add_class::<Reader>()?;
    module.add_function(wrap_pyfunction!(create, module)?)?;
    module.add_class::<Writer>()?;
    let py = module.py();
    module.add("Refused", py.get_type::<Refused>())?;
    module.add("NotPdf", py.get_type::<NotPdf>())?;
    module.add("Locked", py.get_type::<Locked>())?;
    module.add("Unsupported", py.get_type::<Unsupported>())?;
    module.add("Malformed", py.get_type::<Malformed>())?;
    Ok(())
}
