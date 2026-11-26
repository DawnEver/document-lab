//! `document_lab._pdf`: the `pdf` crate, exposed to Python as plain values -- no PDF logic of its own.
//!
//! `open(data, password)` reads a file; its `page(number, curve_samples)` returns what the page draws as tuples,
//! which `document_lab.pdf` turns into the IR, and its `image(object)` an image's samples. `create(path, level)` writes a
//! PDF of image pages, or of pages copied from a `Reader` (`document_lab.capture`), perhaps with a text layer laid
//! over them (`document_lab.ocr`).
//!
//! A refusal raises by its kind: [`Refused`] and its kinds when the reader refuses the input, `ValueError` for a
//! request it cannot honor, `OSError` when writing fails.

use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use pdf::error::Kind;
use pdf::geometry::Rect;
use pdf::image::Placement;
use pdf::interpret::{Event, PathEvent};
use pdf::writer::Image;
use pyo3::create_exception;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

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
/// A span: text, font name (no subset tag), size, bold, box, render mode, the number it shares with the spans
/// its words run on into with no space between (None: none), and whether it is written vertically.
type SpanRow = (String, String, f64, bool, Box4, i64, Option<usize>, bool);
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
/// An image: box, sample width and height, object number (None inline), the matrix `(a, b, c, d, e, f)` from its
/// unit square to page space, and how it shows its samples (`pdf::image::Placement`: turned, rows reversed, columns
/// reversed; None when it is skewed).
type ImageRow = (Box4, i64, i64, Option<u32>, [f64; 6], Option<PlacementRow>);
/// An image placed square to the page: whether its rows run down the page, and its rows and columns shown reversed.
type PlacementRow = (bool, bool, bool);
/// What a page draws: spans, paths, images, warnings.
type PageRows = (Vec<SpanRow>, Vec<PathRow>, Vec<ImageRow>, Vec<String>);
/// An image's samples: width, height, color space family (None: a stencil mask), components, bits per component,
/// the data after every lossless filter, the image codec it is still in (None: raw samples), its `/Decode` array
/// (None: not stated), whether it states a mask, and its fax coding when its codec is `CCITTFaxDecode`.
type SamplesRow = (
    i64,
    i64,
    Option<String>,
    usize,
    i64,
    Vec<u8>,
    Option<String>,
    Option<Vec<f64>>,
    bool,
    Option<FaxRow>,
);
/// A `CCITTFaxDecode` image's coding: K, columns, end-of-line codes, rows byte-aligned, 1 bits black.
type FaxRow = (i64, i64, bool, bool, bool);

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
                let (font, bold) = (s.font.name_without_subset().to_string(), s.font.bold());
                (
                    s.text,
                    font,
                    s.size,
                    bold,
                    box4(s.bbox),
                    s.render_mode,
                    s.unspaced,
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
                    paths.extend(subpaths(path, curve_samples, painted));
                    painted += 1;
                }
                Event::Image(image) => {
                    images.push((
                        box4(image.bbox),
                        image.width,
                        image.height,
                        image.object,
                        image.matrix.0,
                        Placement::of(&image.matrix)
                            .map(|p| (p.turned, p.rows_reversed, p.columns_reversed)),
                    ));
                }
                Event::Text(_) => {}
            }
        }
        Ok((spans, paths, images, content.warnings))
    }

    /// The samples of the image `XObject` `object`.
    fn image(&self, object: u32) -> PyResult<SamplesRow> {
        let samples = self.file.image(object).map_err(|e| refused(&e))?;
        let fax = match &samples.codec {
            Some(codec) => codec.fax().map_err(|e| refused(&e))?,
            None => None,
        };
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
            samples.decode,
            samples.masked,
            fax.map(|f| (f.k, f.columns, f.end_of_line, f.byte_align, f.black_is_1)),
        ))
    }
}

/// A PDF being written, page by page, to a file.
#[pyclass(unsendable)]
struct Writer {
    /// None once closed.
    inner: Option<pdf::writer::Writer<BufWriter<fs::File>>>,
}

/// Start a PDF at `path` (created, or emptied), the streams it builds compressed at zlib `level` (0-10).
#[pyfunction]
#[pyo3(signature = (path, level = pdf::writer::LEVEL))]
#[expect(
    clippy::needless_pass_by_value,
    reason = "pyo3 hands a path over owned"
)]
fn create(path: PathBuf, level: u8) -> PyResult<Writer> {
    if level > 10 {
        return Err(PyValueError::new_err(format!(
            "a compression level is 0 to 10, not {level}"
        )));
    }
    let file = fs::File::create(&path)
        .map_err(|e| PyOSError::new_err(format!("{}: {e}", path.display())))?;
    let inner = pdf::writer::Writer::new(BufWriter::new(file), level).map_err(|e| refused(&e))?;
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

    /// Add page `number` (1-based) of `reader` as the next page, with `words` -- `(text, (x0, y0, x1, y1))`, each
    /// box in page space -- laid over it as an invisible text layer.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 hands a list over owned"
    )]
    fn copy_page_with_text(
        &mut self,
        reader: &Reader,
        number: usize,
        words: Vec<(String, Box4)>,
    ) -> PyResult<()> {
        let page = reader.at(number)?;
        let words: Vec<pdf::writer::Word> = words
            .iter()
            .map(|(text, (x0, y0, x1, y1))| pdf::writer::Word {
                text,
                bbox: Rect {
                    x0: *x0,
                    y0: *y0,
                    x1: *x1,
                    y1: *y1,
                },
            })
            .collect();
        self.open()?
            .copy_page_with_text(&reader.file, page, &words)
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

/// A painted path as one row per subpath, its curves flattened to `samples` points.
fn subpaths(path: &PathEvent, samples: usize, group: usize) -> Vec<PathRow> {
    let stroke = path.stroke.as_ref();
    pdf::paths::subpaths(&path.segments, samples)
        .into_iter()
        .map(|sub| {
            (
                sub.points.iter().map(|p| (p.x, p.y)).collect(),
                sub.closed,
                stroke.map(|s| s.color.rgb_or_components()),
                stroke.map_or(0.0, |s| s.width),
                stroke.map_or_else(Vec::new, |s| s.dashes.clone()),
                stroke.map_or(0.0, |s| s.dash_phase),
                path.fill.as_ref().map(|f| f.color.rgb_or_components()),
                path.fill.as_ref().is_some_and(|f| f.even_odd),
                group,
            )
        })
        .collect()
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
