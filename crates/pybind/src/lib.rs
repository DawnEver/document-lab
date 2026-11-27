//! `document_lab._pdf`: the `pdf` crate, exposed to Python as plain values -- no PDF logic of its own.
//!
//! `open(source, password)` reads a PDF -- a file by its path, read here, or one already in memory as bytes -- and
//! gives its `sha256`; its `page(number, annotations=, curve_samples=, flatness=)` returns what the page draws as
//! tuples and columns, which `document_lab.pdf` turns into the IR, its `structure()` a tagged file's structure tree,
//! its `links(number)` the page's links, its `document()` what the document states beside its pages (outline,
//! labels, metadata, form fields, embedded files), its `render(...)` a region drawn by the own renderer (`render`),
//! its `image(object)` an image's samples, its `rgb(object)` their sRGB where they show as they stand and its
//! `attachment(object)` an embedded file's bytes. Pages are numbered from 1 in every row. `create(path, level)`
//! writes a PDF of image pages, or of pages copied from a `Reader` (`document_lab.capture`), perhaps with a text
//! layer laid over them (`document_lab.ocr`).
//!
//! A refusal raises by its kind: [`Refused`] and its kinds when the reader refuses the input, `ValueError` for a
//! request it cannot honor, `OSError` when writing fails.

use std::cell::OnceCell;
use std::fmt::Write;
use std::fs;
use std::io::{self, BufWriter};
use std::path::PathBuf;

use pdf::color::Pattern;
use pdf::destination::{Action, Destination, Targets};
use pdf::error::Kind;
use pdf::forms::{Field, Value};
use pdf::geometry::{Point, Rect};
use pdf::image::Placement;
use pdf::interpret::{Event, Layer, Marked, PathEvent, Segment};
use pdf::metadata::{Metadata, Newer};
use pdf::outline::Item;
use pdf::paths::{Flattening, Polylines};
use pdf::structure::{Element, Kid, Value as StructureValue};
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
/// The layer an event is drawn in: hidden by optional content, inkless (painted in the colorant `None`), and the
/// subtype of the annotation drawing it.
type LayerRow = (bool, bool, Option<String>);
/// The marked content an event is drawn in: its `/MCID` (None: none), the Form `XObject` holding it (None: the
/// page's content), and whether an `/Artifact` holds it.
type MarkedRow = (Option<i64>, Option<u32>, bool);
/// A span: text, font name (no subset tag), size, bold, box, render mode, the number it shares with the spans its
/// words run on into with no space between (None: none), whether it is written vertically, the unit vector its
/// baseline runs along (page space, y down), its layer, its fill color (None: not filled), and what its marked content
/// says: the text its glyphs draw when an `/ActualText` replaced it (None: the text is that), and the sequence itself.
type SpanRow = (
    String,
    String,
    f64,
    bool,
    Box4,
    i64,
    Option<usize>,
    bool,
    (f64, f64),
    LayerRow,
    Option<Vec<f64>>,
    (Option<String>, MarkedRow),
);
/// What a page paints as paths, as columns -- `document_lab.pdf` takes each buffer whole with `numpy.frombuffer`,
/// no Python object per point. Buffers are native-endian; an entry per subpath (`pdf::paths::Polylines`) unless
/// said:
/// - points: `float64` (x, y) pairs, every subpath's points one after another, curves flattened;
/// - bounds: `int64`, the pair where each subpath starts, then where the last ends (one entry more);
/// - boxes: `float64` (x0, y0, x1, y1), the bounds of its points;
/// - closed: `uint8`, 1 when closed back to its start;
/// - group: `int64`, the painted path it is a subpath of: its index in the styles;
/// - styles: a [`StyleRow`] per painted path, in drawing order -- what its subpaths share.
///
/// A shading painted with `sh` is a painted path of one closed subpath, the rectangle of its box, filled with a
/// `shading`.
type PathColumns<'py> = (
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Vec<StyleRow>,
);
/// How a painted path is painted: stroke color, line width, dashes, dash phase, fill color, even-odd, the fill's
/// pattern (`tiling`, `shading`; None: one color), and its layer.
type StyleRow = (
    Option<Vec<f64>>,
    f64,
    Vec<f64>,
    f64,
    Option<Vec<f64>>,
    bool,
    Option<&'static str>,
    LayerRow,
    MarkedRow,
);
/// An image: box, sample width and height, object number (None inline), the matrix `(a, b, c, d, e, f)` from its
/// unit square to page space, how it shows its samples (`pdf::image::Placement`: turned, rows reversed, columns
/// reversed; None when it is skewed), its layer and its marked content.
type ImageRow = (
    Box4,
    i64,
    i64,
    Option<u32>,
    [f64; 6],
    Option<PlacementRow>,
    LayerRow,
    MarkedRow,
);
/// An image placed square to the page: whether its rows run down the page, and its rows and columns shown reversed.
type PlacementRow = (bool, bool, bool);
/// A structure tree: its elements in tree order, the indexes of those under the root, its warnings.
type StructureRow = (Vec<ElementRow>, Vec<usize>, Vec<String>);
/// A structure element: role as stated, standard role, title, language, alternate description, actual text,
/// expansion, attributes (owner, name, value), page (1-based) and kids.
type ElementRow = (
    String,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Vec<(String, String, AttributeValue)>,
    Option<usize>,
    Vec<KidRow>,
);
/// A kid: `("element", index, None, None, None)`, `("marked", page, mcid, stream, None)` or `("object", page,
/// None, object, rect)`.
type KidRow = (
    &'static str,
    Option<usize>,
    Option<i64>,
    Option<u32>,
    Option<Box4>,
);

/// An attribute's value as Python's: a bool, int, float, str (a name or a text string), list, or list of pairs.
#[derive(IntoPyObject)]
enum AttributeValue {
    Bool(bool),
    Integer(i64),
    Real(f64),
    Text(String),
    Array(Vec<AttributeValue>),
    Pairs(Vec<(String, AttributeValue)>),
}
/// What a page draws: spans, paths (as columns), images, warnings.
type PageRows<'py> = (Vec<SpanRow>, PathColumns<'py>, Vec<ImageRow>, Vec<String>);
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

/// A destination: page number, fit, x, y, zoom, the box a `FitR` fits (page space).
type DestinationRow = (
    usize,
    String,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<Box4>,
);
/// An action: kind, destination, URI, file, name.
type ActionRow = (
    String,
    Option<DestinationRow>,
    Option<String>,
    Option<String>,
    Option<String>,
);
/// An outline item, the items flattened in document order: depth (0 top level), title, destination, action.
type ItemRow = (
    usize,
    Option<String>,
    Option<DestinationRow>,
    Option<ActionRow>,
);
/// A link: box (page space), destination, action.
type LinkRow = (Box4, Option<DestinationRow>, Option<ActionRow>);
/// The XMP packet: title, creators, description, created, modified, metadata date, creator tool, producer,
/// keywords, the packet's text.
type XmpRow = (
    Option<String>,
    Vec<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    String,
);
/// The metadata: the `/Info` entries, the XMP packet, which is newer (`info`, `xmp`, `same`).
type MetadataRow = (
    Option<Vec<(String, String)>>,
    Option<XmpRow>,
    Option<&'static str>,
);
/// A widget: page number, box (page space), appearance state, export value.
type WidgetRow = (Option<usize>, Option<Box4>, Option<String>, Option<String>);
/// A form field: name, type, flags, value, default, options (export, shown), widgets.
type FieldRow = (
    Option<String>,
    Option<String>,
    Option<i64>,
    Option<ValueRow>,
    Option<ValueRow>,
    Vec<(String, String)>,
    Vec<WidgetRow>,
);
/// An embedded file: tree key, file name, description, MIME type, size, created, modified, page number, object.
type AttachmentRow = (
    Option<String>,
    Option<String>,
    Option<String>,
    Option<String>,
    Option<i64>,
    Option<String>,
    Option<String>,
    Option<usize>,
    u32,
);
/// What the document states beside its pages: outline, page labels (None: none stated), metadata, fields,
/// embedded files, warnings.
type DocumentRows = (
    Vec<ItemRow>,
    Option<Vec<Option<String>>>,
    MetadataRow,
    Vec<FieldRow>,
    Vec<AttachmentRow>,
    Vec<String>,
);

/// A form field's value: one string (a text string or a name, as stated), or several.
#[derive(IntoPyObject)]
enum ValueRow {
    One(String),
    Several(Vec<String>),
}
/// A region drawn: width, height, channels (3 RGB, 4 RGBA), the pixels rows top-down, and each kind of mark not
/// drawn as the file states it with how many.
type RasterRow = (u32, u32, u8, Vec<u8>, Vec<(String, usize)>);

/// A PDF opened.
#[pyclass(unsendable)]
struct Reader {
    file: pdf::File,
    pages: Vec<pdf::Page>,
    /// What resolving destinations needs, and what reading it warned of: read when first needed.
    targets: OnceCell<(Targets, Vec<String>)>,
}

/// Open the PDF `source` with `password` (empty: none): the file at a path (`str` or `os.PathLike`), read whole
/// here, or the bytes of one already in memory (`bytes`), copied. Python holds no copy of a file read here.
#[pyfunction]
fn open(source: &Bound<'_, PyAny>, password: &[u8]) -> PyResult<Reader> {
    let data = if let Ok(bytes) = source.cast::<PyBytes>() {
        bytes.as_bytes().to_vec()
    } else {
        let path: PathBuf = source.extract()?;
        fs::read(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?
    };
    let file = pdf::File::open(data, password).map_err(|e| refused(&e))?;
    let pages = file.pages().map_err(|e| refused(&e))?;
    Ok(Reader {
        file,
        pages,
        targets: OnceCell::new(),
    })
}

#[pymethods]
impl Reader {
    /// What was repaired to read the file, in words.
    #[getter]
    fn repairs(&self) -> Vec<String> {
        self.file.repairs()
    }

    /// The SHA-256 digest of the bytes read, in hexadecimal.
    #[getter]
    fn sha256(&self) -> String {
        self.file
            .sha256()
            .iter()
            .fold(String::with_capacity(64), |mut hex, b| {
                write!(hex, "{b:02x}").expect("writing into a String cannot fail");
                hex
            })
    }

    /// Each page's number, width and height as displayed (points).
    #[getter]
    fn pages(&self) -> Vec<(usize, f64, f64)> {
        self.pages
            .iter()
            .map(|p| (p.number, p.size().0, p.size().1))
            .collect()
    }

    /// What page `number` (1-based) draws, its annotations' appearances too when `annotations`: (spans, paths,
    /// images, warnings); each curve flattened into `curve_samples` segments, or into as few as keep within
    /// `flatness` points of it -- one of the two (`pdf::paths::Flattening`).
    #[pyo3(signature = (number, *, annotations, curve_samples = None, flatness = None))]
    fn page<'py>(
        &self,
        py: Python<'py>,
        number: usize,
        annotations: bool,
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
        let pdf::Content { events, warnings } = self
            .file
            .interpret(page, annotations)
            .map_err(|e| refused(&e))?;
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
                    s.vertical,
                    s.direction,
                    layer(&s.layer),
                    s.fill.as_ref().map(pdf::color::Color::rgb_or_components),
                    (s.drawn, marked(s.marked)),
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
                    images.push((
                        box4(image.bbox),
                        image.width,
                        image.height,
                        image.object,
                        image.matrix.0,
                        Placement::of(&image.matrix)
                            .map(|p| (p.turned, p.rows_reversed, p.columns_reversed)),
                        layer(&image.layer),
                        marked(image.marked),
                    ));
                }
                Event::Shading(shading) => {
                    lines.add(&rectangle(shading.bbox), flattening);
                    groups.resize(lines.len(), styles.len());
                    styles.push((
                        None,
                        0.0,
                        Vec::new(),
                        0.0,
                        Some(Vec::new()),
                        false,
                        Some(pattern(Pattern::Shading)),
                        layer(&shading.layer),
                        marked(shading.marked),
                    ));
                }
                Event::Text(_) => {}
            }
        }
        let paths = columns(py, &lines, &groups, styles);
        Ok((spans, paths, images, warnings))
    }

    /// The structure tree, when the file is tagged: (elements, roots, warnings); None when it is not.
    fn structure(&self) -> PyResult<Option<StructureRow>> {
        let structure = self.file.structure(&self.pages).map_err(|e| refused(&e))?;
        Ok(structure.map(|s| {
            (
                s.elements.iter().map(element).collect(),
                s.roots,
                s.warnings,
            )
        }))
    }

    /// The links on page `number` (1-based), read whether or not annotations are drawn: (links, warnings).
    fn links(&self, number: usize) -> PyResult<(Vec<LinkRow>, Vec<String>)> {
        let page = self.at(number)?;
        let mut notes = Vec::new();
        let links = self
            .file
            .links(page, self.targets(), &mut notes)
            .into_iter()
            .map(|l| {
                (
                    box4(l.rect),
                    l.destination.as_ref().map(destination),
                    l.action.as_ref().map(action),
                )
            })
            .collect();
        Ok((links, notes))
    }

    /// What the document states beside its pages: (outline, page labels, metadata, fields, embedded files,
    /// warnings).
    fn document(&self) -> DocumentRows {
        let (targets, notes) = self.targets.get_or_init(|| self.read_targets());
        let mut notes = notes.clone();
        let mut outline = Vec::new();
        flatten(&self.file.outline(targets, &mut notes), 0, &mut outline);
        let labels = self.file.page_labels(self.pages.len(), &mut notes);
        let metadata = metadata(self.file.metadata(&mut notes));
        let fields = self
            .file
            .fields(&self.pages, &mut notes)
            .into_iter()
            .map(field)
            .collect();
        let attachments = self
            .file
            .attachments(&self.pages, &mut notes)
            .into_iter()
            .map(|a| {
                (
                    a.key,
                    a.name,
                    a.description,
                    a.mime,
                    a.size,
                    a.created,
                    a.modified,
                    a.page.map(|p| p + 1),
                    a.object,
                )
            })
            .collect();
        (outline, labels, metadata, fields, attachments, notes)
    }

    /// The bytes of the embedded file stream `object`, every filter undone.
    fn attachment(&self, object: u32) -> PyResult<Vec<u8>> {
        self.file.embedded(object).map_err(|e| refused(&e))
    }

    /// The `region` (page space; None: the whole page) of page `number` drawn by the own renderer (`render`) at
    /// `scale` pixels a point -- text only when `text` (refused until stage 2), annotations' appearances when
    /// `annotations`, on transparency when `alpha` (else white): (width, height, channels, pixels, undrawn), the
    /// pixels rows top-down, `undrawn` each kind of mark in the region not drawn as the file states it and how many.
    #[pyo3(signature = (number, region, scale, *, text, annotations, alpha))]
    fn render(
        &self,
        number: usize,
        region: Option<Box4>,
        scale: f64,
        text: bool,
        annotations: bool,
        alpha: bool,
    ) -> PyResult<RasterRow> {
        let page = self.at(number)?;
        let options = render::Options {
            region: region.map(|(x0, y0, x1, y1)| Rect { x0, y0, x1, y1 }),
            scale,
            text,
            annotations,
            alpha,
        };
        let raster = render::render(&self.file, page, &options).map_err(|e| refused(&e))?;
        Ok((
            raster.width,
            raster.height,
            raster.channels,
            raster.data,
            raster.undrawn.into_iter().collect(),
        ))
    }

    /// The image `XObject` `object`'s samples as 8-bit sRGB, three bytes a sample, rows top-down -- through its
    /// profile when it has one -- when they show as they stand (`pdf::image::Samples::rgb`); None when they do not.
    fn rgb(&self, object: u32) -> PyResult<Option<Vec<u8>>> {
        Ok(self.file.image(object).map_err(|e| refused(&e))?.rgb())
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

    /// Add page `number` (1-based) of `reader` as the next page, with `words` -- `(text, (x0, y0, x1, y1), angle,
    /// vertical)`, each box in page space as the word stands turned back level, `angle` the degrees its baseline is
    /// turned, clockwise as seen, `vertical` whether it is written down its box -- laid over it as an invisible text
    /// layer.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "pyo3 hands a list over owned"
    )]
    fn copy_page_with_text(
        &mut self,
        reader: &Reader,
        number: usize,
        words: Vec<(String, Box4, f64, bool)>,
    ) -> PyResult<()> {
        let page = reader.at(number)?;
        let words: Vec<pdf::writer::Word> = words
            .iter()
            .map(
                |(text, (x0, y0, x1, y1), angle, vertical)| pdf::writer::Word {
                    text,
                    bbox: Rect {
                        x0: *x0,
                        y0: *y0,
                        x1: *x1,
                        y1: *y1,
                    },
                    angle: *angle,
                    vertical: *vertical,
                },
            )
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
    fn targets(&self) -> &Targets {
        &self.targets.get_or_init(|| self.read_targets()).0
    }

    fn read_targets(&self) -> (Targets, Vec<String>) {
        let mut notes = Vec::new();
        let targets = Targets::new(&self.file, &self.pages, &mut notes);
        (targets, notes)
    }

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

/// A destination as its row, its page numbered from 1.
fn destination(d: &Destination) -> DestinationRow {
    (
        d.page + 1,
        d.fit.clone(),
        d.x,
        d.y,
        d.zoom,
        d.rect.map(box4),
    )
}

/// An action as its row.
fn action(a: &Action) -> ActionRow {
    (
        a.kind.clone(),
        a.destination.as_ref().map(destination),
        a.uri.clone(),
        a.file.clone(),
        a.name.clone(),
    )
}

/// `items` at `depth` and every item below them, in document order, into `out`.
fn flatten(items: &[Item], depth: usize, out: &mut Vec<ItemRow>) {
    for item in items {
        out.push((
            depth,
            item.title.clone(),
            item.destination.as_ref().map(destination),
            item.action.as_ref().map(action),
        ));
        flatten(&item.children, depth + 1, out);
    }
}

fn metadata(m: Metadata) -> MetadataRow {
    let xmp = m.xmp.map(|x| {
        (
            x.title,
            x.creators,
            x.description,
            x.created,
            x.modified,
            x.metadata_date,
            x.creator_tool,
            x.producer,
            x.keywords,
            x.packet,
        )
    });
    let newer = m.newer.map(|n| match n {
        Newer::Info => "info",
        Newer::Xmp => "xmp",
        Newer::Same => "same",
    });
    (m.info, xmp, newer)
}

fn field(f: Field) -> FieldRow {
    let value = |v: Value| match v {
        Value::Text(s) | Value::Name(s) => ValueRow::One(s),
        Value::List(l) => ValueRow::Several(l),
    };
    (
        f.name,
        f.kind,
        f.flags,
        f.value.map(value),
        f.default.map(value),
        f.options,
        f.widgets
            .into_iter()
            .map(|w| (w.page.map(|p| p + 1), w.rect.map(box4), w.state, w.export))
            .collect(),
    )
}

/// A layer as its row.
fn layer(l: &Layer) -> LayerRow {
    (l.hidden, l.inkless, l.annotation.clone())
}

/// Marked content as its row.
fn marked(m: Marked) -> MarkedRow {
    (m.mcid, m.stream, m.artifact)
}

/// A structure element as its row.
fn element(e: &Element) -> ElementRow {
    (
        e.role.clone(),
        e.standard.clone(),
        e.title.clone(),
        e.lang.clone(),
        e.alt.clone(),
        e.actual_text.clone(),
        e.expansion.clone(),
        e.attributes
            .iter()
            .map(|a| (a.owner.clone(), a.name.clone(), value(&a.value)))
            .collect(),
        e.page,
        e.kids
            .iter()
            .map(|k| match *k {
                Kid::Element(i) => ("element", Some(i), None, None, None),
                Kid::Marked { page, mcid, stream } => {
                    ("marked", Some(page), Some(mcid), stream, None)
                }
                Kid::Object { page, object, rect } => {
                    ("object", page, None, Some(object), rect.map(box4))
                }
            })
            .collect(),
    )
}

/// An attribute's value as Python's.
fn value(v: &StructureValue) -> AttributeValue {
    match v {
        StructureValue::Bool(b) => AttributeValue::Bool(*b),
        StructureValue::Integer(i) => AttributeValue::Integer(*i),
        StructureValue::Real(r) => AttributeValue::Real(*r),
        StructureValue::Name(t) | StructureValue::Text(t) => AttributeValue::Text(t.clone()),
        StructureValue::Array(items) => AttributeValue::Array(items.iter().map(value).collect()),
        StructureValue::Dict(entries) => {
            AttributeValue::Pairs(entries.iter().map(|(k, v)| (k.clone(), value(v))).collect())
        }
    }
}

/// A pattern's kind, by name.
fn pattern(p: Pattern) -> &'static str {
    match p {
        Pattern::Tiling { .. } => "tiling",
        Pattern::Shading => "shading",
    }
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
        path.fill
            .as_ref()
            .and_then(|f| f.color.pattern)
            .map(pattern),
        layer(&path.layer),
        marked(path.marked),
    )
}

/// The closed rectangle of `r`, as a path's segments.
fn rectangle(r: Rect) -> [Segment; 5] {
    let corner = |x, y| Point { x, y };
    [
        Segment::MoveTo(corner(r.x0, r.y0)),
        Segment::LineTo(corner(r.x1, r.y0)),
        Segment::LineTo(corner(r.x1, r.y1)),
        Segment::LineTo(corner(r.x0, r.y1)),
        Segment::Close,
    ]
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
