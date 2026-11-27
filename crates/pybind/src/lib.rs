//! `document_lab._pdf`: the `pdf` crate, exposed to Python as plain values -- no PDF logic of its own.
//!
//! `open(data, password)` reads a file; its `page(number, curve_samples, annotations)` returns what the page draws
//! as tuples, which `document_lab.pdf` turns into the IR, its `links(number)` the page's links, its `document()` what
//! the document states beside its pages (outline, labels, metadata, form fields, embedded files), its
//! `image(object)` an image's samples and its `attachment(object)` an embedded file's bytes. Pages are numbered from
//! 1 in every row. `create(path)` writes a PDF of image pages, or of pages copied from a `Reader`
//! (`document_lab.capture`).
//!
//! A refusal raises by its kind: [`Refused`] and its kinds when the reader refuses the input, `ValueError` for a
//! request it cannot honor, `OSError` when writing fails.

use std::cell::OnceCell;
use std::fs;
use std::io::BufWriter;
use std::path::PathBuf;

use pdf::color::Pattern;
use pdf::destination::{Action, Destination, Targets};
use pdf::error::Kind;
use pdf::forms::{Field, Value};
use pdf::geometry::Rect;
use pdf::interpret::{Event, Layer, PathEvent};
use pdf::metadata::{Metadata, Newer};
use pdf::outline::Item;
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
/// The layer an event is drawn in: hidden by optional content, and the subtype of the annotation drawing it.
type LayerRow = (bool, Option<String>);
/// A span: text, font name (no subset tag), size, bold, box, render mode, the number it shares with the spans its
/// words run on into with no space between (None: none), and its layer.
type SpanRow = (
    String,
    String,
    f64,
    bool,
    Box4,
    i64,
    Option<usize>,
    LayerRow,
);
/// A path (one subpath): points, closed, stroke color, line width, dashes, dash phase, fill color, even-odd, the
/// fill's pattern (`tiling`, `shading`; None: one color), the index of the painted path it belongs to on its
/// page, and its layer. A shading painted with `sh` is the closed rectangle of its box, filled with a `shading`.
type PathRow = (
    Vec<(f64, f64)>,
    bool,
    Option<Vec<f64>>,
    f64,
    Vec<f64>,
    f64,
    Option<Vec<f64>>,
    bool,
    Option<&'static str>,
    usize,
    LayerRow,
);
/// An image: box, sample width and height, object number (None inline), layer.
type ImageRow = (Box4, i64, i64, Option<u32>, LayerRow);
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

/// A PDF opened.
#[pyclass(unsendable)]
struct Reader {
    file: pdf::File,
    pages: Vec<pdf::Page>,
    /// What resolving destinations needs, and what reading it warned of: read when first needed.
    targets: OnceCell<(Targets, Vec<String>)>,
}

/// Open the PDF `data` with `password` (empty: none).
#[pyfunction]
fn open(data: Vec<u8>, password: &[u8]) -> PyResult<Reader> {
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

    /// Each page's number, width and height as displayed (points).
    #[getter]
    fn pages(&self) -> Vec<(usize, f64, f64)> {
        self.pages
            .iter()
            .map(|p| (p.number, p.size().0, p.size().1))
            .collect()
    }

    /// What page `number` (1-based) draws, its annotations' appearances too when `annotations`: (spans, paths,
    /// images, warnings); each curve flattened to `curve_samples` points.
    fn page(&self, number: usize, curve_samples: usize, annotations: bool) -> PyResult<PageRows> {
        let page = self.at(number)?;
        let content = self
            .file
            .interpret(page, annotations)
            .map_err(|e| refused(&e))?;
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
                    layer(&s.layer),
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
                        layer(&image.layer),
                    ));
                }
                Event::Shading(shading) => {
                    let r = shading.bbox;
                    let corners = vec![(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)];
                    paths.push((
                        corners,
                        true,
                        None,
                        0.0,
                        Vec::new(),
                        0.0,
                        Some(Vec::new()),
                        false,
                        Some(pattern(Pattern::Shading)),
                        painted,
                        layer(&shading.layer),
                    ));
                    painted += 1;
                }
                Event::Text(_) => {}
            }
        }
        Ok((spans, paths, images, content.warnings))
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
    (l.hidden, l.annotation.clone())
}

/// A pattern's kind, by name.
fn pattern(p: Pattern) -> &'static str {
    match p {
        Pattern::Tiling => "tiling",
        Pattern::Shading => "shading",
    }
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
                path.fill
                    .as_ref()
                    .and_then(|f| f.color.pattern)
                    .map(pattern),
                group,
                layer(&path.layer),
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
