//! The image-page writer: a PDF written as its pages come -- each an image filling the page, or a page copied
//! whole from a PDF this crate reads, perhaps with an invisible text layer laid over it -- so a document of any
//! length costs one page of memory. [`Writer::finish`] adds the text layers' font, the page tree, the catalog and
//! the cross-reference table (ISO 32000-2 sections 7.5 and 7.7).
//!
//! An image is handed in as stored: its samples, encoded or not, and the name of the filter they are encoded with;
//! it is written as given, as is every stream of a copied page. The streams the writer builds itself -- a page's
//! drawing, a text layer, its font's `/ToUnicode` -- it compresses (`FlateDecode`, zlib) at the [`Writer`]'s level. <!-- spelling-ignore-line: the PDF name -->
//!
//! A text layer ([`Writer::copy_page_with_text`]) shows each word in render mode 3 -- invisible -- stretched over
//! the box it is seen in, so selecting and reading the page find it there. Its font draws nothing and embeds
//! nothing: a `CIDFontType2` whose every glyph is half an em wide and steps an em down a column (Tesseract's
//! `GlyphLessFont`); each character met is given the next CID, and the `/ToUnicode` maps the CIDs back, so Latin,
//! CJK and any other script are laid alike. A word written horizontally is shown through the `Type0` font over it
//! by `Identity-H`, along the box's bottom; one written vertically (CJK set in columns) by `Identity-V`, writing
//! mode 1 (ISO 32000-2 9.7.4.3): its characters stand upright and step down the box, so a reader reads it as
//! vertical writing, not as text turned sideways. The fonts serve the whole document, written when it is finished.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::io::{self, Write};

use crate::error::{Error, Kind, Result};
use crate::file::File;
use crate::geometry::{Matrix, Rect};
use crate::object::{Dict, ObjRef, Object};
use crate::page::Page;

/// The catalog's object number.
const CATALOG: u32 = 1;
/// The page tree's object number: every page's parent, written when the document is finished.
const PAGES: u32 = 2;
/// The image's name in a written page's resources.
const IMAGE: &[u8] = b"Im0";
/// The text layer's font's name, as Tesseract names a font that draws nothing.
const GLYPHLESS: &[u8] = b"GlyphLessFont";
/// Every glyph's advance in the text layer's font, in thousandths of an em.
const ADVANCE: f64 = 500.0;
/// The most `bfchar` entries one block of a `CMap` may hold.
const BFCHAR_BLOCK: usize = 100;
/// The compression level callers make a [`Writer`] with unless they choose another: the knee measured on dense OCR
/// layers (650 Latin words, 45 KB of streams; 1800 CJK words, 134 KB): level 1 leaves them 34% and 30% of their size,
/// 6 leaves 24% and 25% for 1-2 ms more a page, and 9 or 10 save at most 2.3% more for 3 to 5 times 6's time.
pub const LEVEL: u8 = 6;

/// A word of a text layer: its text, the box it is seen in, in page space (points, y down, origin at the top
/// left of the page as displayed), and how it is written.
#[derive(Debug, Clone, Copy)]
pub struct Word<'a> {
    /// The text.
    pub text: &'a str,
    /// The box, in page space.
    pub bbox: Rect,
    /// Whether it is written vertically: its characters stand upright down the box, read top to bottom.
    pub vertical: bool,
}

/// The text layers' fonts: the object number of each `Type0` font -- `Identity-H` written horizontally,
/// `Identity-V` vertically -- reserved when a word first needs it, and each character's CID (its place + 1), which
/// both share.
struct LayerFont {
    numbers: [Option<u32>; 2],
    cids: HashMap<char, u16>,
    chars: Vec<char>,
}

/// An image as stored in the file: its samples, and the filter they are encoded with.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    /// Samples across.
    pub width: u32,
    /// Samples down.
    pub height: u32,
    /// Its color space: `DeviceGray`, `DeviceRGB` or `DeviceCMYK`.
    pub space: &'a [u8],
    /// Bits per sample component: 1, 2, 4, 8 or 16.
    pub bits: u8,
    /// The filter `data` is encoded with (`FlateDecode`, `DCTDecode`, ...); None for raw samples. <!-- spelling-ignore-line: the PDF name -->
    pub filter: Option<&'a [u8]>,
    /// The samples as stored.
    pub data: &'a [u8],
}

/// A PDF being written to `W`, page by page.
pub struct Writer<W: Write> {
    out: W,
    /// Bytes written so far: the next object's offset.
    written: u64,
    /// Each object's offset by number - 1; None while reserved and not yet written.
    offsets: Vec<Option<u64>>,
    /// The pages' object numbers, in order.
    kids: Vec<u32>,
    /// The text layers' font, once a layer is laid.
    font: Option<LayerFont>,
    /// The zlib level its own streams are compressed at: 0 (stored) to 10.
    level: u8,
}

impl<W: Write> Writer<W> {
    /// Start a PDF on `out`: its header. The streams it builds are compressed at zlib `level`: 0 (stored, no
    /// compression) to 10 (smallest, slowest); [`LEVEL`] is the measured balance.
    ///
    /// # Errors
    ///
    /// Refuses a level beyond 10, and `out` that cannot be written.
    pub fn new(out: W, level: u8) -> Result<Self> {
        if level > 10 {
            return Err(Error::invalid(format!(
                "a compression level is 0 to 10, not {level}"
            )));
        }
        let mut writer = Self {
            out,
            written: 0,
            offsets: vec![None, None],
            kids: Vec::new(),
            font: None,
            level,
        };
        writer.put(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n")?;
        Ok(writer)
    }

    /// How many pages were added.
    #[must_use]
    pub fn pages(&self) -> usize {
        self.kids.len()
    }

    /// Add a page `size` points (width, height) showing `image` over all of it.
    ///
    /// # Errors
    ///
    /// Refuses a color space or bit depth no image may have, a size that is not positive, and a failed write.
    pub fn image_page(&mut self, image: &Image, size: (f64, f64)) -> Result<()> {
        if !matches!(image.space, b"DeviceGray" | b"DeviceRGB" | b"DeviceCMYK") {
            return Err(Error::invalid(format!(
                "an image page is DeviceGray, DeviceRGB or DeviceCMYK, not {}",
                String::from_utf8_lossy(image.space)
            )));
        }
        if !matches!(image.bits, 1 | 2 | 4 | 8 | 16) {
            return Err(Error::invalid(format!(
                "an image has 1, 2, 4, 8 or 16 bits a component, not {}",
                image.bits
            )));
        }
        let (width, height) = size;
        if !(width.is_finite() && height.is_finite() && width > 0.0 && height > 0.0) {
            return Err(Error::invalid(format!(
                "a page is a positive size, not {width} x {height}"
            )));
        }
        let (xobject, content, page) = (self.reserve(), self.reserve(), self.reserve());
        let mut dict = Dict::new();
        dict.insert(b"Type".to_vec(), name(b"XObject"));
        dict.insert(b"Subtype".to_vec(), name(b"Image"));
        dict.insert(b"Width".to_vec(), Object::Integer(image.width.into()));
        dict.insert(b"Height".to_vec(), Object::Integer(image.height.into()));
        dict.insert(b"ColorSpace".to_vec(), name(image.space));
        dict.insert(
            b"BitsPerComponent".to_vec(),
            Object::Integer(image.bits.into()),
        );
        if let Some(filter) = image.filter {
            dict.insert(b"Filter".to_vec(), name(filter));
        }
        self.stream(xobject, &dict, image.data)?;
        let draw = format!("q {} 0 0 {} 0 0 cm /Im0 Do Q", real(width), real(height));
        self.own_stream(content, draw.as_bytes())?;
        let mut images = Dict::new();
        images.insert(IMAGE.to_vec(), reference(xobject));
        let mut resources = Dict::new();
        resources.insert(b"XObject".to_vec(), Object::Dict(images));
        let mut dict = Dict::new();
        dict.insert(b"Type".to_vec(), name(b"Page"));
        dict.insert(b"Parent".to_vec(), reference(PAGES));
        dict.insert(b"MediaBox".to_vec(), numbers(&[0.0, 0.0, width, height]));
        dict.insert(b"Resources".to_vec(), Object::Dict(resources));
        dict.insert(b"Contents".to_vec(), reference(content));
        self.object(page, &Object::Dict(dict))?;
        self.kids.push(page);
        Ok(())
    }

    /// Add `page` of `file` as the next page: its dictionary with what it inherits made its own, and every object
    /// it reaches, renumbered. A reference to another page or to the page tree (a link's target) becomes null: only
    /// this page is copied.
    ///
    /// # Errors
    ///
    /// Refuses an object of `file` that cannot be read, and a failed write.
    pub fn copy_page(&mut self, file: &File, page: &Page) -> Result<()> {
        self.copy(file, page, &[])
    }

    /// Add `page` of `file` as [`Writer::copy_page`] does, with `words` laid over it as an invisible text layer.
    /// What the page draws is drawn first within a saved graphics state, so nothing it leaves changed moves a word.
    ///
    /// # Errors
    ///
    /// Refuses a word without text or without a box of positive size, besides what [`Writer::copy_page`] refuses.
    pub fn copy_page_with_text(&mut self, file: &File, page: &Page, words: &[Word]) -> Result<()> {
        for word in words {
            let Rect { x0, y0, x1, y1 } = word.bbox;
            if word.text.is_empty()
                || !([x0, y0, x1, y1].iter().all(|v| v.is_finite()) && x1 > x0 && y1 > y0)
            {
                return Err(Error::invalid(format!(
                    "a word of a text layer has text and a box of positive size, not {:?} at [{x0} {y0} {x1} {y1}]",
                    word.text
                )));
            }
        }
        self.copy(file, page, words)
    }

    /// `page` of `file` copied, with `words` laid over it when there are any.
    fn copy(&mut self, file: &File, page: &Page, words: &[Word]) -> Result<()> {
        let layered = !words.is_empty();
        let mut own = Dict::new();
        for (key, value) in page.dict.iter() {
            let replaced = matches!(
                key,
                b"Parent" | b"Resources" | b"MediaBox" | b"CropBox" | b"Rotate"
            ) || (layered && key == b"Contents");
            if !replaced {
                own.insert(key.to_vec(), value.clone());
            }
        }
        let mut resources = page.resources.clone();
        if layered {
            // Made direct here, so the layer's font and content are added to the copies after renumbering.
            let fonts = match file.get(&resources, b"Font")? {
                Some(Object::Dict(fonts)) => fonts,
                _ => Dict::new(),
            };
            resources = replaced(&resources, b"Font", Object::Dict(fonts));
            let drawn = match page.dict.get(b"Contents") {
                None => Vec::new(),
                Some(Object::Reference(r)) => match file.object(r.num)? {
                    Object::Array(streams) => streams,
                    _ => vec![Object::Reference(*r)],
                },
                Some(Object::Array(streams)) => streams.clone(),
                Some(other) => vec![other.clone()],
            };
            own.insert(b"Contents".to_vec(), Object::Array(drawn));
        }
        own.insert(b"Resources".to_vec(), Object::Dict(resources));
        own.insert(b"MediaBox".to_vec(), rect(page.media_box));
        own.insert(b"CropBox".to_vec(), rect(page.crop_box));
        own.insert(
            b"Rotate".to_vec(),
            Object::Integer(i64::from(page.quarter_turns) * 90),
        );
        let number = self.reserve();
        let mut copy = Copy {
            file,
            numbers: HashMap::new(),
            queue: Vec::new(),
        };
        let Object::Dict(mut own) = copy.renumber(self, Object::Dict(own), 0)? else {
            unreachable!("a dictionary renumbers to a dictionary")
        };
        while let Some((at, value)) = copy.queue.pop() {
            let value = copy.renumber(self, value, 0)?;
            self.object(at, &value)?;
        }
        if layered {
            own = self.lay(&own, page, words)?;
        }
        own.insert(b"Parent".to_vec(), reference(PAGES));
        self.object(number, &Object::Dict(own))?;
        self.kids.push(number);
        Ok(())
    }

    /// `own`, a copied page whose resources and contents are direct, with `words` shown over what it draws: its
    /// contents saved before and restored after, then a stream showing each word, invisible, stretched over its box.
    fn lay(&mut self, own: &Dict, page: &Page, words: &[Word]) -> Result<Dict> {
        let resources = own
            .get(b"Resources")
            .and_then(Object::as_dict)
            .cloned()
            .unwrap_or_else(Dict::new);
        let mut fonts = resources
            .get(b"Font")
            .and_then(Object::as_dict)
            .cloned()
            .unwrap_or_else(Dict::new);
        self.font.get_or_insert_with(|| LayerFont {
            numbers: [None, None],
            cids: HashMap::new(),
            chars: Vec::new(),
        });
        // A resource name for each way the words are written, free on the page: of two more names than there are
        // fonts, two are free.
        let mut free = (0..fonts.len() + 2)
            .map(|i| format!("OCR{i}").into_bytes())
            .filter(|n| fonts.get(n).is_none())
            .collect::<Vec<_>>()
            .into_iter();
        let mut names: [Option<Vec<u8>>; 2] = [None, None];
        for way in [false, true] {
            if words.iter().any(|w| w.vertical == way) {
                let known = self.font.as_ref().and_then(|f| f.numbers[usize::from(way)]);
                let number = if let Some(number) = known {
                    number
                } else {
                    let number = self.reserve();
                    let font = self.font.as_mut().expect("made above");
                    font.numbers[usize::from(way)] = Some(number);
                    number
                };
                let resource = free.next().expect("a free name");
                fonts.insert(resource.clone(), reference(number));
                names[usize::from(way)] = Some(resource);
            }
        }
        let font = self.font.as_mut().expect("made above");
        let mut shows = String::from("Q\nBT\n3 Tr\n");
        let mut writing = None; // the way the font set writes
        let to_user = inverse(page.to_page_space());
        for word in words {
            let cids = word
                .text
                .chars()
                .map(|c| font.cid(c))
                .collect::<Result<Vec<_>>>()?;
            if writing != Some(word.vertical) {
                let resource = names[usize::from(word.vertical)]
                    .as_ref()
                    .expect("named above");
                let _ = writeln!(shows, "/{} 1 Tf", String::from_utf8_lossy(resource));
                writing = Some(word.vertical);
            }
            #[expect(clippy::cast_precision_loss, reason = "a word's length is small")]
            let length = cids.len() as f64;
            let Rect { x0, y0, x1, y1 } = word.bbox;
            // Text space, font size 1, onto the box. Horizontally: the baseline along its bottom, an em its height,
            // the advances its width. Vertically: the first glyph's vertical origin at its top center, an em its
            // width, its height shared out among the em steps down.
            let onto = if word.vertical {
                Matrix([
                    x1 - x0,
                    0.0,
                    0.0,
                    (y0 - y1) / length,
                    f64::midpoint(x0, x1),
                    y0,
                ])
            } else {
                Matrix([
                    (x1 - x0) / (length * ADVANCE / 1000.0),
                    0.0,
                    0.0,
                    y0 - y1,
                    x0,
                    y1,
                ])
            };
            let Matrix(m) = onto.then(to_user);
            let m: Vec<String> = m.iter().map(|&v| real(v)).collect();
            let _ = write!(shows, "{} Tm <", m.join(" "));
            for cid in cids {
                let _ = write!(shows, "{cid:04X}");
            }
            shows.push_str("> Tj\n");
        }
        shows.push_str("ET\n");
        let (save, layer) = (self.reserve(), self.reserve());
        self.own_stream(save, b"q\n")?;
        self.own_stream(layer, shows.as_bytes())?;
        let mut contents = vec![reference(save)];
        if let Some(drawn) = own.get(b"Contents").and_then(Object::as_array) {
            contents.extend_from_slice(drawn);
        }
        contents.push(reference(layer));
        let resources = replaced(&resources, b"Font", Object::Dict(fonts));
        let own = replaced(own, b"Resources", Object::Dict(resources));
        Ok(replaced(&own, b"Contents", Object::Array(contents)))
    }

    /// The text layers' fonts: each `Type0` font a layer used -- `Identity-H`, `Identity-V` -- over one
    /// `CIDFontType2` that draws nothing, every glyph half an em wide and stepping an em down a column, and one
    /// `/ToUnicode` naming each CID's character.
    fn write_font(&mut self, font: &LayerFont) -> Result<()> {
        let (descendant, descriptor, to_unicode) = (self.reserve(), self.reserve(), self.reserve());
        let mut cmap = String::from(
            "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n\
             /CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n\
             /CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n\
             1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n",
        );
        for (block, chars) in font.chars.chunks(BFCHAR_BLOCK).enumerate() {
            let _ = writeln!(cmap, "{} beginbfchar", chars.len());
            for (i, c) in chars.iter().enumerate() {
                let _ = write!(cmap, "<{:04X}> <", block * BFCHAR_BLOCK + i + 1);
                for unit in c.encode_utf16(&mut [0; 2]) {
                    let _ = write!(cmap, "{unit:04X}");
                }
                cmap.push_str(">\n");
            }
            cmap.push_str("endbfchar\n");
        }
        cmap.push_str("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n");
        self.own_stream(to_unicode, cmap.as_bytes())?;
        let mut dict = Dict::new();
        dict.insert(b"Type".to_vec(), name(b"FontDescriptor"));
        dict.insert(b"FontName".to_vec(), name(GLYPHLESS));
        dict.insert(b"Flags".to_vec(), Object::Integer(5)); // fixed pitch, symbolic
        dict.insert(b"FontBBox".to_vec(), numbers(&[0.0, 0.0, ADVANCE, 1000.0]));
        dict.insert(b"ItalicAngle".to_vec(), Object::Integer(0));
        dict.insert(b"Ascent".to_vec(), Object::Integer(1000));
        dict.insert(b"Descent".to_vec(), Object::Integer(0));
        dict.insert(b"CapHeight".to_vec(), Object::Integer(1000));
        dict.insert(b"StemV".to_vec(), Object::Integer(80));
        self.object(descriptor, &Object::Dict(dict))?;
        let mut system = Dict::new();
        system.insert(b"Registry".to_vec(), Object::String(b"Adobe".to_vec()));
        system.insert(b"Ordering".to_vec(), Object::String(b"Identity".to_vec()));
        system.insert(b"Supplement".to_vec(), Object::Integer(0));
        let mut dict = Dict::new();
        dict.insert(b"Type".to_vec(), name(b"Font"));
        dict.insert(b"Subtype".to_vec(), name(b"CIDFontType2"));
        dict.insert(b"BaseFont".to_vec(), name(GLYPHLESS));
        dict.insert(b"CIDSystemInfo".to_vec(), Object::Dict(system));
        dict.insert(b"FontDescriptor".to_vec(), reference(descriptor));
        dict.insert(b"DW".to_vec(), Object::Real(ADVANCE));
        // Written vertically: the origin 880 above the glyph... and an em's step down; the default, stated.
        dict.insert(b"DW2".to_vec(), numbers(&[880.0, -1000.0]));
        dict.insert(b"CIDToGIDMap".to_vec(), name(b"Identity"));
        self.object(descendant, &Object::Dict(dict))?;
        for (number, encoding) in font.numbers.iter().zip([b"Identity-H", b"Identity-V"]) {
            let Some(number) = *number else { continue };
            let mut dict = Dict::new();
            dict.insert(b"Type".to_vec(), name(b"Font"));
            dict.insert(b"Subtype".to_vec(), name(b"Type0"));
            dict.insert(b"BaseFont".to_vec(), name(GLYPHLESS));
            dict.insert(b"Encoding".to_vec(), name(encoding));
            dict.insert(
                b"DescendantFonts".to_vec(),
                Object::Array(vec![reference(descendant)]),
            );
            dict.insert(b"ToUnicode".to_vec(), reference(to_unicode));
            self.object(number, &Object::Dict(dict))?;
        }
        Ok(())
    }

    /// Write the text layers' font if a layer was laid, the page tree, the catalog and the cross-reference table,
    /// and hand `out` back.
    ///
    /// # Errors
    ///
    /// Refuses when `out` cannot be written.
    pub fn finish(mut self) -> Result<W> {
        if let Some(font) = self.font.take() {
            self.write_font(&font)?;
        }
        let mut tree = Dict::new();
        tree.insert(b"Type".to_vec(), name(b"Pages"));
        tree.insert(
            b"Kids".to_vec(),
            Object::Array(self.kids.iter().map(|&k| reference(k)).collect()),
        );
        tree.insert(
            b"Count".to_vec(),
            Object::Integer(i64::try_from(self.kids.len()).unwrap_or(i64::MAX)),
        );
        self.object(PAGES, &Object::Dict(tree))?;
        let mut catalog = Dict::new();
        catalog.insert(b"Type".to_vec(), name(b"Catalog"));
        catalog.insert(b"Pages".to_vec(), reference(PAGES));
        self.object(CATALOG, &Object::Dict(catalog))?;
        let start = self.written;
        let size = self.offsets.len() + 1;
        let mut table = format!("xref\n0 {size}\n0000000000 65535 f \n");
        for offset in &self.offsets {
            // An object reserved and never written (a copy that failed part way) is free.
            table.push_str(&offset.map_or_else(
                || "0000000000 00001 f \n".to_owned(),
                |o| format!("{o:010} 00000 n \n"),
            ));
        }
        self.put(table.as_bytes())?;
        let trailer =
            format!("trailer\n<< /Size {size} /Root {CATALOG} 0 R >>\nstartxref\n{start}\n%%EOF\n");
        self.put(trailer.as_bytes())?;
        self.out.flush().map_err(|e| failed(&e))?;
        Ok(self.out)
    }

    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        self.out.write_all(bytes).map_err(|e| failed(&e))?;
        self.written += bytes.len() as u64;
        Ok(())
    }

    fn reserve(&mut self) -> u32 {
        self.offsets.push(None);
        u32::try_from(self.offsets.len()).expect("fewer than 2^32 objects")
    }

    fn begin(&mut self, number: u32) -> Result<()> {
        self.offsets[number as usize - 1] = Some(self.written);
        self.put(format!("{number} 0 obj\n").as_bytes())
    }

    fn object(&mut self, number: u32, value: &Object) -> Result<()> {
        if let Object::Stream(s) = value {
            return self.stream(number, &s.dict, &s.raw);
        }
        self.begin(number)?;
        let mut bytes = Vec::new();
        serialize(value, &mut bytes);
        bytes.extend_from_slice(b"\nendobj\n");
        self.put(&bytes)
    }

    /// A stream the writer built: `data` compressed, `/FlateDecode`. <!-- spelling-ignore-line: the PDF name -->
    fn own_stream(&mut self, number: u32, data: &[u8]) -> Result<()> {
        let mut dict = Dict::new();
        dict.insert(b"Filter".to_vec(), name(b"FlateDecode")); // spelling-ignore-line: the PDF name
        let deflated = miniz_oxide::deflate::compress_to_vec_zlib(data, self.level);
        self.stream(number, &dict, &deflated)
    }

    /// A stream object: `dict` with its `/Length` made `data`'s, then `data` as it is.
    fn stream(&mut self, number: u32, dict: &Dict, data: &[u8]) -> Result<()> {
        self.begin(number)?;
        let mut own = Dict::new();
        for (key, value) in dict.iter().filter(|(k, _)| *k != b"Length") {
            own.insert(key.to_vec(), value.clone());
        }
        own.insert(
            b"Length".to_vec(),
            Object::Integer(i64::try_from(data.len()).expect("a stream fits")),
        );
        let mut bytes = Vec::new();
        serialize(&Object::Dict(own), &mut bytes);
        bytes.extend_from_slice(b"\nstream\n");
        self.put(&bytes)?;
        self.put(data)?;
        self.put(b"\nendstream\nendobj\n")
    }
}

/// One page's copy: the source file, its object numbers mapped to new ones, and the objects still to write.
struct Copy<'a> {
    file: &'a File,
    numbers: HashMap<u32, u32>,
    queue: Vec<(u32, Object)>,
}

impl Copy<'_> {
    /// `value` with each reference to the source made one to its copy, the copy reserved and queued when new.
    fn renumber<W: Write>(
        &mut self,
        writer: &mut Writer<W>,
        value: Object,
        depth: usize,
    ) -> Result<Object> {
        if depth > 256 {
            return Err(Error::malformed("objects nested too deep"));
        }
        Ok(match value {
            Object::Reference(r) => self.target(writer, r)?,
            Object::Array(items) => Object::Array(
                items
                    .into_iter()
                    .map(|v| self.renumber(writer, v, depth + 1))
                    .collect::<Result<_>>()?,
            ),
            Object::Dict(dict) => Object::Dict(self.dict(writer, &dict, depth, false)?),
            Object::Stream(mut stream) => {
                stream.dict = self.dict(writer, &stream.dict, depth, true)?;
                Object::Stream(stream)
            }
            other => other,
        })
    }

    /// `dict` renumbered; a stream's (`stream`) without its `/Length`, which the writer states: the source's own,
    /// perhaps a reference, is not copied.
    fn dict<W: Write>(
        &mut self,
        writer: &mut Writer<W>,
        dict: &Dict,
        depth: usize,
        stream: bool,
    ) -> Result<Dict> {
        let mut out = Dict::new();
        for (key, value) in dict.iter().filter(|(k, _)| !(stream && *k == b"Length")) {
            out.insert(
                key.to_vec(),
                self.renumber(writer, value.clone(), depth + 1)?,
            );
        }
        Ok(out)
    }

    fn target<W: Write>(&mut self, writer: &mut Writer<W>, r: ObjRef) -> Result<Object> {
        if let Some(&n) = self.numbers.get(&r.num) {
            return Ok(reference(n));
        }
        let value = self.file.object(r.num)?;
        let kind = value
            .as_dict()
            .and_then(|d| d.get(b"Type"))
            .and_then(Object::as_name);
        if matches!(kind, Some(b"Page" | b"Pages")) {
            return Ok(Object::Null);
        }
        let n = writer.reserve();
        self.numbers.insert(r.num, n);
        self.queue.push((n, value));
        Ok(reference(n))
    }
}

impl LayerFont {
    /// `c`'s CID: its own, or the next when first met.
    fn cid(&mut self, c: char) -> Result<u16> {
        if let Some(&cid) = self.cids.get(&c) {
            return Ok(cid);
        }
        let cid = u16::try_from(self.chars.len() + 1).map_err(|_| {
            Error::invalid("a text layer's font names at most 65535 characters, and more were met")
        })?;
        self.cids.insert(c, cid);
        self.chars.push(c);
        Ok(cid)
    }
}

/// The matrix undoing `m`, which turns, flips and moves (a page's, never singular).
fn inverse(m: Matrix) -> Matrix {
    let [xx, xy, yx, yy, dx, dy] = m.0;
    let det = xx * yy - xy * yx;
    Matrix([
        yy / det,
        -xy / det,
        -yx / det,
        xx / det,
        (yx * dy - yy * dx) / det,
        (xy * dx - xx * dy) / det,
    ])
}

/// `dict` with `key`'s value `value`, added where it had none.
fn replaced(dict: &Dict, key: &[u8], value: Object) -> Dict {
    let mut out = Dict::new();
    for (k, v) in dict.iter().filter(|(k, _)| *k != key) {
        out.insert(k.to_vec(), v.clone());
    }
    out.insert(key.to_vec(), value);
    out
}

fn failed(e: &io::Error) -> Error {
    Error::new(Kind::Io, format!("writing the PDF: {e}"))
}

fn name(n: &[u8]) -> Object {
    Object::Name(n.to_vec())
}

fn reference(num: u32) -> Object {
    Object::Reference(ObjRef { num, generation: 0 })
}

fn numbers(values: &[f64]) -> Object {
    Object::Array(values.iter().map(|&v| Object::Real(v)).collect())
}

fn rect(r: Rect) -> Object {
    numbers(&[r.x0, r.y0, r.x1, r.y1])
}

/// A number as PDF writes it: an integer without a point, a real in plain decimals (never an exponent).
fn real(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_owned();
    }
    #[expect(clippy::cast_possible_truncation, reason = "a whole number under 2^53")]
    if v.fract() == 0.0 && v.abs() < 9e15 {
        return format!("{}", v as i64);
    }
    format!("{v}")
}

/// `value` in PDF syntax: a string as hex, a name with every byte it cannot hold as it is `#`-escaped.
fn serialize(value: &Object, out: &mut Vec<u8>) {
    match value {
        Object::Null | Object::Stream(_) => out.extend_from_slice(b"null"), // a stream is never a direct value
        Object::Bool(b) => out.extend_from_slice(if *b { b"true" } else { b"false" }),
        Object::Integer(i) => out.extend_from_slice(i.to_string().as_bytes()),
        Object::Real(r) => out.extend_from_slice(real(*r).as_bytes()),
        Object::String(s) => {
            out.push(b'<');
            for b in s {
                out.extend_from_slice(format!("{b:02X}").as_bytes());
            }
            out.push(b'>');
        }
        Object::Name(n) => {
            out.push(b'/');
            for &b in n {
                if (0x21..=0x7E).contains(&b) && !b"#()<>[]{}/%".contains(&b) {
                    out.push(b);
                } else {
                    out.extend_from_slice(format!("#{b:02X}").as_bytes());
                }
            }
        }
        Object::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b' ');
                }
                serialize(item, out);
            }
            out.push(b']');
        }
        Object::Dict(dict) => {
            out.extend_from_slice(b"<<");
            for (key, item) in dict.iter() {
                out.push(b' ');
                serialize(&Object::Name(key.to_vec()), out);
                out.push(b' ');
                serialize(item, out);
            }
            out.extend_from_slice(b" >>");
        }
        Object::Reference(r) => {
            out.extend_from_slice(format!("{} {} R", r.num, r.generation).as_bytes());
        }
    }
}
