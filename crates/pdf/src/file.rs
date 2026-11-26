//! A PDF opened: its objects resolved by reference, object streams included, a broken table repaired.

use std::cell::{OnceCell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::crypt::Decryptor;
use crate::error::{Error, Kind, Result};
use crate::filter::{self, Decoded};
use crate::object::{Dict, Object, Stream};
use crate::parser::{Parser, find};
use crate::xref::{self, Entry, Xref};

/// An object stream's members: (number, value) in stream order.
type Members = Vec<(u32, Object)>;

/// How many references a chain may follow before it is refused as a loop.
const MAX_CHAIN: usize = 32;

/// A PDF file's bytes and the table that finds each object in them.
pub struct File {
    data: Vec<u8>,
    xref: Xref,
    /// The table rebuilt by scanning, made the first time an object is not where the table says.
    scanned: OnceCell<Xref>,
    /// Each object stream's objects, decoded once: (number, value) in stream order.
    streams: RefCell<HashMap<u32, Rc<Members>>>,
    /// What was repaired to read this file, in words (empty for a sound file).
    repairs: RefCell<Vec<String>>,
    /// The decryption of an encrypted file, and its encryption dictionary's object number (never decrypted).
    decryptor: Option<(Decryptor, Option<u32>)>,
}

impl File {
    /// Open `data` with `password` (empty: none, as a file anyone may read needs): the cross-reference sections
    /// when they lead to a catalog, else the table rebuilt by a scan.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here, and an encrypted file `password` does not open.
    pub fn open(data: Vec<u8>, password: &[u8]) -> Result<Self> {
        if find(&data[..data.len().min(1024)], b"%PDF-").is_none() {
            return Err(Error::new(
                Kind::NotPdf,
                "not a PDF: no `%PDF-` header in the first 1024 bytes",
            ));
        }
        let (xref, repair) = match xref::read(&data) {
            Ok(x) if x.trailer.get(b"Root").is_some() => (x, None),
            Ok(_) => (
                xref::scan(&data),
                Some("the trailer names no /Root; the table was rebuilt by a scan".into()),
            ),
            Err(e) => (
                xref::scan(&data),
                Some(format!(
                    "the cross-reference sections are unreadable ({e}); rebuilt by a scan"
                )),
            ),
        };
        let mut file = Self {
            data,
            xref,
            scanned: OnceCell::new(),
            streams: RefCell::new(HashMap::new()),
            repairs: RefCell::new(repair.into_iter().collect()),
            decryptor: None,
        };
        if let Some(encrypt) = file.xref.trailer.get(b"Encrypt").cloned() {
            let dict = match file.resolve(&encrypt)? {
                Object::Dict(d) => d,
                other => {
                    return Err(Error::malformed(format!(
                        "/Encrypt is a {}, not a dictionary",
                        other.kind()
                    )));
                }
            };
            let first_id = match file.get(&file.xref.trailer, b"ID")? {
                Some(Object::Array(ids)) => {
                    match ids.first().map(|i| file.resolve(i)).transpose()? {
                        Some(Object::String(s)) => s,
                        _ => Vec::new(),
                    }
                }
                _ => Vec::new(),
            };
            let decryptor = Decryptor::new(&dict, &first_id, password)?;
            file.decryptor = Some((decryptor, encrypt.as_reference().map(|r| r.num)));
        }
        file.catalog()?;
        Ok(file)
    }

    /// Note a repair made to read this file.
    pub(crate) fn repaired(&self, what: String) {
        self.repairs.borrow_mut().push(what);
    }

    /// The trailer dictionary.
    pub fn trailer(&self) -> &Dict {
        &self.xref.trailer
    }

    /// What was repaired to read this file, in words; empty for a sound file.
    pub fn repairs(&self) -> Vec<String> {
        self.repairs.borrow().clone()
    }

    /// Every object number the table holds, free ones excluded.
    pub fn numbers(&self) -> Vec<u32> {
        self.xref
            .entries
            .iter()
            .filter(|(_, e)| **e != Entry::Free)
            .map(|(n, _)| *n)
            .collect()
    }

    /// The document catalog (`/Root`).
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn catalog(&self) -> Result<Dict> {
        let root = self
            .xref
            .trailer
            .get(b"Root")
            .ok_or_else(|| Error::malformed("the trailer names no /Root"))?;
        match self.resolve(root)? {
            Object::Dict(d) => Ok(d),
            other => Err(Error::malformed(format!(
                "/Root is a {}, not a dictionary",
                other.kind()
            ))),
        }
    }

    /// The object `num`; a free or missing object is null (7.3.10).
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn object(&self, num: u32) -> Result<Object> {
        match self.xref.entries.get(&num).copied().unwrap_or(Entry::Free) {
            Entry::Free => Ok(Object::Null),
            Entry::InFile { offset, .. } => match self.in_file(num, offset) {
                Ok(o) => Ok(o),
                Err(e) => self.found_by_scan(num, &e),
            },
            Entry::InStream { stream, index } => self.in_stream(num, stream, index),
        }
    }

    /// `value`, its references followed to a direct value.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn resolve(&self, value: &Object) -> Result<Object> {
        let mut current = value.clone();
        for _ in 0..MAX_CHAIN {
            match current {
                Object::Reference(r) => current = self.object(r.num)?,
                other => return Ok(other),
            }
        }
        Err(Error::malformed("a reference chain loops"))
    }

    /// `dict[key]`, resolved; None when absent or null.
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn get(&self, dict: &Dict, key: &[u8]) -> Result<Option<Object>> {
        match dict.get(key).map(|v| self.resolve(v)).transpose()? {
            Some(Object::Null) | None => Ok(None),
            some => Ok(some),
        }
    }

    /// A stream's data through its filters (references in `/Filter` and `/DecodeParms` resolved). <!-- spelling-ignore-line: the PDF name -->
    ///
    /// # Errors
    ///
    /// Refuses input that is not what a PDF must be here; the [`Error`] says what and where.
    pub fn decode(&self, stream: &Stream) -> Result<Decoded> {
        filter::decode(&stream.dict, &stream.raw, &|o| self.resolve(o))
    }

    fn length(&self, value: &Object) -> Option<usize> {
        match value {
            Object::Integer(v) => usize::try_from(*v).ok(),
            Object::Reference(r) => self
                .object(r.num)
                .ok()?
                .as_integer()
                .and_then(|v| usize::try_from(v).ok()),
            _ => None,
        }
    }

    fn in_file(&self, num: u32, offset: usize) -> Result<Object> {
        if offset >= self.data.len() {
            return Err(Error::malformed_at(
                offset,
                format!("object {num}: offset past the end of the file"),
            ));
        }
        let (id, mut value) =
            Parser::new(&self.data, offset).indirect_object(|o| self.length(o))?;
        if id.num != num {
            return Err(Error::malformed_at(
                offset,
                format!("object {num}: the table points at object {}", id.num),
            ));
        }
        if let Some((decryptor, own)) = &self.decryptor
            && *own != Some(num)
        {
            decryptor.object(id, &mut value)?;
        }
        Ok(value)
    }

    fn found_by_scan(&self, num: u32, cause: &Error) -> Result<Object> {
        let scanned = self.scanned.get_or_init(|| xref::scan(&self.data));
        match scanned.entries.get(&num) {
            Some(&Entry::InFile { offset, .. }) => {
                let value = self.in_file(num, offset)?;
                self.repairs.borrow_mut().push(format!(
                    "object {num}: {cause}; found by a scan at byte {offset}"
                ));
                Ok(value)
            }
            _ => Err(cause.clone()),
        }
    }

    fn in_stream(&self, num: u32, stream: u32, index: u32) -> Result<Object> {
        let objects = self.object_stream(stream)?;
        let hit = objects
            .get(index as usize)
            .filter(|(n, _)| *n == num)
            .or_else(|| objects.iter().find(|(n, _)| *n == num));
        hit.map(|(_, o)| o.clone()).ok_or_else(|| {
            Error::malformed(format!("object {num} is not in object stream {stream}"))
        })
    }

    fn object_stream(&self, stream: u32) -> Result<Rc<Members>> {
        if let Some(found) = self.streams.borrow().get(&stream) {
            return Ok(Rc::clone(found));
        }
        let Object::Stream(s) = self.object(stream)? else {
            return Err(Error::malformed(format!(
                "object {stream} is named an object stream but is no stream"
            )));
        };
        let data = self.decode(&s)?.into_bytes()?;
        let count = self
            .get(&s.dict, b"N")?
            .and_then(|o| o.as_integer())
            .unwrap_or(0);
        let first = self
            .get(&s.dict, b"First")?
            .and_then(|o| o.as_integer())
            .unwrap_or(0);
        let first = usize::try_from(first)
            .map_err(|_| Error::malformed("object stream /First out of range"))?;
        let mut header = Parser::new(&data, 0);
        let mut objects = Vec::new();
        for _ in 0..count {
            let n = u32::try_from(header.unsigned()?)
                .map_err(|_| Error::malformed("object number out of range"))?;
            let at = usize::try_from(header.unsigned()?)
                .map_err(|_| Error::malformed("offset out of range"))?;
            let value = Parser::new(&data, first + at).object()?;
            objects.push((n, value));
        }
        let objects = Rc::new(objects);
        self.streams
            .borrow_mut()
            .insert(stream, Rc::clone(&objects));
        Ok(objects)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PDF with a correct classic table, built with real offsets.
    fn classic(objects: &[&str], trailer: &str) -> Vec<u8> {
        let mut out = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (i, body) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", i + 1).as_bytes());
        }
        let at = out.len();
        out.extend_from_slice(
            format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
        );
        for o in &offsets {
            out.extend_from_slice(format!("{o:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(format!("trailer\n{trailer}\nstartxref\n{at}\n%%EOF\n").as_bytes());
        out
    }

    const CATALOG: &str = "<< /Type /Catalog /Pages 2 0 R >>";
    const PAGES: &str = "<< /Type /Pages /Kids [] /Count 0 >>";

    #[test]
    fn a_classic_table_resolves_references() {
        let file =
            File::open(classic(&[CATALOG, PAGES], "<< /Size 3 /Root 1 0 R >>"), b"").unwrap();
        let pages = file
            .get(&file.catalog().unwrap(), b"Pages")
            .unwrap()
            .unwrap();
        assert_eq!(
            pages.as_dict().unwrap().get(b"Count"),
            Some(&Object::Integer(0))
        );
        assert_eq!(file.repairs(), Vec::<String>::new());
        assert_eq!(file.object(99).unwrap(), Object::Null);
    }

    #[test]
    fn a_stream_length_given_by_reference_is_followed() {
        let data = classic(
            &[
                CATALOG,
                PAGES,
                "<< /Length 4 0 R >>\nstream\nabcdef\nendstream",
                "6",
            ],
            "<< /Size 5 /Root 1 0 R >>",
        );
        let file = File::open(data, b"").unwrap();
        let Object::Stream(s) = file.object(3).unwrap() else {
            panic!()
        };
        assert_eq!(s.raw, b"abcdef");
    }

    #[test]
    fn shifted_offsets_are_repaired_by_a_scan() {
        let mut data = classic(&[CATALOG, PAGES], "<< /Size 3 /Root 1 0 R >>");
        data.splice(
            9..9,
            b"% an inserted comment shifts every object\n"
                .iter()
                .copied(),
        );
        let file = File::open(data, b"").unwrap();
        assert!(file.catalog().is_ok());
        assert_ne!(file.repairs(), Vec::<String>::new());
    }

    #[test]
    fn a_missing_table_is_rebuilt() {
        let data =
            b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\ntrailer << /Root 1 0 R >>\n%%EOF"
                .to_vec();
        let file = File::open(data, b"").unwrap();
        assert_eq!(
            file.catalog().unwrap().get(b"Type"),
            Some(&Object::Name(b"Catalog".to_vec()))
        );
        assert_eq!(file.repairs().len(), 1);
    }

    #[test]
    fn an_object_stream_and_an_xref_stream_are_read() {
        // Object 2 lives in object stream 3; object 4 is the cross-reference stream (uncompressed, W [1 2 1]).
        let objstm_body = "2 0 << /Type /Pages /Kids [] /Count 0 >>";
        let first = "2 0 ".len();
        let mut out = b"%PDF-1.7\n".to_vec();
        let o1 = out.len();
        out.extend_from_slice(format!("1 0 obj\n{CATALOG}\nendobj\n").as_bytes());
        let o3 = out.len();
        out.extend_from_slice(
            format!("3 0 obj\n<< /Type /ObjStm /N 1 /First {first} /Length {} >>\nstream\n{objstm_body}\nendstream\nendobj\n", objstm_body.len())
                .as_bytes(),
        );
        let o4 = out.len();
        let mut rows = vec![0u8, 0, 0, 0];
        rows.extend([&[1u8][..], &u16::try_from(o1).unwrap().to_be_bytes(), &[0]].concat());
        rows.extend([2, 0, 3, 0]);
        rows.extend([&[1u8][..], &u16::try_from(o3).unwrap().to_be_bytes(), &[0]].concat());
        rows.extend([&[1u8][..], &u16::try_from(o4).unwrap().to_be_bytes(), &[0]].concat());
        out.extend_from_slice(
            format!(
                "4 0 obj\n<< /Type /XRef /Size 5 /W [1 2 1] /Root 1 0 R /Length {} >>\nstream\n",
                rows.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(&rows);
        out.extend_from_slice(format!("\nendstream\nendobj\nstartxref\n{o4}\n%%EOF\n").as_bytes());
        let file = File::open(out, b"").unwrap();
        let pages = file
            .get(&file.catalog().unwrap(), b"Pages")
            .unwrap()
            .unwrap();
        assert_eq!(
            pages.as_dict().unwrap().get(b"Type"),
            Some(&Object::Name(b"Pages".to_vec()))
        );
        assert_eq!(file.repairs(), Vec::<String>::new());
    }

    #[test]
    fn an_incremental_update_overrides_the_older_object() {
        let mut data = classic(&[CATALOG, PAGES], "<< /Size 3 /Root 1 0 R >>");
        let prev = data.windows(9).rposition(|w| w == b"startxref").unwrap();
        let prev_at: usize = String::from_utf8_lossy(&data[prev + 10..])
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap();
        let o2 = data.len();
        data.extend_from_slice(b"2 0 obj\n<< /Type /Pages /Kids [] /Count 7 >>\nendobj\n");
        let at = data.len();
        data.extend_from_slice(
            format!("xref\n2 1\n{o2:010} 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R /Prev {prev_at} >>\nstartxref\n{at}\n%%EOF\n").as_bytes(),
        );
        let file = File::open(data, b"").unwrap();
        assert_eq!(
            file.object(2).unwrap().as_dict().unwrap().get(b"Count"),
            Some(&Object::Integer(7))
        );
    }

    #[test]
    fn not_a_pdf_and_an_unknown_security_handler_are_refused() {
        let kind = |r: Result<File>| r.err().map(|e| e.kind);
        assert_eq!(kind(File::open(b"hello".to_vec(), b"")), Some(Kind::NotPdf));
        let data = classic(
            &[CATALOG, PAGES, "<< /Filter /Adobe.PubSec /R 4 >>"],
            "<< /Size 4 /Root 1 0 R /Encrypt 3 0 R >>",
        );
        assert_eq!(kind(File::open(data, b"")), Some(Kind::Unsupported));
    }
}
