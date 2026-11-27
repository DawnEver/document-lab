//! Embedded files (ISO 32000-2 section 7.11.4): the files the document carries -- in the catalog's `/Names
//! /EmbeddedFiles` tree, and in pages' file attachment annotations (12.5.6.15) -- each with its name, size, MIME
//! subtype and dates as stated, and the object its bytes are in ([`File::embedded`] decodes them).

use crate::destination::spec_name;
use crate::error::{Error, Result};
use crate::file::File;
use crate::object::{Dict, Object, text_string};
use crate::page::Page;

/// One embedded file.
#[derive(Debug, Clone, PartialEq)]
pub struct Attachment {
    /// Its key in the embedded files tree; None for a page's attachment.
    pub key: Option<String>,
    /// Its file name: the file specification's `/UF`, else its `/F`.
    pub name: Option<String>,
    /// Its description (`/Desc`).
    pub description: Option<String>,
    /// Its MIME type as stated (the stream's `/Subtype`: `application/pdf`, ...).
    pub mime: Option<String>,
    /// Its size in bytes as stated (`/Params /Size`).
    pub size: Option<i64>,
    /// When it was made, as stated (`/Params /CreationDate`).
    pub created: Option<String>,
    /// When it was last changed, as stated (`/Params /ModDate`).
    pub modified: Option<String>,
    /// The page (0-based) whose file attachment annotation carries it; None for the document's.
    pub page: Option<usize>,
    /// The embedded file stream's object number: what [`File::embedded`] reads.
    pub object: u32,
}

impl File {
    /// Every embedded file: the document's, in tree order, then each page's attachment annotations'. What cannot
    /// be read goes to `notes`.
    pub fn attachments(&self, pages: &[Page], notes: &mut Vec<String>) -> Vec<Attachment> {
        let mut out = Vec::new();
        let names = self.catalog().and_then(|c| self.get(&c, b"Names"));
        if let Ok(Some(Object::Dict(names))) = names
            && let Some(root) = names.get(b"EmbeddedFiles")
        {
            for (key, spec) in self.name_tree(root, "the embedded files", notes) {
                let key = text_string(&key);
                match self.attachment(&spec, Some(key.clone()), None) {
                    Ok(a) => out.push(a),
                    Err(e) => notes.push(format!("embedded file '{key}': skipped: {e}")),
                }
            }
        }
        for (i, page) in pages.iter().enumerate() {
            let Ok(Some(Object::Array(annots))) = self.get(&page.dict, b"Annots") else {
                continue;
            };
            for annot in &annots {
                let Ok(Object::Dict(a)) = self.resolve(annot) else {
                    continue;
                };
                if a.get(b"Subtype").and_then(Object::as_name) != Some(b"FileAttachment") {
                    continue;
                }
                match a.get(b"FS").map(|fs| self.attachment(fs, None, Some(i))) {
                    Some(Ok(found)) => out.push(found),
                    Some(Err(e)) => {
                        notes.push(format!("page {}: a file attachment skipped: {e}", i + 1));
                    }
                    None => notes.push(format!(
                        "page {}: a file attachment without its /FS; skipped",
                        i + 1
                    )),
                }
            }
        }
        out
    }

    /// The bytes of the embedded file stream `object`, every filter undone.
    ///
    /// # Errors
    ///
    /// Refuses an object that is no stream, and data its filters do not decode.
    pub fn embedded(&self, object: u32) -> Result<Vec<u8>> {
        match self.object(object)? {
            Object::Stream(s) => self.decode(&s)?.into_bytes(),
            other => Err(Error::invalid(format!(
                "object {object} is a {}, no embedded file stream",
                other.kind()
            ))),
        }
    }

    /// The file specification `spec` (7.11.3) as an attachment.
    fn attachment(
        &self,
        spec: &Object,
        key: Option<String>,
        page: Option<usize>,
    ) -> Result<Attachment> {
        let Object::Dict(spec) = self.resolve(spec)? else {
            return Err(Error::malformed("its file specification is no dictionary"));
        };
        let Some(Object::Dict(ef)) = self.get(&spec, b"EF")? else {
            return Err(Error::malformed("it names a file but embeds none (no /EF)"));
        };
        let reference = [&b"UF"[..], b"F"]
            .iter()
            .find_map(|k| ef.get(k).and_then(Object::as_reference))
            .ok_or_else(|| Error::malformed("its /EF names no stream object"))?;
        let Object::Stream(stream) = self.object(reference.num)? else {
            return Err(Error::malformed(format!(
                "its /EF object {} is no stream",
                reference.num
            )));
        };
        let params = match self.get(&stream.dict, b"Params")? {
            Some(Object::Dict(p)) => p,
            _ => Dict::new(),
        };
        let text = |dict: &Dict, k: &[u8]| -> Result<Option<String>> {
            Ok(match self.get(dict, k)? {
                Some(Object::String(s)) => Some(text_string(&s)),
                _ => None,
            })
        };
        Ok(Attachment {
            key,
            name: spec_name(self, &spec)?,
            description: text(&spec, b"Desc")?,
            mime: self
                .get(&stream.dict, b"Subtype")?
                .and_then(|s| s.as_name().map(|n| String::from_utf8_lossy(n).into_owned())),
            size: self.get(&params, b"Size")?.and_then(|s| s.as_integer()),
            created: text(&params, b"CreationDate")?,
            modified: text(&params, b"ModDate")?,
            page,
            object: reference.num,
        })
    }
}
