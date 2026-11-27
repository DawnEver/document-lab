//! document-lab's PDF reader and image-page writer.
//!
//! Bytes in, per-page drawing events out: what each page draws (text, paths, images) and where, in PDF points
//! with the origin top-left and y growing down. Layout, regions and pixels are not this crate's.
//!
//! Layers, each depending only on those above it: [`error`], [`object`] (the values), [`parser`] (bytes to
//! values), [`filter`] (stream data decoded), [`crypt`] (strings and streams decrypted), [`xref`] (where each object is), [`file`] (a PDF opened: objects
//! resolved by reference), [`page`] (the page tree), [`content`] (a content stream's operations), [`font`] and [`cmap`], [`function`] (PDF functions), [`color`] and
//! [`geometry`], [`image`] (an image's samples), [`optional`] (which optional content shows), and [`interpret`] (what a
//! page draws, as events in page space),
//! read as [`spans`] (runs of text) and [`paths`] (subpaths, curves flattened).
//!
//! What the document states beside its pages, each read from a [`File`] and its pages: [`outline`] (bookmarks),
//! [`links`] (each page's), both leading to a [`destination`]; page [`labels`]; [`metadata`] (`/Info` and XMP, read
//! by [`xml`]); [`forms`] (field values); [`attachments`] (embedded files). Name and number trees are [`trees`].
//! [`writer`] writes a PDF of image pages, or of pages copied from a [`File`].

mod accents;
pub mod attachments;
mod base14;
mod cid2unicode;
pub mod cmap;
mod cmaps;
pub mod color;
mod compositions;
pub mod content;
pub mod crypt;
pub mod destination;
pub mod error;
pub mod file;
pub mod filter;
pub mod font;
mod fontfile;
pub mod forms;
pub mod function;
pub mod geometry;
pub mod image;
mod inflate;
pub mod interpret;
pub mod labels;
pub mod links;
pub mod metadata;
pub mod object;
pub mod optional;
pub mod outline;
pub mod page;
pub mod parser;
pub mod paths;
pub mod spans;
pub mod trees;
pub mod writer;
pub mod xml;
pub mod xref;

pub use error::{Error, Result};
pub use file::File;
pub use interpret::{Content, Event};
pub use object::{Dict, ObjRef, Object, Stream};
pub use page::Page;
