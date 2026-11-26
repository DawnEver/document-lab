//! document-lab's PDF reader and image-page writer.
//!
//! Bytes in, per-page drawing events out: what each page draws (text, paths, images) and where, in PDF points
//! with the origin top-left and y growing down. Layout, regions and pixels are not this crate's.
//!
//! Layers, each depending only on those above it: [`error`], [`object`] (the values), [`parser`] (bytes to
//! values), [`filter`] (stream data decoded), [`crypt`] (strings and streams decrypted), [`xref`] (where each object is), [`file`] (a PDF opened: objects
//! resolved by reference), [`page`] (the page tree), [`content`] (a content stream's operations), [`font`] and [`cmap`], [`color`] and
//! [`geometry`], and [`interpret`] (what a page draws, as events in page space).

mod base14;
pub mod cmap;
pub mod color;
pub mod content;
pub mod crypt;
pub mod error;
pub mod file;
pub mod filter;
pub mod font;
mod fontfile;
pub mod geometry;
mod inflate;
pub mod interpret;
pub mod object;
pub mod page;
pub mod parser;
pub mod xref;

pub use error::{Error, Result};
pub use file::File;
pub use interpret::{Content, Event};
pub use object::{Dict, ObjRef, Object, Stream};
pub use page::Page;
