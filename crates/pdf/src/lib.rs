//! document-lab's PDF reader and image-page writer.
//!
//! Bytes in, per-page drawing events out: what each page draws (text, paths, images) and where, in PDF points
//! with the origin top-left and y growing down. Layout, regions and pixels are not this crate's.
//!
//! Layers, each depending only on those above it: [`error`], [`object`] (the values), [`parser`] (bytes to
//! values), [`filter`] (stream data decoded), [`xref`] (where each object is), [`file`] (a PDF opened: objects
//! resolved by reference).

pub mod error;
pub mod file;
pub mod filter;
mod inflate;
pub mod object;
pub mod parser;
pub mod xref;

pub use error::{Error, Result};
pub use file::File;
pub use object::{Dict, ObjRef, Object, Stream};
