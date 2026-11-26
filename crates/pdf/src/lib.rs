//! document-lab's PDF reader and image-page writer.
//!
//! Bytes in, per-page drawing events out: what each page draws (text, paths, images) and where, in PDF points
//! with the origin top-left and y growing down. Layout, regions and pixels are not this crate's.
