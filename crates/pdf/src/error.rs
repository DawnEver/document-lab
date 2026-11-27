//! The one error type: what kind of refusal, what went wrong and, when known, the byte offset it went wrong at.

use std::fmt;

/// What kind of refusal an [`Error`] is: what a caller may do about it differs by kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// The input is no PDF at all (no `%PDF-` header).
    NotPdf,
    /// The file is encrypted and the password given -- none, or a wrong one -- opens no access.
    Password,
    /// The file uses what this reader does not read (a security handler, a filter, an encryption version).
    Unsupported,
    /// The input is not what a PDF must be here.
    Malformed,
    /// A caller's request this crate cannot honor: an object that is not what was asked for, a value the writer
    /// may not write.
    Invalid,
    /// Writing the output failed.
    Io,
}

/// A refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// What kind of refusal it is.
    pub kind: Kind,
    /// The byte offset in the file (or in the decoded stream being parsed), when one applies.
    pub offset: Option<usize>,
    /// What is wrong, in words.
    pub message: String,
}

impl Error {
    /// A refusal of `kind` with no offset.
    pub fn new(kind: Kind, message: impl Into<String>) -> Self {
        Self {
            kind,
            offset: None,
            message: message.into(),
        }
    }

    /// Malformed input.
    pub fn malformed(message: impl Into<String>) -> Self {
        Self::new(Kind::Malformed, message)
    }

    /// Malformed input at byte `offset`.
    pub fn malformed_at(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset: Some(offset),
            ..Self::malformed(message)
        }
    }

    /// What this reader does not read.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(Kind::Unsupported, message)
    }

    /// A caller's request this crate cannot honor.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(Kind::Invalid, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.offset {
            Some(offset) => write!(f, "byte {offset}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for Error {}

/// This crate's result.
pub type Result<T> = std::result::Result<T, Error>;
