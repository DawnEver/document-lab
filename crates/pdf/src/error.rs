//! The one error type: what went wrong and, when known, the byte offset it went wrong at.

use std::fmt;

/// A refusal: the input is not what a PDF must be here, or uses what this reader does not read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// The byte offset in the file (or in the decoded stream being parsed), when one applies.
    pub offset: Option<usize>,
    /// What is wrong, in words.
    pub message: String,
}

impl Error {
    /// An error at `offset`.
    pub fn at(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset: Some(offset),
            message: message.into(),
        }
    }

    /// An error with no offset.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            offset: None,
            message: message.into(),
        }
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
