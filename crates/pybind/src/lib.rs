//! `document_lab._pdf`: the `pdf` crate, exposed to Python.

use pyo3::prelude::*;

#[pymodule]
fn _pdf(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
