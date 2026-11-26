//! Encrypted files open with the user password, the owner password, or none when anyone may read them; their
//! strings and streams come back decrypted. The fixtures are synthetic (`fixtures/make_encrypted.py`).

use pdf::error::Kind;
use pdf::{File, Object};

const METHODS: [&str; 4] = ["rc4-40", "rc4-128", "aes-128", "aes-256"];

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!(
        "{}/tests/fixtures/{name}.pdf",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

/// The title from `/Info` and the first page's content stream, decoded.
fn title_and_content(file: &File) -> (Vec<u8>, Vec<u8>) {
    let Some(Object::Dict(info)) = file.get(file.trailer(), b"Info").unwrap() else {
        panic!("no /Info")
    };
    let Some(Object::String(title)) = file.get(&info, b"Title").unwrap() else {
        panic!("no /Title")
    };
    let Some(Object::Dict(pages)) = file.get(&file.catalog().unwrap(), b"Pages").unwrap() else {
        panic!()
    };
    let Some(Object::Array(kids)) = file.get(&pages, b"Kids").unwrap() else {
        panic!()
    };
    let Object::Dict(page) = file.resolve(&kids[0]).unwrap() else {
        panic!()
    };
    let content = match file.get(&page, b"Contents").unwrap() {
        Some(Object::Stream(s)) => s,
        Some(Object::Array(parts)) => match file.resolve(&parts[0]).unwrap() {
            Object::Stream(s) => s,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    };
    (title, file.decode(&content).unwrap().into_bytes().unwrap())
}

fn assert_reads(file: &File, method: &str) {
    let (title, content) = title_and_content(file);
    assert_eq!(title, b"Secret Title", "{method}");
    let text = b"<7365637265742074657874203432>"; // "secret text 42", hex-encoded by the writer
    assert!(
        content.windows(text.len()).any(|w| w == text),
        "{method}: {}",
        String::from_utf8_lossy(&content)
    );
}

#[test]
fn every_method_opens_with_the_user_password_and_the_owner_password() {
    for method in METHODS {
        for password in [&b"user"[..], b"owner"] {
            let file =
                File::open(fixture(method), password).unwrap_or_else(|e| panic!("{method}: {e}"));
            assert_reads(&file, method);
        }
    }
}

#[test]
fn a_file_anyone_may_read_opens_without_a_password() {
    for method in METHODS {
        let file = File::open(fixture(&format!("open-{method}")), b"")
            .unwrap_or_else(|e| panic!("{method}: {e}"));
        assert_reads(&file, method);
    }
}

#[test]
fn a_wrong_or_missing_password_is_refused_as_a_password_refusal() {
    for method in METHODS {
        let Err(err) = File::open(fixture(method), b"wrong") else {
            panic!("{method} opened")
        };
        assert_eq!(err.kind, Kind::Password, "{method}: {err}");
        let Err(err) = File::open(fixture(method), b"") else {
            panic!("{method} opened without its password")
        };
        assert_eq!(err.kind, Kind::Password, "{method}: {err}");
    }
}
