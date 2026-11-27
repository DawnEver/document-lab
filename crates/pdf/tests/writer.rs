//! The writer: image pages read back as written, a copied page draws what its source drew, and a page tree and
//! cross-reference table a reader needs no repair to read.

mod common;

use pdf::color::Space;
use pdf::geometry::Rect;
use pdf::writer::{Image, LEVEL, Word, Writer};
use pdf::{Event, File, Object};

/// Every text glyph a page draws, as its text.
fn text(file: &File, page: &pdf::Page) -> String {
    file.interpret(page)
        .unwrap()
        .events
        .iter()
        .filter_map(|e| match e {
            Event::Text(t) => Some(
                t.glyphs
                    .iter()
                    .filter_map(|g| g.text.clone())
                    .collect::<String>(),
            ),
            _ => None,
        })
        .collect()
}

#[test]
fn image_pages_read_back_as_written() {
    let samples: Vec<u8> = (0..4 * 2 * 3).map(|v| u8::try_from(v).unwrap()).collect();
    let image = Image {
        width: 4,
        height: 2,
        space: b"DeviceRGB",
        bits: 8,
        filter: None,
        data: &samples,
    };
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer.image_page(&image, (30.0, 15.5)).unwrap();
    writer.image_page(&image, (60.0, 31.0)).unwrap();
    assert_eq!(writer.pages(), 2);
    let file = File::open(writer.finish().unwrap(), b"").unwrap();
    assert!(file.repairs().is_empty(), "{:?}", file.repairs());
    let pages = file.pages().unwrap();
    assert_eq!(
        pages.iter().map(pdf::Page::size).collect::<Vec<_>>(),
        [(30.0, 15.5), (60.0, 31.0)]
    );
    let events = file.interpret(&pages[1]).unwrap().events;
    let [Event::Image(placed)] = events.as_slice() else {
        panic!("{events:?}")
    };
    assert_eq!(
        (
            placed.bbox.x0,
            placed.bbox.y0,
            placed.bbox.x1,
            placed.bbox.y1
        ),
        (0.0, 0.0, 60.0, 31.0)
    );
    let read = file.image(placed.object.unwrap()).unwrap();
    assert_eq!((read.width, read.height, read.bits), (4, 2, 8));
    assert_eq!(read.space, Some(Space::Rgb));
    assert_eq!((read.data, read.codec), (samples, None));
}

#[test]
fn encoded_samples_are_written_as_given_with_their_filter() {
    let image = Image {
        width: 1,
        height: 1,
        space: b"DeviceGray",
        bits: 8,
        filter: Some(b"ASCIIHexDecode"),
        data: b"7f>",
    };
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer.image_page(&image, (10.0, 10.0)).unwrap();
    let file = File::open(writer.finish().unwrap(), b"").unwrap();
    let page = &file.pages().unwrap()[0];
    let events = file.interpret(page).unwrap().events;
    let [Event::Image(placed)] = events.as_slice() else {
        panic!("{events:?}")
    };
    let read = file.image(placed.object.unwrap()).unwrap();
    assert_eq!((read.space, read.data), (Some(Space::Gray), vec![0x7f]));
}

#[test]
fn a_space_or_depth_no_image_page_has_is_refused() {
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    let mut image = Image {
        width: 1,
        height: 1,
        space: b"Lab",
        bits: 8,
        filter: None,
        data: &[0, 0, 0],
    };
    let error = writer.image_page(&image, (1.0, 1.0)).unwrap_err();
    assert_eq!(error.kind, pdf::error::Kind::Invalid, "{error}");
    assert!(error.message.contains("not Lab"), "{error}");
    image.space = b"DeviceRGB";
    image.bits = 3;
    assert!(writer.image_page(&image, (1.0, 1.0)).is_err());
    image.bits = 8;
    assert!(writer.image_page(&image, (0.0, 1.0)).is_err());
    assert_eq!(writer.pages(), 0);
}

#[test]
fn a_copied_page_draws_what_its_source_drew_with_what_it_inherits() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
    let source = common::one_page(
        "/Resources << /Font << /F1 5 0 R >> >> /Rotate 90",
        "BT /F1 12 Tf 10 20 Td (copied) Tj ET",
        &[font],
    );
    let source = File::open(source, b"").unwrap();
    let page = &source.pages().unwrap()[0];
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer.copy_page(&source, page).unwrap();
    writer.copy_page(&source, page).unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    assert!(copy.repairs().is_empty(), "{:?}", copy.repairs());
    let pages = copy.pages().unwrap();
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[1].size(), page.size()); // the media box inherited from the source's tree, and its turn
    assert_eq!(pages[1].quarter_turns, 1);
    assert_eq!(text(&copy, &pages[1]), "copied");
}

#[test]
fn a_page_of_an_encrypted_file_is_copied_decrypted() {
    let data = std::fs::read(format!(
        "{}/tests/fixtures/aes-256.pdf",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    let source = File::open(data, b"user").unwrap();
    let page = &source.pages().unwrap()[0];
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer.copy_page(&source, page).unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    let copied = &copy.pages().unwrap()[0];
    assert_eq!(text(&copy, copied), text(&source, page));
    assert!(text(&copy, copied).contains("secret text 42"));
}

#[test]
fn a_link_to_another_page_is_not_followed_into_the_whole_tree() {
    let annotation = "<< /Type /Annot /Subtype /Link /Rect [0 0 10 10] /Dest [3 0 R /Fit] >>";
    let source = common::one_page("/Annots [5 0 R]", "", &[annotation]);
    let source = File::open(source, b"").unwrap();
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer
        .copy_page(&source, &source.pages().unwrap()[0])
        .unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    // catalog, page tree, the page, its content, the annotation: the link's target page is not copied again
    assert_eq!(copy.numbers().len(), 5);
}

/// Each span a page draws: its text, its box in page space, and its render mode.
fn spans(file: &File, page: &pdf::Page) -> Vec<(String, [f64; 4], i64)> {
    pdf::spans::spans(&file.interpret(page).unwrap().events)
        .into_iter()
        .map(|s| {
            let b = s.bbox;
            (s.text, [b.x0, b.y0, b.x1, b.y1], s.render_mode)
        })
        .collect()
}

fn word(text: &str, [x0, y0, x1, y1]: [f64; 4]) -> Word<'_> {
    Word {
        text,
        bbox: Rect { x0, y0, x1, y1 },
    }
}

/// `actual` is `expected` within a hundredth of a point at each edge.
fn near(actual: [f64; 4], expected: [f64; 4]) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(a, e)| (a - e).abs() < 0.01)
}

/// `source`'s first page copied with `words` laid over it, read back.
fn laid(source: &[u8], words: &[Word]) -> (File, Vec<pdf::Page>) {
    let source = File::open(source.to_vec(), b"").unwrap();
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer
        .copy_page_with_text(&source, &source.pages().unwrap()[0], words)
        .unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    assert!(copy.repairs().is_empty(), "{:?}", copy.repairs());
    let pages = copy.pages().unwrap();
    (copy, pages)
}

#[test]
fn a_text_layer_reads_back_invisible_over_its_words() {
    let source = common::one_page("", "0 0 1 rg 0 0 200 100 re f", &[]);
    let words = [
        word("stator", [10.0, 20.0, 70.0, 35.0]),
        word(
            "\u{5b9a}\u{5b50}\u{7ed5}\u{7ec4}",
            [100.0, 50.0, 180.0, 70.0],
        ), // stator winding
    ];
    let (copy, pages) = laid(&source, &words);
    let read = spans(&copy, &pages[0]);
    assert_eq!(read.len(), 2, "{read:?}");
    for ((text, bbox, mode), word) in read.iter().zip(&words) {
        assert_eq!(text, word.text);
        let b = word.bbox;
        assert!(near(*bbox, [b.x0, b.y0, b.x1, b.y1]), "{bbox:?} for {text}");
        assert_eq!(*mode, 3);
    }
    let events = copy.interpret(&pages[0]).unwrap().events;
    assert!(events.iter().any(|e| matches!(e, Event::Path(_)))); // what the page drew is drawn still
}

#[test]
fn a_text_layer_on_a_turned_page_lies_where_it_is_seen() {
    let source = common::one_page("/Rotate 90 /CropBox [10 5 190 95]", "", &[]);
    let (copy, pages) = laid(&source, &[word("turned", [12.0, 30.0, 60.0, 44.0])]);
    assert_eq!(pages[0].size(), (90.0, 180.0));
    let read = spans(&copy, &pages[0]);
    assert_eq!(read.len(), 1, "{read:?}");
    assert_eq!(read[0].0, "turned");
    assert!(near(read[0].1, [12.0, 30.0, 60.0, 44.0]), "{read:?}");
}

#[test]
fn what_the_page_leaves_changed_does_not_move_the_layer() {
    // a content stream that leaves the page scaled, its text squeezed, raised and filled
    let source = common::one_page("", "3 0 0 3 50 50 cm 4 Tz 9 Ts 2 Tr", &[]);
    let (copy, pages) = laid(&source, &[word("kept", [10.0, 10.0, 40.0, 20.0])]);
    let read = spans(&copy, &pages[0]);
    assert!(near(read[0].1, [10.0, 10.0, 40.0, 20.0]), "{read:?}");
    assert_eq!(read[0].2, 3);
}

#[test]
fn a_page_s_own_fonts_and_text_are_kept_beside_the_layer() {
    let font = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>";
    let source = common::one_page(
        "/Resources << /Font 5 0 R >>",
        "BT /F1 12 Tf 10 20 Td (printed) Tj ET",
        &["<< /F1 6 0 R >>", font],
    );
    let (copy, pages) = laid(&source, &[word("laid", [100.0, 10.0, 130.0, 20.0])]);
    let texts: Vec<String> = spans(&copy, &pages[0]).into_iter().map(|s| s.0).collect();
    assert_eq!(texts, ["printed", "laid"]);
}

/// A word with a character beyond the Basic Multilingual Plane (U+1D538, double-struck A): two UTF-16 units.
const BEYOND_BMP: &str = "b\u{1d538}c";

#[test]
fn pages_laid_in_one_document_share_one_font_that_names_every_character() {
    let source = common::one_page("", "", &[]);
    let source = File::open(source, b"").unwrap();
    let page = &source.pages().unwrap()[0];
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    writer
        .copy_page_with_text(&source, page, &[word("ab", [0.0, 0.0, 20.0, 10.0])])
        .unwrap();
    writer.copy_page(&source, page).unwrap();
    writer
        .copy_page_with_text(&source, page, &[word(BEYOND_BMP, [0.0, 0.0, 30.0, 10.0])])
        .unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    let pages = copy.pages().unwrap();
    assert_eq!(text(&copy, &pages[0]), "ab");
    assert_eq!(text(&copy, &pages[1]), "");
    assert_eq!(text(&copy, &pages[2]), BEYOND_BMP);
    let fonts = copy
        .numbers()
        .into_iter()
        .filter(|&n| {
            let object = copy.object(n).unwrap();
            object
                .as_dict()
                .and_then(|d| d.get(b"Subtype"))
                .and_then(Object::as_name)
                == Some(b"Type0")
        })
        .count();
    assert_eq!(fonts, 1);
}

#[test]
fn a_word_without_text_or_a_box_is_refused() {
    let source = File::open(common::one_page("", "", &[]), b"").unwrap();
    let page = &source.pages().unwrap()[0];
    let mut writer = Writer::new(Vec::new(), LEVEL).unwrap();
    for bad in [
        word("", [0.0, 0.0, 10.0, 10.0]),
        word("flat", [0.0, 5.0, 10.0, 5.0]),
        word("nan", [f64::NAN, 0.0, 10.0, 10.0]),
    ] {
        let error = writer
            .copy_page_with_text(&source, page, &[bad])
            .unwrap_err();
        assert!(error.message.contains("a word of a text layer"), "{error}");
        assert_eq!(error.kind, pdf::error::Kind::Invalid, "{error}");
    }
    assert_eq!(writer.pages(), 0);
}

/// A stream read back: the filter it names, the bytes it stores and its data decoded.
type Read = (Option<Vec<u8>>, Vec<u8>, Vec<u8>);

/// Each stream of `pdf`, by object number.
fn streams(pdf: Vec<u8>) -> Vec<Read> {
    let file = File::open(pdf, b"").unwrap();
    assert!(file.repairs().is_empty(), "{:?}", file.repairs());
    file.numbers()
        .into_iter()
        .filter_map(|n| match file.object(n).unwrap() {
            Object::Stream(s) => {
                let filter = s
                    .dict
                    .get(b"Filter")
                    .and_then(Object::as_name)
                    .map(<[u8]>::to_vec);
                let data = file.decode(&s).unwrap().into_bytes().unwrap();
                Some((filter, s.raw, data))
            }
            _ => None,
        })
        .collect()
}

/// A document of every kind of stream: an image page (its image hex-encoded by the caller) and a page copied with
/// a text layer (its drawing stored unfiltered), written at `level`.
fn every_stream(level: u8) -> Vec<u8> {
    let image = Image {
        width: 1,
        height: 1,
        space: b"DeviceGray",
        bits: 8,
        filter: Some(b"ASCIIHexDecode"),
        data: b"7f>",
    };
    let source = File::open(common::one_page("", "0 0 1 rg 0 0 200 100 re f", &[]), b"").unwrap();
    let mut writer = Writer::new(Vec::new(), level).unwrap();
    writer.image_page(&image, (10.0, 10.0)).unwrap();
    let words = [
        word("stator", [10.0, 20.0, 70.0, 35.0]),
        word("\u{5b9a}\u{5b50}", [80.0, 20.0, 120.0, 35.0]),
    ];
    writer
        .copy_page_with_text(&source, &source.pages().unwrap()[0], &words)
        .unwrap();
    writer.finish().unwrap()
}

#[test]
fn the_streams_the_writer_builds_are_deflated_and_read_back_identically() {
    let stored = streams(every_stream(0));
    let deflated = DEFLATE_FILTER.to_vec();
    // the image as the caller encoded it, the copied drawing as its source stored it, every other one built here
    let kinds: Vec<Option<&[u8]>> = stored.iter().map(|(f, ..)| f.as_deref()).collect();
    assert_eq!(
        kinds,
        [
            Some(&b"ASCIIHexDecode"[..]),
            Some(DEFLATE_FILTER), // the image page's drawing
            None,                 // the copied page's own drawing
            Some(DEFLATE_FILTER), // the saved state before it
            Some(DEFLATE_FILTER), // the text layer
            Some(DEFLATE_FILTER), // the font's ToUnicode
        ]
    );
    assert_eq!(stored[0].1, b"7f>");
    for level in [1, LEVEL, 10] {
        let written = streams(every_stream(level));
        for (at, (a, b)) in stored.iter().zip(&written).enumerate() {
            assert_eq!((&a.0, &a.2), (&b.0, &b.2), "stream {at} at level {level}");
            if a.0.as_ref() == Some(&deflated) && a.2.len() > 100 {
                assert!(
                    b.1.len() < a.1.len(),
                    "stream {at} not smaller at level {level}"
                );
            }
        }
    }
    assert!(String::from_utf8_lossy(&stored[4].2).contains("3 Tr"));
}

#[test]
fn a_compression_level_beyond_10_is_refused() {
    let error = Writer::new(Vec::new(), 11).err().unwrap();
    assert_eq!(error.kind, pdf::error::Kind::Invalid, "{error}");
    assert!(error.message.contains("0 to 10, not 11"), "{error}");
}

/// The filter the writer compresses its own streams with.
const DEFLATE_FILTER: &[u8] = b"FlateDecode"; // spelling-ignore-line: the PDF name
