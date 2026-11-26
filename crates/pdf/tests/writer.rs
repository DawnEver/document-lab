//! The writer: image pages read back as written, a copied page draws what its source drew, and a page tree and
//! cross-reference table a reader needs no repair to read.

mod common;

use pdf::color::Space;
use pdf::writer::{Image, Writer};
use pdf::{Event, File};

/// Every text glyph a page draws, as its text.
fn text(file: &File, page: &pdf::Page) -> String {
    file.interpret(page, false)
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
    let mut writer = Writer::new(Vec::new()).unwrap();
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
    let events = file.interpret(&pages[1], false).unwrap().events;
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
    let mut writer = Writer::new(Vec::new()).unwrap();
    writer.image_page(&image, (10.0, 10.0)).unwrap();
    let file = File::open(writer.finish().unwrap(), b"").unwrap();
    let page = &file.pages().unwrap()[0];
    let events = file.interpret(page, false).unwrap().events;
    let [Event::Image(placed)] = events.as_slice() else {
        panic!("{events:?}")
    };
    let read = file.image(placed.object.unwrap()).unwrap();
    assert_eq!((read.space, read.data), (Some(Space::Gray), vec![0x7f]));
}

#[test]
fn a_space_or_depth_no_image_page_has_is_refused() {
    let mut writer = Writer::new(Vec::new()).unwrap();
    let mut image = Image {
        width: 1,
        height: 1,
        space: b"Lab",
        bits: 8,
        filter: None,
        data: &[0, 0, 0],
    };
    let error = writer.image_page(&image, (1.0, 1.0)).unwrap_err();
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
    let mut writer = Writer::new(Vec::new()).unwrap();
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
    let mut writer = Writer::new(Vec::new()).unwrap();
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
    let mut writer = Writer::new(Vec::new()).unwrap();
    writer
        .copy_page(&source, &source.pages().unwrap()[0])
        .unwrap();
    let copy = File::open(writer.finish().unwrap(), b"").unwrap();
    // catalog, page tree, the page, its content, the annotation: the link's target page is not copied again
    assert_eq!(copy.numbers().len(), 5);
}
