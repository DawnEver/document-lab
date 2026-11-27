//! What a document states beside its pages: outline, destinations and links, page labels, metadata, form fields
//! and embedded files -- each read from a synthetic PDF whose every value is known.

mod common;

use pdf::destination::{Destination, Targets};
use pdf::forms::Value;
use pdf::geometry::Rect;
use pdf::metadata::Newer;
use pdf::{File, Page};

/// A two-page PDF: the catalog carries `catalog_extra`; page 1 (200 x 100) carries `page1_extra`, page 2 (the same
/// box turned a quarter, so 100 x 200 as displayed) `page2_extra`; `extra` objects follow as 6, 7, ...
fn two_pages(catalog_extra: &str, page1_extra: &str, page2_extra: &str, extra: &[&str]) -> Vec<u8> {
    let catalog = format!("<< /Type /Catalog /Pages 2 0 R {catalog_extra} >>");
    let page1 = format!("<< /Type /Page /Parent 2 0 R /Contents 5 0 R {page1_extra} >>");
    let page2 = format!("<< /Type /Page /Parent 2 0 R /Contents 5 0 R /Rotate 90 {page2_extra} >>");
    let content = common::stream("", "");
    let mut objects = vec![
        catalog.as_str(),
        "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 200 100] >>",
        &page1,
        &page2,
        &content,
    ];
    objects.extend_from_slice(extra);
    common::pdf(&objects)
}

fn opened(data: Vec<u8>) -> (File, Vec<Page>) {
    let file = File::open(data, b"").unwrap();
    let pages = file.pages().unwrap();
    (file, pages)
}

fn at(page: usize, fit: &str, x: Option<f64>, y: Option<f64>) -> Destination {
    Destination {
        page,
        fit: fit.into(),
        x,
        y,
        zoom: None,
        rect: None,
    }
}

#[test]
fn an_outline_is_a_tree_of_titles_leading_to_places_as_displayed() {
    let data = two_pages(
        "/Outlines 6 0 R /Dests << /intro [3 0 R /FitR 10 20 30 40] >> /Names << /Dests 12 0 R >>",
        "",
        "",
        &[
            "<< /Type /Outlines /First 7 0 R /Last 9 0 R >>",
            "<< /Title (Introduction) /Parent 6 0 R /Next 9 0 R /First 8 0 R /Last 8 0 R /Dest [3 0 R /XYZ 50 80 2] >>",
            "<< /Title <FEFF00C9007400E9> /Parent 7 0 R /Dest /intro >>",
            "<< /Title (Turned) /Parent 6 0 R /Prev 7 0 R /Next 10 0 R /A << /S /GoTo /D [4 0 R /XYZ 50 80 0] >> >>",
            "<< /Title (Site) /Parent 6 0 R /Prev 9 0 R /Next 11 0 R /A << /S /URI /URI (https://example.org/) >> >>",
            "<< /Title (Named) /Parent 6 0 R /Prev 10 0 R /Dest (byname) >>",
            "<< /Names [(byname) [4 0 R /FitH 80]] >>",
        ],
    );
    let (file, pages) = opened(data);
    let mut notes = Vec::new();
    let targets = Targets::new(&file, &pages, &mut notes);
    let outline = file.outline(&targets, &mut notes);
    assert_eq!(notes, Vec::<String>::new());
    let titles: Vec<_> = outline
        .iter()
        .map(|i| i.title.as_deref().unwrap())
        .collect();
    assert_eq!(titles, ["Introduction", "Turned", "Site", "Named"]);
    // XYZ on an upright page: y measured down from the top (100 - 80); the zoom as stated.
    let intro = outline[0].destination.clone().unwrap();
    assert_eq!(
        intro,
        Destination {
            zoom: Some(2.0),
            ..at(0, "XYZ", Some(50.0), Some(20.0))
        }
    );
    // A child named through /Dests, fitting a rectangle, as displayed.
    let child = &outline[0].children[0];
    assert_eq!(child.title.as_deref(), Some("\u{c9}t\u{e9}"));
    assert_eq!(
        child.destination,
        Some(Destination {
            rect: Some(Rect {
                x0: 10.0,
                y0: 60.0,
                x1: 30.0,
                y1: 80.0
            }),
            ..at(0, "FitR", Some(10.0), Some(60.0))
        })
    );
    // A GoTo action on a page turned a quarter: user y becomes displayed x; a zoom of 0 is left to the viewer.
    assert_eq!(
        outline[1].destination,
        Some(at(1, "XYZ", Some(80.0), Some(50.0)))
    );
    assert_eq!(outline[1].action, None);
    let site = outline[2].action.clone().unwrap();
    assert_eq!(
        (site.kind.as_str(), site.uri.as_deref()),
        ("URI", Some("https://example.org/"))
    );
    assert_eq!(outline[2].destination, None);
    // Named in the name tree, FitH on the turned page: only the displayed x is stated.
    assert_eq!(
        outline[3].destination,
        Some(at(1, "FitH", Some(80.0), None))
    );
}

#[test]
fn an_outline_that_loops_or_names_nothing_is_cut_by_name_never_hangs() {
    let data = two_pages(
        "/Outlines 6 0 R",
        "",
        "",
        &[
            "<< /Type /Outlines /First 7 0 R >>",
            "<< /Title (One) /Next 8 0 R /Dest /nowhere >>",
            "<< /Title (Two) /Next 7 0 R /First 8 0 R /Dest [99 0 R /Fit] >>",
        ],
    );
    let (file, pages) = opened(data);
    let mut notes = Vec::new();
    let targets = Targets::new(&file, &pages, &mut notes);
    let outline = file.outline(&targets, &mut notes);
    let titles: Vec<_> = outline.iter().map(|i| i.title.clone().unwrap()).collect();
    assert_eq!(titles, ["One", "Two"]);
    assert!(outline.iter().all(|i| i.destination.is_none()));
    assert_eq!(outline[1].children.len(), 0);
    let joined = notes.join("\n");
    assert!(
        joined.contains("'One'") && joined.contains("no destination is named 'nowhere'"),
        "{joined}"
    );
    assert!(
        joined.contains("'Two'") && joined.contains("object 99, no page"),
        "{joined}"
    );
    assert!(
        joined.contains("leads back to object 8") && joined.contains("leads back to object 7"),
        "{joined}"
    );
}

#[test]
fn links_are_read_as_data_with_their_rectangle_as_displayed() {
    let data = two_pages(
        "",
        "/Annots [6 0 R 7 0 R 8 0 R]",
        "/Annots [9 0 R]",
        &[
            "<< /Type /Annot /Subtype /Link /Rect [10 20 60 30] /A << /S /URI /URI (https://example.org/a) >> >>",
            "<< /Type /Annot /Subtype /Link /Rect [10 40 60 50] /Dest [4 0 R /Fit] /F 2 >>",
            "<< /Type /Annot /Subtype /Text /Rect [0 0 10 10] >>",
            "<< /Type /Annot /Subtype /Link /Rect [10 20 60 30] /Dest /missing >>",
        ],
    );
    let (file, pages) = opened(data);
    let mut notes = Vec::new();
    let targets = Targets::new(&file, &pages, &mut notes);
    let first = file.links(&pages[0], &targets, &mut notes);
    assert_eq!(
        first.len(),
        2,
        "a text annotation is no link; a hidden link is still one"
    );
    assert_eq!(
        first[0].rect,
        Rect {
            x0: 10.0,
            y0: 70.0,
            x1: 60.0,
            y1: 80.0
        }
    );
    assert_eq!(
        first[0].action.as_ref().unwrap().uri.as_deref(),
        Some("https://example.org/a")
    );
    assert_eq!(first[1].destination, Some(at(1, "Fit", None, None)));
    assert_eq!(notes, Vec::<String>::new());
    // On the turned page the user box (x 10..60, y 20..30) is displayed as x 20..30, y 10..60.
    let second = file.links(&pages[1], &targets, &mut notes);
    assert_eq!(
        second[0].rect,
        Rect {
            x0: 20.0,
            y0: 10.0,
            x1: 30.0,
            y1: 60.0
        }
    );
    assert_eq!(second[0].destination, None);
    assert!(
        notes[0].contains("link 0") && notes[0].contains("'missing'"),
        "{notes:?}"
    );
}

#[test]
fn page_labels_spell_each_page_by_its_range() {
    let five = |labels: &str| {
        let kids = "[3 0 R 4 0 R 6 0 R 7 0 R 8 0 R]";
        let catalog = format!("<< /Type /Catalog /Pages 2 0 R /PageLabels {labels} >>");
        let pages = format!("<< /Type /Pages /Kids {kids} /Count 5 /MediaBox [0 0 200 100] >>");
        let page = "<< /Type /Page /Parent 2 0 R /Contents 5 0 R >>";
        let content = common::stream("", "");
        let data = common::pdf(&[&catalog, &pages, page, page, &content, page, page, page]);
        let (file, pages) = opened(data);
        let mut notes = Vec::new();
        let labels = file.page_labels(pages.len(), &mut notes);
        (labels, notes)
    };
    let (labels, notes) =
        five("<< /Nums [0 << /S /r >> 2 << /S /D /P (A-) /St 8 >> 4 << /P (cover) >>] >>");
    let labels: Vec<_> = labels.unwrap().into_iter().map(Option::unwrap).collect();
    assert_eq!(labels, ["i", "ii", "A-8", "A-9", "cover"]);
    assert_eq!(notes, Vec::<String>::new());
    // A tree split into kids; pages before its first range have no label.
    let (labels, _) =
        five("<< /Kids [<< /Nums [1 << /S /A >>] >> << /Nums [3 << /S /R /St 4 >>] >>] >>");
    assert_eq!(
        labels.unwrap(),
        [
            None,
            Some("A".into()),
            Some("B".into()),
            Some("IV".into()),
            Some("V".into())
        ]
    );
    let bare = File::open(common::one_page("", "", &[]), b"").unwrap();
    assert_eq!(
        bare.page_labels(1, &mut Vec::new()),
        None,
        "no labels stated, none made up"
    );
}

const XMP: &str = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xmp="http://ns.adobe.com/xap/1.0/"
 xmp:CreateDate="2024-01-02T03:04:05Z">
<dc:title><rdf:Alt><rdf:li xml:lang="en-GB">Heading</rdf:li><rdf:li xml:lang="x-default">Title &amp; more</rdf:li></rdf:Alt></dc:title>
<dc:creator><rdf:Seq><rdf:li>Ada</rdf:li><rdf:li>Grace</rdf:li></rdf:Seq></dc:creator>
<xmp:ModifyDate>2024-03-04T05:06:07+01:00</xmp:ModifyDate>
</rdf:Description>
<rdf:Description rdf:about="" xmlns:p="http://ns.adobe.com/pdf/1.3/"><p:Producer>Maker 1.0</p:Producer></rdf:Description>
</rdf:RDF></x:xmpmeta>
<?xpacket end="w"?>"#;

#[test]
fn metadata_reports_info_and_xmp_as_stated_and_which_is_newer() {
    let xmp = common::stream("/Type /Metadata /Subtype /XML", XMP);
    let read = |info: &str| {
        let mut data = two_pages("/Metadata 6 0 R", "", "", &[&xmp, info]);
        // The trailer names /Info 7 0 R: it follows the table, so no object moves.
        let at = data.windows(7).rposition(|w| w == b"/Root 1").unwrap();
        data.splice(at..at, b"/Info 7 0 R ".iter().copied());
        let file = File::open(data, b"").unwrap();
        let mut notes = Vec::new();
        (file.metadata(&mut notes), notes)
    };
    let (meta, notes) = read(
        "<< /Title (Old title) /Author (Ada) /ModDate (D:20240304050607+02'00') /Trapped /False >>",
    );
    assert_eq!(notes, Vec::<String>::new());
    let info = meta.info.unwrap();
    assert_eq!(
        info,
        [
            ("Title".to_string(), "Old title".to_string()),
            ("Author".into(), "Ada".into()),
            ("ModDate".into(), "D:20240304050607+02'00'".into()),
            ("Trapped".into(), "False".into()),
        ]
    );
    let xmp = meta.xmp.unwrap();
    assert_eq!(xmp.title.as_deref(), Some("Title & more"));
    assert_eq!(xmp.creators, ["Ada", "Grace"]);
    assert_eq!(xmp.created.as_deref(), Some("2024-01-02T03:04:05Z"));
    assert_eq!(xmp.modified.as_deref(), Some("2024-03-04T05:06:07+01:00"));
    assert_eq!(xmp.producer.as_deref(), Some("Maker 1.0"));
    assert!(xmp.packet.contains("<x:xmpmeta"));
    // 05:06:07+02 is an hour before 05:06:07+01: the XMP is newer.
    assert_eq!(meta.newer, Some(Newer::Xmp));
    let (meta, _) = read("<< /ModDate (D:20250101) >>");
    assert_eq!(meta.newer, Some(Newer::Info));
    let (meta, notes) = read("<< /ModDate (yesterday) >>");
    assert_eq!(meta.newer, None);
    assert!(notes[0].contains("'yesterday' is no date"), "{notes:?}");
    let bare = File::open(common::one_page("", "", &[]), b"")
        .unwrap()
        .metadata(&mut Vec::new());
    assert_eq!((bare.info, bare.xmp, bare.newer), (None, None, None));
}

#[test]
fn form_fields_carry_their_qualified_names_values_and_widgets() {
    let data = two_pages(
        "/AcroForm << /Fields [6 0 R 9 0 R 10 0 R 13 0 R] >>",
        "/Annots [7 0 R 11 0 R 12 0 R]",
        "/Annots [8 0 R]",
        &[
            // 6: a parent naming its text kids
            "<< /T (person) /FT /Tx /Kids [7 0 R 8 0 R] >>",
            "<< /T (name) /Parent 6 0 R /V (Ada) /DV (none) /Subtype /Widget /Rect [10 20 110 40] >>",
            "<< /T (city) /Parent 6 0 R /Subtype /Widget /Rect [10 20 110 40] >>",
            // 9: a multiple choice
            "<< /T (langs) /FT /Ch /Ff 2097152 /Opt [(en) [(de) (Deutsch)]] /V [(en) (de)] >>",
            // 10: radio buttons, export values in /Opt; widgets 11 and 12 on page 1
            "<< /T (pick) /FT /Btn /Ff 49152 /V /1 /Opt [(left) (right)] /Kids [11 0 R 12 0 R] >>",
            "<< /Parent 10 0 R /Subtype /Widget /Rect [0 0 10 10] /AS /Off /AP << /N << /0 14 0 R /Off 14 0 R >> >> >>",
            "<< /Parent 10 0 R /Subtype /Widget /Rect [20 0 30 10] /AS /1 /AP << /N << /1 14 0 R /Off 14 0 R >> >> >>",
            // 13: a check box on no page's /Annots but naming page 2 by /P
            "<< /T (agree) /FT /Btn /V /Yes /Subtype /Widget /P 4 0 R /Rect [0 0 10 10] /AS /Yes /AP << /N << /Yes 14 0 R /Off 14 0 R >> >> >>",
            &common::stream("/Type /XObject /Subtype /Form /BBox [0 0 10 10]", ""),
        ],
    );
    let (file, pages) = opened(data);
    let mut notes = Vec::new();
    let fields = file.fields(&pages, &mut notes);
    assert_eq!(notes, Vec::<String>::new());
    let names: Vec<_> = fields.iter().map(|f| f.name.as_deref().unwrap()).collect();
    assert_eq!(
        names,
        ["person.name", "person.city", "langs", "pick", "agree"]
    );
    let name = &fields[0];
    assert_eq!(name.kind.as_deref(), Some("Tx"), "the type is inherited");
    assert_eq!(name.value, Some(Value::Text("Ada".into())));
    assert_eq!(name.default, Some(Value::Text("none".into())));
    assert_eq!(name.widgets[0].page, Some(0));
    assert_eq!(
        name.widgets[0].rect,
        Some(Rect {
            x0: 10.0,
            y0: 60.0,
            x1: 110.0,
            y1: 80.0
        })
    );
    assert_eq!(fields[1].value, None, "a value not stated is absent");
    assert_eq!(fields[1].widgets[0].page, Some(1));
    let langs = &fields[2];
    assert_eq!(
        langs.value,
        Some(Value::List(vec!["en".into(), "de".into()]))
    );
    assert_eq!(
        langs.options,
        [("en".into(), "en".into()), ("de".into(), "Deutsch".into())]
    );
    assert_eq!(langs.flags, Some(2_097_152));
    let pick = &fields[3];
    assert_eq!(pick.value, Some(Value::Name("1".into())));
    let exports: Vec<_> = pick
        .widgets
        .iter()
        .map(|w| w.export.as_deref().unwrap())
        .collect();
    assert_eq!(
        exports,
        ["left", "right"],
        "a button's /Opt names its widgets' export values"
    );
    let states: Vec<_> = pick
        .widgets
        .iter()
        .map(|w| w.state.as_deref().unwrap())
        .collect();
    assert_eq!(states, ["Off", "1"]);
    let agree = &fields[4];
    assert_eq!(
        agree.widgets[0].export.as_deref(),
        Some("Yes"),
        "else its on appearance state"
    );
    assert_eq!(agree.widgets[0].page, Some(1), "placed by /P");
    assert_eq!(agree.flags, None);
}

#[test]
fn embedded_files_carry_their_names_sizes_types_and_bytes() {
    let data = two_pages(
        "/Names << /EmbeddedFiles << /Names [(data.csv) 6 0 R] >> >>",
        "/Annots [8 0 R]",
        "",
        &[
            "<< /Type /Filespec /F (data.csv) /UF <FEFF0064002E006300730076> /Desc (numbers) /EF << /F 7 0 R >> >>",
            &common::stream(
                "/Type /EmbeddedFile /Subtype /text#2Fcsv /Params << /Size 6 /ModDate (D:20240101) >>",
                "a,b\n1,",
            ),
            "<< /Type /Annot /Subtype /FileAttachment /Rect [0 0 10 10] /FS << /Type /Filespec /F (note.txt) /EF << /F 9 0 R >> >> >>",
            &common::stream("/Type /EmbeddedFile /Filter /ASCIIHexDecode", "6869>"),
        ],
    );
    let (file, pages) = opened(data);
    let mut notes = Vec::new();
    let found = file.attachments(&pages, &mut notes);
    assert_eq!(notes, Vec::<String>::new());
    assert_eq!(found.len(), 2);
    let csv = &found[0];
    assert_eq!(csv.key.as_deref(), Some("data.csv"));
    assert_eq!(
        csv.name.as_deref(),
        Some("d.csv"),
        "the Unicode file name before /F"
    );
    assert_eq!(csv.description.as_deref(), Some("numbers"));
    assert_eq!(csv.mime.as_deref(), Some("text/csv"));
    assert_eq!(csv.size, Some(6));
    assert_eq!(csv.modified.as_deref(), Some("D:20240101"));
    assert_eq!(csv.page, None);
    assert_eq!(file.embedded(csv.object).unwrap(), b"a,b\n1,");
    let note = &found[1];
    assert_eq!(
        (note.key.as_deref(), note.name.as_deref(), note.page),
        (None, Some("note.txt"), Some(0))
    );
    assert_eq!(
        (note.size, note.mime.as_deref()),
        (None, None),
        "what is not stated is absent"
    );
    assert_eq!(file.embedded(note.object).unwrap(), b"hi");
    assert!(
        file.embedded(6).is_err(),
        "a file specification is no embedded file stream"
    );
}
