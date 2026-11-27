//! The logical structure of a tagged PDF: its elements in order, their roles as stated and as standard, their
//! attributes and text entries, the content they cover; cycles and broken references refused by name.

mod common;

use pdf::File;
use pdf::geometry::Rect;
use pdf::structure::{Attribute, Element, Kid, Structure, Value};

/// A two-page PDF: catalog (1, carrying `catalog`), pages (2), page 1 (3, carrying `first`), page 2 (4), the
/// structure tree root (5, `root`), then `extra` objects as 6, 7, ...
fn tagged(catalog: &str, first: &str, root: &str, extra: &[&str]) -> Option<Structure> {
    let catalog = format!("<< /Type /Catalog /Pages 2 0 R {catalog} >>");
    let first = format!("<< /Type /Page /Parent 2 0 R {first} >>");
    let mut objects = vec![
        catalog.as_str(),
        "<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 /MediaBox [0 0 200 100] >>",
        &first,
        "<< /Type /Page /Parent 2 0 R /StructParents 1 >>",
        root,
    ];
    objects.extend_from_slice(extra);
    let file = File::open(common::pdf(&objects), b"").unwrap();
    let pages = file.pages().unwrap();
    file.structure(&pages).unwrap()
}

const ROOT: &str = "/StructTreeRoot 5 0 R";

fn element<'a>(structure: &'a Structure, role: &str) -> &'a Element {
    structure
        .elements
        .iter()
        .find(|e| e.role == role)
        .unwrap_or_else(|| panic!("no {role} in {structure:?}"))
}

#[test]
fn an_untagged_file_has_no_structure() {
    let file = File::open(common::one_page("", "0 0 1 1 re f", &[]), b"").unwrap();
    assert!(file.structure(&file.pages().unwrap()).unwrap().is_none());
    assert!(tagged("", "", "<< >>", &[]).is_none());
}

#[test]
fn elements_come_in_tree_order_with_their_roles_text_entries_and_content() {
    let structure = tagged(
        ROOT,
        "/StructParents 0",
        "<< /Type /StructTreeRoot /K 6 0 R /ParentTree << /Nums [0 [7 0 R 8 0 R] 1 [8 0 R]] >> >>",
        &[
            "<< /Type /StructElem /S /Document /K [7 0 R 8 0 R] /Lang (en-GB) >>",
            "<< /Type /StructElem /S /H1 /P 6 0 R /Pg 3 0 R /K 0 /T (Intro) >>",
            "<< /Type /StructElem /S /P /P 6 0 R /Pg 3 0 R /K [1 << /Type /MCR /Pg 4 0 R /MCID 0 >>] \
             /Alt (a paragraph) /ActualText <FEFF00E9> /E (exp.) >>",
        ],
    )
    .unwrap();
    assert_eq!(structure.warnings, Vec::<String>::new());
    let roles: Vec<&str> = structure.elements.iter().map(|e| e.role.as_str()).collect();
    assert_eq!(roles, ["Document", "H1", "P"]);
    assert_eq!(structure.roots, [0]);
    let document = &structure.elements[0];
    assert_eq!(document.kids, [Kid::Element(1), Kid::Element(2)]);
    assert_eq!(document.lang.as_deref(), Some("en-GB"));
    let h1 = element(&structure, "H1");
    assert_eq!(
        (h1.title.as_deref(), h1.page, &h1.kids),
        (
            Some("Intro"),
            Some(1),
            &vec![Kid::Marked {
                page: 1,
                mcid: 0,
                stream: None
            }]
        )
    );
    let p = element(&structure, "P");
    assert_eq!(
        p.kids,
        [
            Kid::Marked {
                page: 1,
                mcid: 1,
                stream: None
            },
            Kid::Marked {
                page: 2,
                mcid: 0,
                stream: None
            }
        ]
    );
    assert_eq!(
        (
            p.alt.as_deref(),
            p.actual_text.as_deref(),
            p.expansion.as_deref()
        ),
        (Some("a paragraph"), Some("\u{e9}"), Some("exp."))
    );
}

#[test]
fn a_custom_role_keeps_its_name_beside_the_standard_role_it_maps_to() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K [6 0 R 7 0 R 8 0 R 9 0 R] /RoleMap << /Heading1 /Head /Head /H1 /Mine /Unknown /Loop1 /Loop2 /Loop2 /Loop1 \
         /P /Span >> >>",
        &[
            "<< /S /Heading1 >>",
            "<< /S /Mine >>",
            "<< /S /Loop1 >>",
            "<< /S /P >>",
        ],
    )
    .unwrap();
    let roles: Vec<(&str, Option<&str>)> = structure
        .elements
        .iter()
        .map(|e| (e.role.as_str(), e.standard.as_deref()))
        .collect();
    // A chain is followed; a role mapped to no standard one, or round a loop, has none; a standard role is itself.
    assert_eq!(
        roles,
        [
            ("Heading1", Some("H1")),
            ("Mine", None),
            ("Loop1", None),
            ("P", Some("P"))
        ]
    );
}

#[test]
fn a_heading_of_any_level_and_the_pdf_2_roles_are_standard() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K [6 0 R 7 0 R 8 0 R] >>",
        &["<< /S /H9 >>", "<< /S /Title >>", "<< /S /FENote >>"],
    )
    .unwrap();
    let standard: Vec<Option<&str>> = structure
        .elements
        .iter()
        .map(|e| e.standard.as_deref())
        .collect();
    assert_eq!(standard, [Some("H9"), Some("Title"), Some("FENote")]);
}

#[test]
fn attributes_come_from_the_element_then_its_classes_the_elements_own_first() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K 6 0 R /ClassMap << /Wide << /O /Table /ColSpan 3 /Scope /Column >> \
         /Both [<< /O /Layout /Placement /Block >> 7 0 R] >> >>",
        &[
            "<< /S /TH /A [<< /O /Table /ColSpan 2 /Headers [(h1) (h2)] >> 1] /C [/Wide /Both 0] >>",
            "<< /O /Layout /BBox [0 0 10 5] /Color [0.5 0 0] /Summary (the cell) /Shown true >>",
        ],
    )
    .unwrap();
    let attribute = |owner: &str, name: &str, value: Value| Attribute {
        owner: owner.into(),
        name: name.into(),
        value,
    };
    let integers = |v: &[i64]| Value::Array(v.iter().copied().map(Value::Integer).collect());
    assert_eq!(
        structure.elements[0].attributes,
        [
            attribute("Table", "ColSpan", Value::Integer(2)),
            attribute(
                "Table",
                "Headers",
                Value::Array(vec![Value::Text("h1".into()), Value::Text("h2".into())])
            ),
            attribute("Table", "Scope", Value::Name("Column".into())),
            attribute("Layout", "Placement", Value::Name("Block".into())),
            attribute("Layout", "BBox", integers(&[0, 0, 10, 5])),
            attribute(
                "Layout",
                "Color",
                Value::Array(vec![Value::Real(0.5), Value::Integer(0), Value::Integer(0)])
            ),
            attribute("Layout", "Summary", Value::Text("the cell".into())),
            attribute("Layout", "Shown", Value::Bool(true)),
        ]
    );
}

#[test]
fn marked_content_in_a_form_names_its_stream_and_an_object_reference_its_object() {
    let structure = tagged(
        ROOT,
        "/Annots [8 0 R]",
        "<< /K 6 0 R >>",
        &[
            "<< /S /Link /Pg 3 0 R /K [<< /Type /MCR /Stm 7 0 R /MCID 2 >> << /Type /OBJR /Obj 8 0 R >>] >>",
            &common::stream(
                "/Type /XObject /Subtype /Form /BBox [0 0 9 9]",
                "/P << /MCID 2 >> BDC EMC",
            ),
            "<< /Type /Annot /Subtype /Link /Rect [10 20 50 30] >>",
        ],
    )
    .unwrap();
    assert_eq!(
        structure.elements[0].kids,
        [
            Kid::Marked {
                page: 1,
                mcid: 2,
                stream: Some(7)
            },
            // The annotation's /Rect in page space: y down from the page's top (100).
            Kid::Object {
                page: Some(1),
                object: 8,
                rect: Some(Rect {
                    x0: 10.0,
                    y0: 70.0,
                    x1: 50.0,
                    y1: 80.0
                })
            }
        ]
    );
}

#[test]
fn a_page_is_inherited_from_the_nearest_element_above_that_names_one() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K 6 0 R >>",
        &["<< /S /Sect /Pg 4 0 R /K 7 0 R >>", "<< /S /P /K 3 >>"],
    )
    .unwrap();
    assert_eq!(
        structure.elements[1].kids,
        [Kid::Marked {
            page: 2,
            mcid: 3,
            stream: None
        }]
    );
}

#[test]
fn the_parent_tree_names_the_page_of_an_mcid_no_element_places() {
    let structure = tagged(
        ROOT,
        "/StructParents 0",
        "<< /K 6 0 R /ParentTree << /Kids [<< /Limits [0 1] /Nums [0 [null 6 0 R] 1 []] >>] >> >>",
        &["<< /S /P /K 1 >>"],
    )
    .unwrap();
    assert_eq!(structure.warnings, Vec::<String>::new());
    assert_eq!(
        structure.elements[0].kids,
        [Kid::Marked {
            page: 1,
            mcid: 1,
            stream: None
        }]
    );
}

#[test]
fn an_mcid_whose_page_nothing_names_is_left_out_by_name() {
    let structure = tagged(ROOT, "", "<< /K 6 0 R >>", &["<< /S /P /K [1 2] >>"]).unwrap();
    assert_eq!(structure.elements[0].kids, []);
    assert_eq!(structure.warnings.len(), 1, "{:?}", structure.warnings);
    assert!(
        structure.warnings[0].contains("no page") && structure.warnings[0].contains("P (6 0 R)"),
        "{:?}",
        structure.warnings
    );
}

#[test]
fn a_parent_tree_disagreeing_with_the_tree_is_reported() {
    let structure = tagged(
        ROOT,
        "/StructParents 0",
        "<< /K [6 0 R 7 0 R] /ParentTree << /Nums [0 [7 0 R 7 0 R]] >> >>",
        &[
            "<< /S /P /Pg 3 0 R /K 0 >>",
            "<< /S /Span /Pg 3 0 R /K [1 2] >>",
        ],
    )
    .unwrap();
    let joined = structure.warnings.join("\n");
    assert!(
        joined.contains("parent tree") && joined.contains("MCID 0") && joined.contains("MCID 2"),
        "{joined}"
    );
    // The tree is kept as it states its kids: the parent tree only reports.
    assert_eq!(
        structure.elements[0].kids,
        [Kid::Marked {
            page: 1,
            mcid: 0,
            stream: None
        }]
    );
}

#[test]
fn a_cycle_is_refused_by_name_and_the_rest_kept() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K 6 0 R >>",
        &[
            "<< /S /Sect /K [7 0 R] >>",
            "<< /S /P /K [6 0 R 8 0 R] >>",
            "<< /S /Span >>",
        ],
    )
    .unwrap();
    let roles: Vec<&str> = structure.elements.iter().map(|e| e.role.as_str()).collect();
    assert_eq!(roles, ["Sect", "P", "Span"]);
    assert_eq!(structure.elements[1].kids, [Kid::Element(2)]);
    assert_eq!(structure.warnings.len(), 1, "{:?}", structure.warnings);
    assert!(
        structure.warnings[0].contains("6 0 R") && structure.warnings[0].contains("cycle"),
        "{:?}",
        structure.warnings
    );
}

#[test]
fn an_element_with_two_parents_is_kept_once() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K [6 0 R 7 0 R] >>",
        &[
            "<< /S /Sect /K 8 0 R >>",
            "<< /S /Sect /K 8 0 R >>",
            "<< /S /P >>",
        ],
    )
    .unwrap();
    assert_eq!(structure.elements.len(), 3);
    assert_eq!(structure.elements[2].kids, []);
    assert!(
        structure.warnings.len() == 1 && structure.warnings[0].contains("8 0 R"),
        "{:?}",
        structure.warnings
    );
}

#[test]
fn a_broken_reference_is_left_out_by_name() {
    let structure = tagged(
        ROOT,
        "",
        "<< /K [6 0 R 40 0 R (text)] >>",
        &[
            "<< /S /P /K [<< /Type /OBJR >> 7 0 R] >>",
            "<< /Type /MCR >>",
        ],
    )
    .unwrap();
    assert_eq!(structure.roots, [0]);
    assert_eq!(structure.elements[0].kids, []);
    let joined = structure.warnings.join("\n");
    assert_eq!(structure.warnings.len(), 4, "{joined}");
    for named in ["40 0 R", "string", "/OBJR", "7 0 R"] {
        assert!(joined.contains(named), "{named}: {joined}");
    }
}

#[test]
fn a_root_that_is_no_dictionary_is_a_structure_with_a_warning() {
    let structure = tagged(ROOT, "", "[1 2]", &[]).unwrap();
    assert_eq!(structure.elements, Vec::<Element>::new());
    assert!(
        structure.warnings[0].contains("StructTreeRoot"),
        "{:?}",
        structure.warnings
    );
}
