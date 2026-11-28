//! Transparency (ISO 32000-2 section 11): transparency groups and soft masks.
//!
//! A group's marks are drawn on a canvas of their own over the group's box, then the result is composited onto what
//! lies below with the group's alpha, blend mode and soft mask (11.6.6). An isolated group starts from a transparent
//! backdrop; a non-isolated one from what lies below, and its result is that backdrop's contribution taken out again
//! (11.4.8: `C = Cn + (Cn - C0) (a0 / ag - a0)`, the group's own alpha `ag` from its marks drawn alone) -- over an
//! opaque backdrop with `Normal` blending this is the group's canvas mixed with the backdrop by the group's alpha,
//! which needs no second drawing. In a knockout group each mark is composited onto the group's backdrop rather than
//! onto the marks before it, and replaces them where it lies by its shape -- its coverage, read as its alpha drawn
//! alone over its constant alpha (a mark with a soft mask or an image with an alpha of its own knocks out by that
//! alpha too). A group drawn opaquely, not knockout and -- when isolated -- holding only `Normal` marks, is its marks
//! drawn as they come: the same pixels. Blending is in sRGB, whatever the group's `/CS`.
//!
//! A soft mask (11.6.5.2) is its group drawn on a canvas of the page's pixels, under the matrix in force where the
//! mask was set: for a luminosity mask over its backdrop (`/BC`, opaque; black when not stated) and read as each
//! pixel's luminosity (0.30 red + 0.59 green + 0.11 blue, as the `Luminosity` blend mode reckons it), for an alpha
//! mask over transparency and read as each pixel's alpha; then through its transfer function (`/TR`). What a mark
//! shows is its coverage times the mask's value.

use std::rc::Rc;

use pdf::function::Function;
use pdf::geometry::Rect;
use pdf::icc::Intent;
use pdf::interpret::{Blend, Event, Group, SoftMask};
use tiny_skia::{
    FilterQuality, IntSize, Mask, Pixmap, PixmapPaint, PremultipliedColorU8, Transform,
};

use crate::paint::NESTING;
use crate::{Canvas, Grid, blend_mode};

/// What names a soft mask made: its group (by object, else by address), the matrix it was set under, and how it is
/// read.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MaskKey {
    group: (Option<u32>, usize),
    matrix: [u64; 6],
    luminosity: bool,
    backdrop: Vec<u64>,
    transfer: String,
}

impl MaskKey {
    fn of(mask: &SoftMask) -> Self {
        Self {
            group: match mask.object {
                Some(n) => (Some(n), 0),
                None => (None, Rc::as_ptr(&mask.group) as usize),
            },
            matrix: mask.matrix.0.map(f64::to_bits),
            luminosity: mask.luminosity,
            backdrop: mask.backdrop.iter().map(|v| v.to_bits()).collect(),
            transfer: format!("{:?}", mask.transfer),
        }
    }
}

/// The group of the chain `chain` (innermost first) directly within `within` (None: the page's); None when the
/// chain's innermost group is `within` itself, or the chain is not within it.
pub(crate) fn child_of(chain: Option<&Rc<Group>>, within: Option<&Rc<Group>>) -> Option<Rc<Group>> {
    let mut child: Option<&Rc<Group>> = None;
    let mut at = chain;
    while let Some(group) = at {
        if within.is_some_and(|w| Rc::ptr_eq(w, group)) {
            return child.cloned();
        }
        child = Some(group);
        at = group.compositing.group.as_ref();
    }
    if within.is_none() {
        child.cloned()
    } else {
        None
    }
}

/// Whether `group` is in the chain `chain`.
pub(crate) fn holds(chain: Option<&Rc<Group>>, group: &Rc<Group>) -> bool {
    let mut at = chain;
    while let Some(g) = at {
        if Rc::ptr_eq(g, group) {
            return true;
        }
        at = g.compositing.group.as_ref();
    }
    false
}

/// The constant alpha `event` is painted at: its fill's, its stroke's, or the greater when it paints both.
pub(crate) fn alpha_of(event: &Event) -> f64 {
    let c = event.compositing();
    match event {
        Event::Path(p) => match (p.fill.is_some(), p.stroke.is_some()) {
            (true, true) => c.fill_alpha.max(c.stroke_alpha),
            (false, true) => c.stroke_alpha,
            _ => c.fill_alpha,
        },
        Event::Text(t) => match t.render_mode {
            1 | 5 => c.stroke_alpha,
            2 | 6 => c.fill_alpha.max(c.stroke_alpha),
            _ => c.fill_alpha,
        },
        Event::Image(_) | Event::Shading(_) => c.fill_alpha,
    }
}

/// Whether every mark of `events` within `group`, and every group between, blends `Normal` and knocks nothing out:
/// an isolated group of them composites as its marks do.
fn normal_within(events: &[Event], group: &Rc<Group>) -> bool {
    events.iter().all(|e| {
        let c = e.compositing();
        let mut at = c.group.as_ref();
        let mut normal = c.blend == Blend::Normal;
        while let Some(g) = at.filter(|g| !Rc::ptr_eq(g, group)) {
            normal &= !g.knockout && g.compositing.blend == Blend::Normal;
            at = g.compositing.group.as_ref();
        }
        normal
    })
}

/// Two masks' product, pixel by pixel.
pub(crate) fn product(a: &Mask, b: &Mask) -> Mask {
    let mut out = a.clone();
    for (o, m) in out.data_mut().iter_mut().zip(b.data()) {
        *o = mul(*o, *m);
    }
    out
}

/// `a * b / 255`, rounded.
fn mul(a: u8, b: u8) -> u8 {
    let v = u32::from(a) * u32::from(b) + 128;
    #[expect(clippy::cast_possible_truncation, reason = "at most 255")]
    let out = ((v + (v >> 8)) >> 8) as u8;
    out
}

impl<'a> Canvas<'a> {
    /// One element (a mark, or a group) drawn by `draw` at constant alpha `alpha`: as it comes, or in a knockout
    /// group composited onto the group's backdrop and replacing what lies below by its shape -- both drawn over the
    /// pixels of its `extent` alone (page space, [`Canvas::extent`]; None: it paints nothing), a pixel wider for
    /// anti-aliasing, so a group of many small marks costs their areas, not their count times the group's.
    pub(crate) fn element(
        &mut self,
        alpha: f64,
        extent: Option<Rect>,
        draw: &dyn Fn(&mut Canvas<'a>),
    ) {
        let Some(backdrop) = self.knockout.clone() else {
            draw(self);
            return;
        };
        let Some(area) = extent.and_then(|e| self.grid.pixels(e.grown(1.0 / self.grid.scale)))
        else {
            return; // it paints nothing here
        };
        let window = self.grid.window(area);
        let start = backdrop.clone_rect(area).expect("within the canvas");
        let mut over = self.child(window, start);
        draw(&mut over);
        let onto = self.adopt(over);
        // Its shape: drawn alone, counted once.
        let undrawn = self.kept.undrawn.clone();
        let blank = Pixmap::new(area.width(), area.height()).expect("the area's size");
        let mut alone = self.child(window, blank);
        draw(&mut alone);
        let alone = self.adopt(alone);
        self.kept.undrawn = undrawn;
        let (stride, across) = (self.pixmap.width() as usize, area.width() as usize);
        #[expect(clippy::cast_sign_loss, reason = "within the canvas")]
        let (left, top) = (area.x() as usize, area.y() as usize);
        let rows = onto
            .pixels()
            .chunks_exact(across)
            .zip(alone.pixels().chunks_exact(across));
        for (row, (overs, shapes)) in rows.enumerate() {
            let at = (top + row) * stride + left;
            let outs = &mut self.pixmap.pixels_mut()[at..at + across];
            for ((out, over), shape) in outs.iter_mut().zip(overs).zip(shapes) {
                let a = f64::from(shape.alpha()) / 255.0;
                let f = if alpha > 0.0 {
                    (a / alpha).min(1.0)
                } else {
                    0.0
                };
                if f > 0.0 {
                    *out = mix(*out, *over, f);
                }
            }
        }
    }

    /// The run `events` of `group`'s marks drawn as the group (see the module).
    pub(crate) fn group(&mut self, group: &Rc<Group>, events: &[Event]) {
        let c = &group.compositing;
        if !group.knockout && !c.translucent() && (!group.isolated || normal_within(events, group))
        {
            self.draw(events, Some(group));
            return;
        }
        let Some(area) = self.grid.pixels(group.bbox) else {
            return;
        };
        let mask = match &c.soft_mask {
            Some(soft) => self.soft_mask(soft),
            None => None,
        };
        let backdrop = self.pixmap.clone_rect(area).expect("within the canvas");
        let grid = self.grid.sub(area);
        let blank = || Pixmap::new(area.width(), area.height()).expect("the area's size");
        let start = if group.isolated {
            blank()
        } else {
            backdrop.clone()
        };
        let drawn = self.pass(group, events, grid, start);
        let opaque = backdrop.pixels().iter().all(|p| p.alpha() == 255);
        let result = if group.isolated || (c.blend == Blend::Normal && opaque) {
            drawn
        } else {
            let undrawn = self.kept.undrawn.clone();
            let alone = self.pass(group, events, grid, blank());
            self.kept.undrawn = undrawn;
            unbackdrop(&drawn, &backdrop, &alone)
        };
        #[expect(clippy::cast_possible_truncation, reason = "an alpha")]
        let paint = PixmapPaint {
            opacity: c.fill_alpha as f32,
            blend_mode: blend_mode(c.blend),
            quality: FilterQuality::Nearest,
        };
        self.pixmap.draw_pixmap(
            area.x(),
            area.y(),
            result.as_ref(),
            &paint,
            Transform::identity(),
            mask.as_deref(),
        );
    }

    /// `group`'s `events` drawn onto `start`, on `grid`.
    fn pass(&mut self, group: &Rc<Group>, events: &[Event], grid: Grid, start: Pixmap) -> Pixmap {
        let mut canvas = self.child(grid, start);
        if group.knockout {
            canvas.knockout = Some(Rc::new(canvas.pixmap.clone()));
        }
        canvas.draw(events, Some(group));
        self.adopt(canvas)
    }

    /// The soft mask `mask` over this canvas, made once; None when it cannot be made (counted: what it masks is
    /// drawn without it).
    pub(crate) fn soft_mask(&mut self, mask: &Rc<SoftMask>) -> Option<Rc<Mask>> {
        let key = MaskKey::of(mask);
        if let Some(made) = self.masks.get(&key) {
            return made.clone();
        }
        let made = match self.make_mask(mask) {
            Ok(m) => Some(Rc::new(m)),
            Err(why) => {
                self.undrawn(&format!("soft mask not drawn ({why}); drawn without it"));
                None
            }
        };
        self.masks.insert(key, made.clone());
        made
    }

    fn make_mask(&mut self, mask: &SoftMask) -> Result<Mask, String> {
        if self.kept.nesting >= NESTING {
            return Err("patterns and soft masks nest too deep".to_string());
        }
        let transfer = mask
            .transfer
            .as_ref()
            .map(|t| self.file.function(t))
            .transpose()
            .map_err(|e| e.to_string())?;
        let table = transfer_table(transfer.as_ref())?;
        let (width, height) = (self.grid.width, self.grid.height);
        let mut pixmap = Pixmap::new(width, height).expect("the canvas's size");
        if mask.luminosity {
            let [r, g, b] = mask
                .backdrop_rgb(self.file, Intent::default())
                .map_err(|e| e.to_string())?;
            #[expect(clippy::cast_possible_truncation, reason = "color components")]
            pixmap.fill(
                tiny_skia::Color::from_rgba(r as f32, g as f32, b as f32, 1.0)
                    .unwrap_or(tiny_skia::Color::BLACK),
            );
        }
        let grid = self.grid;
        let content = self
            .file
            .form_drawing(
                &mask.group,
                &mask.resources,
                mask.matrix,
                grid.page_box(),
                &|m| grid.glyph_placed(m),
            )
            .map_err(|e| e.to_string())?;
        self.kept.nesting += 1;
        let mut canvas = self.child(grid, pixmap);
        canvas.content(&content);
        let pixmap = self.adopt(canvas);
        self.kept.nesting -= 1;
        let data = pixmap
            .pixels()
            .iter()
            .map(|p| {
                let v = if mask.luminosity {
                    luminosity(*p)
                } else {
                    p.alpha()
                };
                table[usize::from(v)]
            })
            .collect();
        Mask::from_vec(
            data,
            IntSize::from_wh(width, height).expect("the canvas's size"),
        )
        .ok_or_else(|| "a mask of the wrong size".to_string())
    }
}

/// The transfer function `transfer` (None: the identity) at each byte value, as a byte.
fn transfer_table(transfer: Option<&Function>) -> Result<[u8; 256], String> {
    let mut table = [0u8; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        #[expect(clippy::cast_possible_truncation, reason = "a byte")]
        let byte = i as u8;
        *slot = match transfer {
            None => byte,
            Some(f) => {
                let x = f64::from(byte) / 255.0;
                let y = f.evaluate(&[x]).map_err(|e| e.to_string())?;
                let y = y
                    .first()
                    .copied()
                    .ok_or("a transfer function of no output")?;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "a byte"
                )]
                let out = (y.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                out
            }
        };
    }
    Ok(table)
}

/// A pixel's luminosity, its color read as on an opaque backdrop.
fn luminosity(p: PremultipliedColorU8) -> u8 {
    let c = p.demultiply();
    let y = 77 * u32::from(c.red()) + 151 * u32::from(c.green()) + 28 * u32::from(c.blue()) + 128;
    #[expect(clippy::cast_possible_truncation, reason = "at most 255")]
    let y = (y >> 8) as u8;
    // A pixel not wholly opaque shows the black under it in proportion.
    mul(y, p.alpha())
}

/// `a` moved toward `b` by `f` in 0..=1, premultiplied.
#[expect(clippy::many_single_char_names, reason = "a mix's terms")]
fn mix(a: PremultipliedColorU8, b: PremultipliedColorU8, f: f64) -> PremultipliedColorU8 {
    let m = |x: u8, y: u8| {
        let v = f64::from(x) + (f64::from(y) - f64::from(x)) * f;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte"
        )]
        let out = v.round().clamp(0.0, 255.0) as u8;
        out
    };
    let alpha = m(a.alpha(), b.alpha());
    let c = |x: u8, y: u8| m(x, y).min(alpha);
    PremultipliedColorU8::from_rgba(
        c(a.red(), b.red()),
        c(a.green(), b.green()),
        c(a.blue(), b.blue()),
        alpha,
    )
    .expect("each channel within its alpha")
}

/// A non-isolated group's result (11.4.8): `drawn` (its marks over `backdrop`) with the backdrop taken out again,
/// at the group's own alpha, `alone`'s (its marks over transparency).
fn unbackdrop(drawn: &Pixmap, backdrop: &Pixmap, alone: &Pixmap) -> Pixmap {
    let mut out = Pixmap::new(drawn.width(), drawn.height()).expect("the area's size");
    for (((o, d), b), a) in out
        .pixels_mut()
        .iter_mut()
        .zip(drawn.pixels())
        .zip(backdrop.pixels())
        .zip(alone.pixels())
    {
        let ag = f64::from(a.alpha()) / 255.0;
        if ag <= 0.0 {
            continue;
        }
        let a0 = f64::from(b.alpha()) / 255.0;
        let an = f64::from(d.alpha()) / 255.0;
        let straight = |v: u8, alpha: f64| {
            if alpha > 0.0 {
                f64::from(v) / 255.0 / alpha
            } else {
                0.0
            }
        };
        let channel = |dv: u8, bv: u8| {
            let (cn, c0) = (straight(dv, an), straight(bv, a0));
            let c = (cn + (cn - c0) * (a0 / ag - a0)).clamp(0.0, 1.0);
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a byte"
            )]
            let out = (c * ag * 255.0 + 0.5) as u8;
            out
        };
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a byte"
        )]
        let alpha = (ag * 255.0 + 0.5) as u8;
        let ch = |dv: u8, bv: u8| channel(dv, bv).min(alpha);
        *o = PremultipliedColorU8::from_rgba(
            ch(d.red(), b.red()),
            ch(d.green(), b.green()),
            ch(d.blue(), b.blue()),
            alpha,
        )
        .expect("each channel within its alpha");
    }
    out
}
