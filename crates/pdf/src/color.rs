//! Color spaces and colors (ISO 32000-2 section 8.6): the components as the file states them and, where the
//! space defines one without a color profile or a function, their RGB.

use crate::error::Result;
use crate::file::File;
use crate::object::{Dict, Object};

/// A color space, resolved far enough to count its components and convert what can be converted.
#[derive(Debug, Clone, PartialEq)]
pub enum Space {
    /// `DeviceGray` / `CalGray`.
    Gray,
    /// `DeviceRGB` / `CalRGB`.
    Rgb,
    /// `DeviceCMYK`.
    Cmyk,
    /// `ICCBased` with `n` components (converted as gray, RGB or CMYK by `n`).
    Icc(usize),
    /// `Indexed`: a base space and its lookup table.
    Indexed(Box<Space>, Vec<u8>),
    /// `Lab`, `Separation`, `DeviceN`: components kept, no RGB (it needs a tint function or a profile).
    Other(Vec<u8>, usize),
    /// `Pattern`: painted with a pattern, not a color.
    Pattern,
}

/// A color: its space's family name, its components, and its RGB when the space gives one.
#[derive(Debug, Clone, PartialEq)]
pub struct Color {
    /// The color space family (`DeviceRGB`, `ICCBased`, `Separation`, ...).
    pub space: Vec<u8>,
    /// The components, as set.
    pub components: Vec<f64>,
    /// Red, green and blue in 0..=1, when the space defines them.
    pub rgb: Option<[f64; 3]>,
}

impl Color {
    /// Its RGB when its space defines one, else its own components (a separation's tint): the most a reader can
    /// say of it without a profile or a tint function.
    #[must_use]
    pub fn rgb_or_components(&self) -> Vec<f64> {
        self.rgb
            .map_or_else(|| self.components.clone(), |rgb| rgb.to_vec())
    }
}

impl Space {
    /// How many components a color of this space has.
    #[must_use]
    pub fn components(&self) -> usize {
        match self {
            Self::Gray | Self::Indexed(..) => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
            Self::Icc(n) | Self::Other(_, n) => *n,
            Self::Pattern => 0,
        }
    }

    /// The family name.
    #[must_use]
    pub fn family(&self) -> Vec<u8> {
        match self {
            Self::Gray => b"DeviceGray".to_vec(),
            Self::Rgb => b"DeviceRGB".to_vec(),
            Self::Cmyk => b"DeviceCMYK".to_vec(),
            Self::Icc(_) => b"ICCBased".to_vec(),
            Self::Indexed(..) => b"Indexed".to_vec(),
            Self::Other(name, _) => name.clone(),
            Self::Pattern => b"Pattern".to_vec(),
        }
    }

    /// The initial color of this space (8.6.8): black, or index 0.
    #[must_use]
    pub fn initial(&self) -> Color {
        let components = match self {
            Self::Cmyk | Self::Icc(4) => vec![0.0, 0.0, 0.0, 1.0],
            Self::Other(name, n) if name == b"Separation" || name == b"DeviceN" => vec![1.0; *n],
            other => vec![0.0; other.components()],
        };
        self.color(components)
    }

    /// A color of this space from its components.
    #[must_use]
    pub fn color(&self, components: Vec<f64>) -> Color {
        Color {
            space: self.family(),
            rgb: self.rgb(&components),
            components,
        }
    }

    fn rgb(&self, c: &[f64]) -> Option<[f64; 3]> {
        let unit = |v: f64| v.clamp(0.0, 1.0);
        match (self, c) {
            (Self::Gray | Self::Icc(1), [g]) => Some([unit(*g); 3]),
            (Self::Rgb | Self::Icc(3), [r, g, b]) => Some([unit(*r), unit(*g), unit(*b)]),
            (Self::Cmyk | Self::Icc(4), [cy, m, y, k]) => {
                Some([1.0 - unit(cy + k), 1.0 - unit(m + k), 1.0 - unit(y + k)])
            }
            (Self::Indexed(base, table), [i]) => {
                let n = base.components();
                let index = i.round().max(0.0);
                // An index is a whole number in 0..=hival; the float-to-usize cast saturates.
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "rounded, non-negative"
                )]
                let start = index as usize * n;
                let entry = table.get(start..start + n)?;
                base.rgb(
                    &entry
                        .iter()
                        .map(|&b| f64::from(b) / 255.0)
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        }
    }
}

impl File {
    /// The color space `value` names or is, in the context of `resources`.
    ///
    /// # Errors
    ///
    /// Refuses a reference that does not resolve.
    pub fn color_space(&self, value: &Object, resources: &Dict) -> Result<Space> {
        self.space_at_depth(value, resources, 0)
    }

    fn space_at_depth(&self, value: &Object, resources: &Dict, depth: usize) -> Result<Space> {
        let value = self.resolve(value)?;
        if depth > 8 {
            return Ok(Space::Other(b"Unknown".to_vec(), 1));
        }
        match &value {
            Object::Name(name) => Ok(match name.as_slice() {
                b"DeviceGray" | b"G" | b"CalGray" => Space::Gray,
                b"DeviceRGB" | b"RGB" | b"CalRGB" => Space::Rgb,
                b"DeviceCMYK" | b"CMYK" => Space::Cmyk,
                b"Pattern" => Space::Pattern,
                _ => match self.get(resources, b"ColorSpace")? {
                    Some(Object::Dict(spaces)) => match spaces.get(name) {
                        Some(named) => self.space_at_depth(named, resources, depth + 1)?,
                        None => Space::Other(name.clone(), 1),
                    },
                    _ => Space::Other(name.clone(), 1),
                },
            }),
            Object::Array(items) => {
                let family = items
                    .first()
                    .and_then(Object::as_name)
                    .unwrap_or(b"")
                    .to_vec();
                Ok(match family.as_slice() {
                    b"DeviceGray" | b"CalGray" => Space::Gray,
                    b"DeviceRGB" | b"CalRGB" => Space::Rgb,
                    b"DeviceCMYK" => Space::Cmyk,
                    b"Pattern" => Space::Pattern,
                    b"Lab" => Space::Other(family, 3),
                    b"ICCBased" => {
                        let n = match items.get(1).map(|s| self.resolve(s)).transpose()? {
                            Some(Object::Stream(s)) => {
                                s.dict.get(b"N").and_then(Object::as_integer).unwrap_or(3)
                            }
                            _ => 3,
                        };
                        Space::Icc(usize::try_from(n).unwrap_or(3))
                    }
                    b"Indexed" | b"I" => {
                        let base = match items.get(1) {
                            Some(b) => self.space_at_depth(b, resources, depth + 1)?,
                            None => Space::Rgb,
                        };
                        let table = match items.get(3).map(|t| self.resolve(t)).transpose()? {
                            Some(Object::String(s)) => s,
                            Some(Object::Stream(s)) => self.decode(&s)?.into_bytes()?,
                            _ => Vec::new(),
                        };
                        Space::Indexed(Box::new(base), table)
                    }
                    b"DeviceN" => {
                        let n = match items.get(1).map(|c| self.resolve(c)).transpose()? {
                            Some(Object::Array(names)) => names.len(),
                            _ => 1,
                        };
                        Space::Other(family, n)
                    }
                    _ => Space::Other(family, 1),
                })
            }
            _ => Ok(Space::Other(b"Unknown".to_vec(), 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_colors_convert_to_rgb() {
        assert_eq!(Space::Gray.color(vec![0.5]).rgb, Some([0.5; 3]));
        assert_eq!(
            Space::Cmyk.color(vec![1.0, 0.0, 0.0, 0.0]).rgb,
            Some([0.0, 1.0, 1.0])
        );
        assert_eq!(Space::Cmyk.initial().rgb, Some([0.0; 3]));
    }

    #[test]
    fn an_indexed_color_looks_up_its_base() {
        let space = Space::Indexed(Box::new(Space::Rgb), vec![0, 0, 0, 255, 0, 0]);
        assert_eq!(space.color(vec![1.0]).rgb, Some([1.0, 0.0, 0.0]));
        assert_eq!(space.color(vec![9.0]).rgb, None);
    }

    #[test]
    fn a_separation_keeps_its_tint_and_gives_no_rgb() {
        let c = Space::Other(b"Separation".to_vec(), 1).color(vec![0.3]);
        assert_eq!((c.components, c.rgb), (vec![0.3], None));
    }

    #[test]
    fn a_color_reads_as_its_rgb_else_its_components() {
        let rgb = Space::Cmyk.color(vec![0.0, 0.0, 0.0, 1.0]);
        assert_eq!(rgb.rgb_or_components(), [0.0; 3]);
        let tint = Space::Other(b"Separation".to_vec(), 1).color(vec![0.3]);
        assert_eq!(tint.rgb_or_components(), [0.3]);
    }
}
