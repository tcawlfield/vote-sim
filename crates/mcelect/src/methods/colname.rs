// © Copyright 2026 Topher Cawlfield
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

/// An optional replacement for a method's default output column name, set with
/// `colname` in the method's config table:
///
/// ```toml
/// [[methods]]
/// Range = { strat = "Strategic", nranks = 10, strategic_stretch_factor = 2.0, colname = "range_s_2x" }
/// ```
///
/// It's only a label, so it never affects whether two methods are equal:
/// `PartialEq` on a method means "elects the same way", which is how the runner
/// matches a strategic method to its honest poll.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ColName(Option<String>);

impl PartialEq for ColName {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl ColName {
    /// True when no name is configured, so the method's default applies.
    pub fn is_default(&self) -> bool {
        self.0.is_none()
    }

    /// The configured name, or else `default()`.
    pub fn or_else(&self, default: impl FnOnce() -> String) -> String {
        self.0.clone().unwrap_or_else(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_configured_name_replaces_the_default() {
        assert_eq!(ColName::default().or_else(|| "pl_h".into()), "pl_h");
        assert_eq!(
            ColName(Some("mine".into())).or_else(|| "pl_h".into()),
            "mine"
        );
    }

    #[test]
    fn names_never_affect_equality() {
        assert_eq!(ColName(Some("a".into())), ColName(Some("b".into())));
        assert_eq!(ColName(Some("a".into())), ColName::default());
    }
}
