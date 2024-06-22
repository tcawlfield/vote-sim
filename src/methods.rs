mod method;
mod method_tracker;
mod plurality;
mod rangevoting;
mod tallies;
mod reweighted_range;

pub use method::{ElectResult, Strategy};
pub use method_tracker::MethodTracker;
pub use plurality::Plurality;
pub use rangevoting::RangeVoting;
pub use reweighted_range::RRV;