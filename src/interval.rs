//! Interval arithmetic for ray parameter ranges.
//!
//! Provides closed intervals [min, max] used for ray t-values and bounds checking.

/// Closed interval [min, max] for range checking.
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    /// Minimum value of the interval
    pub min: f32,
    /// Maximum value of the interval
    pub max: f32,
}

impl Interval {
    /// Create a new interval with given min and max values
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    /// Create an empty interval (min > max)
    pub fn _empty() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    /// Create a universe interval (contains all real numbers)
    pub fn _universe() -> Self {
        Self {
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
        }
    }

    /// Calculate the size (width) of the interval
    pub fn _size(&self) -> f32 {
        self.max - self.min
    }

    /// Check if the interval contains the given value (inclusive bounds)
    pub fn _contains(&self, x: f32) -> bool {
        self.min <= x && x <= self.max
    }

    /// Check if the interval surrounds the given value (exclusive bounds)
    pub fn surrounds(&self, x: f32) -> bool {
        self.min < x && x < self.max
    }

    /// Clamp the given value to be within this interval's bounds
    pub fn _clamp(&self, x: f32) -> f32 {
        x.clamp(self.min, self.max)
    }
}

/// Commonly used interval constants
impl Interval {
    /// Empty interval constant
    pub const _EMPTY: Interval = Interval {
        min: f32::INFINITY,
        max: f32::NEG_INFINITY,
    };

    /// Universe interval constant  
    pub const _UNIVERSE: Interval = Interval {
        min: f32::NEG_INFINITY,
        max: f32::INFINITY,
    };
}