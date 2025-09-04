//! Ray representation for 3D ray tracing.
//!
//! A ray is defined as r(t) = origin + t * direction, representing a semi-infinite
//! line in 3D space used for intersection testing.

use glam::Vec3A;

/// Ray in 3D space defined by origin and direction.
///
/// Mathematical representation: r(t) = origin + t * direction
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Starting point of the ray in world coordinates.
    ///
    /// This represents the ray's origin, typically the camera position for
    /// primary rays or a surface point for secondary rays.
    pub origin: Vec3A,
    
    /// Direction vector of the ray.
    ///
    /// While not required to be normalized, a unit vector simplifies distance
    /// calculations. Non-normalized directions are useful for certain algorithms
    /// like ray differentials or cone tracing.
    pub direction: Vec3A,
}

impl Ray {
    /// Create a new ray with origin and direction.
    pub fn new(origin: Vec3A, direction: Vec3A) -> Self {
        Self { origin, direction }
    }
    
    /// Compute a point at parameter t along the ray.
    ///
    /// Returns r(t) = origin + t * direction.
    pub fn at(&self, t: f32) -> Vec3A {
        self.origin + t * self.direction
    }
}