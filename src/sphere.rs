//! Sphere primitive for ray tracing.
//!
//! Implements efficient ray-sphere intersection using an optimized quadratic formula.

use glam::Vec3A;
use crate::ray::Ray;
use crate::hittable::{Hittable, HitRecord};
use crate::interval::Interval;
use crate::material::MaterialType;

/// Sphere primitive defined by center, radius, and material.
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Center point of the sphere in world coordinates.
    pub center: Vec3A,
    
    /// Radius of the sphere (always non-negative).
    ///
    /// Negative radius values are clamped to 0.0 in the constructor.
    pub radius: f32,
    
    /// Material properties determining light interaction.
    pub material: MaterialType,
}

impl Sphere {
    /// Create a new sphere.
    ///
    /// Negative radius values are clamped to 0.0.
    pub fn new(center: Vec3A, radius: f32, material: MaterialType) -> Self {
        Self {
            center,
            radius: radius.max(0.0), // Ensure radius is non-negative
            material,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        // Vector from ray origin to sphere center
        let oc = self.center - r.origin;
        
        // Optimized quadratic equation coefficients
        let a = r.direction.length_squared();
        let h = r.direction.dot(oc);
        let c = oc.length_squared() - self.radius * self.radius;
        
        // Calculate discriminant
        let discriminant = h * h - a * c;
        if discriminant < 0.0 {
            return false;
        }
        
        let sqrtd = discriminant.sqrt();
        
        // Find the nearest root that lies in the acceptable range
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return false;
            }
        }
        
        // Fill the hit record
        rec.t = root;
        rec.p = r.at(rec.t);
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(r, outward_normal);
        rec.material = self.material;
        
        true
    }
}

