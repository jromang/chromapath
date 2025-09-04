//! Ray-object intersection system.
//!
//! Defines the Hittable trait for geometric primitives and HitRecord for
//! storing intersection data. Supports all three rendering backends.

use glam::Vec3A;
use crate::ray::Ray;
use crate::interval::Interval;
use crate::material::MaterialType;

/// Ray-object intersection information.
///
/// Contains intersection point, surface normal, distance, and material data
/// needed for shading calculations.
#[derive(Debug, Clone)]
pub struct HitRecord {
    /// Point where the ray intersects the object
    pub p: Vec3A,
    /// Surface normal at the intersection point (unit vector)
    pub normal: Vec3A,
    /// Distance along the ray to the intersection point
    pub t: f32,
    /// True if ray hits the front face, false if hits the back face
    pub front_face: bool,
    /// Material of the object at the hit point
    pub material: MaterialType,
}

impl HitRecord {
    /// Set surface normal and determine front/back face.
    ///
    /// Ensures normal always points against the incident ray.
    pub fn set_face_normal(&mut self, r: &Ray, outward_normal: Vec3A) {
        // Determine if we hit the front face by checking if ray and normal point in opposite directions
        self.front_face = r.direction.dot(outward_normal) < 0.0;
        // Always point the normal against the incident ray
        self.normal = if self.front_face {
            outward_normal
        } else {
            -outward_normal
        };
    }
}

/// Trait for objects that can be intersected by rays.
///
/// Core abstraction for geometric primitives. Must be thread-safe (Sync + Send)
/// for parallel rendering across CPU, GPU compute, and hardware RT backends.
pub trait Hittable: Sync + Send {
    /// Test for ray intersection within the given parameter range.
    ///
    /// Returns true if hit, updating the hit record with intersection details.
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool;
}

/// Collection of objects forming a scene.
///
/// Uses linear search for intersection testing. Supports polymorphic
/// objects through Box<dyn Hittable>.
pub struct HittableList {
    /// Vector of boxed hittable objects
    pub objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    /// Create a new empty scene.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
        }
    }

    /// Create a new list with a single hittable object
    pub fn _new_with_object(object: Box<dyn Hittable>) -> Self {
        let mut list = Self::new();
        list.add(object);
        list
    }

    /// Clear all objects from the list
    pub fn _clear(&mut self) {
        self.objects.clear();
    }

    /// Add an object to the scene.
    pub fn add(&mut self, object: Box<dyn Hittable>) {
        self.objects.push(object);
    }
}

impl Hittable for HittableList {
    fn hit(&self, r: &Ray, ray_t: Interval, rec: &mut HitRecord) -> bool {
        let mut temp_rec = HitRecord {
            p: Vec3A::ZERO,
            normal: Vec3A::ZERO,
            t: 0.0,
            front_face: false,
            material: MaterialType::Lambertian { albedo: Vec3A::ZERO },
        };
        let mut hit_anything = false;
        let mut closest_so_far = ray_t.max;

        // Test intersection with each object in the list
        for object in &self.objects {
            if object.hit(r, Interval::new(ray_t.min, closest_so_far), &mut temp_rec) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                *rec = temp_rec.clone();
            }
        }

        hit_anything
    }
}