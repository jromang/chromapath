//! Material system for ray tracing.
//!
//! Implements three material types: Lambertian (diffuse), Metal (specular),
//! and Dielectric (transparent). GPU-compatible with all rendering backends.

use glam::Vec3A;
use crate::ray::Ray;
use crate::hittable::HitRecord;
use crate::random;

/// RGB color type using Vec3A for SIMD optimization.
pub type Color = Vec3A;

/// Material types for ray tracing.
///
/// GPU-compatible enum representing different surface materials.
/// Supports diffuse, metallic, and transparent materials.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum MaterialType {
    /// Lambertian diffuse material for matte surfaces.
    Lambertian { 
        /// Surface color/reflectance.
        albedo: Vec3A 
    },
    
    /// Metallic material with specular reflection.
    Metal { 
        /// Metal color.
        albedo: Vec3A,
        /// Surface roughness (0.0 = mirror, 1.0 = rough).
        fuzz: f32 
    },
    
    /// Dielectric (transparent) material with refraction.
    Dielectric { 
        /// Index of refraction (1.0 = air, 1.5 = glass, etc.).
        refraction_index: f32 
    },
}

impl MaterialType {
    /// Compute ray scattering for this material.
    ///
    /// Returns true if the ray scatters, false if absorbed.
    /// Sets attenuation color and scattered ray direction.
    pub fn scatter(
        &self,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        match self {
            MaterialType::Lambertian { albedo } => {
                self.scatter_lambertian(*albedo, rec, attenuation, scattered)
            }
            MaterialType::Metal { albedo, fuzz } => {
                self.scatter_metal(*albedo, *fuzz, r_in, rec, attenuation, scattered)
            }
            MaterialType::Dielectric { refraction_index } => {
                self.scatter_dielectric(*refraction_index, r_in, rec, attenuation, scattered)
            }
        }
    }

    /// Lambertian diffuse scattering with cosine-weighted distribution.
    fn scatter_lambertian(
        &self,
        albedo: Vec3A,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let mut scatter_direction = rec.normal + random::random_unit_vector();
        
        // Catch degenerate scatter direction (very close to zero)
        if scatter_direction.length_squared() < 1e-8 {
            scatter_direction = rec.normal;
        }
        
        *scattered = Ray::new(rec.p, scatter_direction);
        *attenuation = albedo;
        true
    }

    /// Metallic reflection with optional surface roughness.
    fn scatter_metal(
        &self,
        albedo: Vec3A,
        fuzz: f32,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        let reflected = reflect(r_in.direction, rec.normal);
        let reflected = reflected.normalize() + (fuzz.min(1.0) * random::random_unit_vector());
        *scattered = Ray::new(rec.p, reflected);
        *attenuation = albedo;
        scattered.direction.dot(rec.normal) > 0.0
    }

    /// Dielectric scattering with reflection and refraction using Fresnel equations.
    fn scatter_dielectric(
        &self,
        refraction_index: f32,
        r_in: &Ray,
        rec: &HitRecord,
        attenuation: &mut Color,
        scattered: &mut Ray,
    ) -> bool {
        *attenuation = Vec3A::ONE; // Glass doesn't attenuate light
        
        let ri = if rec.front_face {
            1.0 / refraction_index
        } else {
            refraction_index
        };
        
        let unit_direction = r_in.direction.normalize();
        let cos_theta = (-unit_direction).dot(rec.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        
        let cannot_refract = ri * sin_theta > 1.0;
        
        let direction = if cannot_refract || reflectance(cos_theta, ri) > random::random_f32() {
            reflect(unit_direction, rec.normal)
        } else {
            refract(unit_direction, rec.normal, ri)
        };
        
        *scattered = Ray::new(rec.p, direction);
        true
    }
}

/// Reflect a vector off a surface using the law of reflection.
fn reflect(v: Vec3A, n: Vec3A) -> Vec3A {
    v - 2.0 * v.dot(n) * n
}

/// Refract a vector through an interface using Snell's law.
fn refract(uv: Vec3A, n: Vec3A, etai_over_etat: f32) -> Vec3A {
    let cos_theta = (-uv).dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel = -(1.0 - r_out_perp.length_squared()).abs().sqrt() * n;
    r_out_perp + r_out_parallel
}

/// Compute Fresnel reflectance using Schlick's approximation.
fn reflectance(cosine: f32, refraction_index: f32) -> f32 {
    let r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    let r0 = r0 * r0;
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}