//! Random number generation for ray tracing.
//!
//! Provides thread-safe random number generation with ChaCha20 PRNG.
//! Includes specialized sampling functions for spheres, hemispheres, and colors.

#![allow(dead_code)]

use rand::{Rng, SeedableRng, rng};
use rand_chacha::ChaCha20Rng;
use std::cell::RefCell;
use glam::Vec3A;

thread_local! {
    /// Thread-local ChaCha20 PRNG for quality random numbers.
    static RNG: RefCell<ChaCha20Rng> = RefCell::new(ChaCha20Rng::from_rng(&mut rng()));
}

/// Generate a random f32 in [0.0, 1.0)
pub fn random_f32() -> f32 {
    RNG.with(|rng| rng.borrow_mut().random())
}

/// Generate a random f32 in [min, max)
pub fn random_f32_range(min: f32, max: f32) -> f32 {
    min + (max - min) * random_f32()
}

/// Generate a random f64 in [0.0, 1.0)  
pub fn random_f64() -> f64 {
    RNG.with(|rng| rng.borrow_mut().random())
}

/// Generate a random f64 in [min, max)
pub fn random_f64_range(min: f64, max: f64) -> f64 {
    min + (max - min) * random_f64()
}

/// Generate a random Vec3A with components in [0.0, 1.0) - optimized SIMD version
pub fn random_vec3a() -> Vec3A {
    RNG.with(|rng| rng.borrow_mut().random())
}

/// Generate random Vec3A with components in [min, max) using SIMD operations.
pub fn random_vec3a_range(min: f32, max: f32) -> Vec3A {
    let random_vec = random_vec3a();
    Vec3A::splat(min) + (Vec3A::splat(max - min) * random_vec)
}

/// Generate multiple random Vec3A at once for better SIMD performance
pub fn random_vec3a_batch(count: usize) -> Vec<Vec3A> {
    let mut batch = Vec::with_capacity(count);
    RNG.with(|rng| {
        let mut rng_mut = rng.borrow_mut();
        for _ in 0..count {
            batch.push(rng_mut.random());
        }
    });
    batch
}

/// Generate multiple random Vec3A in a range at once for better SIMD performance
pub fn random_vec3a_range_batch(min: f32, max: f32, count: usize) -> Vec<Vec3A> {
    let min_vec = Vec3A::splat(min);
    let scale_vec = Vec3A::splat(max - min);
    
    random_vec3a_batch(count)
        .into_iter()
        .map(|v| min_vec + scale_vec * v)
        .collect()
}

/// Generate random unit vector uniformly distributed on unit sphere.
pub fn random_unit_vector() -> Vec3A {
    RNG.with(|rng| {
        let mut rng_mut = rng.borrow_mut();
        
        // Generate uniform θ in [0, 2π)
        let theta = 2.0 * std::f32::consts::PI * rng_mut.random::<f32>();
        
        // Generate uniform cos(φ) in [-1, 1] for proper sphere distribution
        let cos_phi = 2.0 * rng_mut.random::<f32>() - 1.0;
        let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
        
        // Convert to Cartesian coordinates
        Vec3A::new(
            sin_phi * theta.cos(),
            sin_phi * theta.sin(),
            cos_phi
        )
    })
}

/// Generate random vector on hemisphere oriented by the given normal.
pub fn random_on_hemisphere(normal: Vec3A) -> Vec3A {
    let on_unit_sphere = random_unit_vector();
    if on_unit_sphere.dot(normal) > 0.0 {
        // In the same hemisphere as the normal
        on_unit_sphere
    } else {
        // Flip to the correct hemisphere
        -on_unit_sphere
    }
}

/// Generate random point inside unit disk using rejection sampling.
pub fn random_in_unit_disk() -> Vec3A {
    loop {
        let p = Vec3A::new(
            random_f32_range(-1.0, 1.0),
            random_f32_range(-1.0, 1.0), 
            0.0
        );
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

/// Generate random RGB color with components in [0.0, 1.0).
pub fn random_color() -> Vec3A {
    Vec3A::new(random_f32(), random_f32(), random_f32())
}

/// Generate random RGB color with components in [min, max).
pub fn random_color_range(min: f32, max: f32) -> Vec3A {
    Vec3A::new(
        random_f32_range(min, max),
        random_f32_range(min, max),
        random_f32_range(min, max)
    )
}

/// Normalize vector to unit length.
pub fn unit_vector(v: Vec3A) -> Vec3A {
    v.normalize()
}