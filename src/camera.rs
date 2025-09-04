//! Camera for ray generation and scene rendering

use glam::Vec3A;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use log::info;
use rayon::prelude::*;

use crate::ray::Ray;
use crate::hittable::{Hittable, HitRecord};
use crate::interval::Interval;
use crate::random;

/// RGB color type using Vec3A for SIMD optimization.
type Color = Vec3A;

/// Camera for ray generation and scene rendering.
///
/// Uses a pinhole camera model with support for depth of field and anti-aliasing
/// via multi-sampling. Supports all three rendering backends.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Rendered image width in pixel count
    pub image_width: u32,
    /// Rendered image height in pixel count
    pub image_height: u32,
    /// Number of random samples for each pixel (for anti-aliasing)
    pub samples_per_pixel: u32,
    /// Maximum number of ray bounces (recursion depth limit)
    pub max_depth: u32,
    /// Vertical field of view in degrees (default: 90)
    pub vfov: f32,
    /// Point camera is looking from (camera position)
    pub lookfrom: Vec3A,
    /// Point camera is looking at (look target)
    pub lookat: Vec3A,
    /// Camera-relative "up" direction vector
    pub vup: Vec3A,
    /// Variation angle of rays through each pixel (defocus blur control)
    pub defocus_angle: f32,
    /// Distance from camera lookfrom point to plane of perfect focus
    pub focus_dist: f32,
    
    /// Camera position in world space (same as lookfrom)
    center: Vec3A,
    /// World position of the top-left pixel (pixel 0,0)
    pixel00_loc: Vec3A,
    /// Offset vector from pixel to pixel horizontally (right direction)
    pixel_delta_u: Vec3A,
    /// Offset vector from pixel to pixel vertically (down direction)
    pixel_delta_v: Vec3A,
    /// Color scale factor for a sum of pixel samples (1.0 / samples_per_pixel)
    pixel_samples_scale: f32,
    /// Camera frame basis vector pointing right (u)
    u: Vec3A,
    /// Camera frame basis vector pointing up (v)  
    v: Vec3A,
    /// Camera frame basis vector pointing opposite view direction (w)
    w: Vec3A,
    /// Defocus disk horizontal radius vector
    defocus_disk_u: Vec3A,
    /// Defocus disk vertical radius vector  
    defocus_disk_v: Vec3A,
    /// Flag to track whether camera parameters have been calculated
    initialized: bool,
}

impl Camera {
    /// Initialize camera and return parameters needed for GPU rendering.
    ///
    /// Returns camera basis vectors and viewport parameters.
    pub fn ensure_initialized(&mut self) -> (Vec3A, Vec3A, Vec3A, Vec3A, Vec3A, Vec3A, Vec3A, Vec3A) {
        if !self.initialized {
            self.initialize();
        }
        (
            self.u, 
            self.v, 
            self.w, 
            self.pixel00_loc, 
            self.pixel_delta_u, 
            self.pixel_delta_v, 
            self.defocus_disk_u, 
            self.defocus_disk_v
        )
    }

    /// Creates a new camera with default settings.
    ///
    /// Default: 100x100 image, 50 samples per pixel, 90° FOV, no defocus blur.
    pub fn new() -> Self {
        Self {
            image_width: 100,
            image_height: 100,
            samples_per_pixel: 50,
            max_depth: 50,
            vfov: 90.0,
            lookfrom: Vec3A::new(0.0, 0.0, 0.0),
            lookat: Vec3A::new(0.0, 0.0, -1.0),
            vup: Vec3A::new(0.0, 1.0, 0.0),
            defocus_angle: 0.0,
            focus_dist: 10.0,
            center: Vec3A::ZERO,
            pixel00_loc: Vec3A::ZERO,
            pixel_delta_u: Vec3A::ZERO,
            pixel_delta_v: Vec3A::ZERO,
            pixel_samples_scale: 0.1,
            u: Vec3A::ZERO,
            v: Vec3A::ZERO,
            w: Vec3A::ZERO,
            defocus_disk_u: Vec3A::ZERO,
            defocus_disk_v: Vec3A::ZERO,
            initialized: false,
        }
    }

    /// Renders the scene using CPU path tracing.
    ///
    /// Generates rays through each pixel, traces them through the scene,
    /// and accumulates color samples. Uses parallel processing for performance.
    ///
    /// Returns an HDR image buffer with linear f32 RGB values.
    pub fn render(&mut self, world: &dyn Hittable) -> ImageBuffer<Rgb<f32>, Vec<f32>> {
        self.initialize();

        let mut image: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(self.image_width, self.image_height);
        
        info!("Generating image using {} CPU cores...", rayon::current_num_threads());
        let generation_start = std::time::Instant::now();
        let pb = ProgressBar::new((self.image_width * self.image_height) as u64);
        pb.set_style(ProgressStyle::default_bar().template("{bar:40} {pos}/{len} ETA: {eta}").unwrap());
        
        // Parallel pixel processing using Rayon with anti-aliasing
        image.enumerate_pixels_mut().par_bridge().for_each(|(i, j, pixel)| {
            let mut pixel_color = Color::ZERO;
            
            // Sample multiple rays per pixel for anti-aliasing
            for _sample in 0..self.samples_per_pixel {
                let r = self.get_ray(i, j);
                pixel_color += self.ray_color(&r, world, self.max_depth);
            }
            
            // Average the samples
            pixel_color *= self.pixel_samples_scale;
            *pixel = Rgb([pixel_color.x, pixel_color.y, pixel_color.z]);
            pb.inc(1);
        });
        
        pb.finish();
        let generation_time = generation_start.elapsed();
        info!("Image generated in {:.2?}", generation_time);

        image
    }

    /// Initialize camera parameters based on current settings.
    ///
    /// Sets up the camera coordinate system and viewport for ray generation.
    /// Automatically called by render() but can be called explicitly for GPU backends.
    fn initialize(&mut self) {
        if self.initialized {
            return;
        }

        // Note: image_height should be set externally, but ensure it's at least 1
        self.image_height = if self.image_height < 1 { 1 } else { self.image_height };

        self.pixel_samples_scale = 1.0 / self.samples_per_pixel as f32;

        // Set camera center to lookfrom position
        self.center = self.lookfrom;

        // Determine viewport dimensions
        let theta = self.vfov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h * self.focus_dist;
        let viewport_width = viewport_height * (self.image_width as f32 / self.image_height as f32);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame
        self.w = (self.lookfrom - self.lookat).normalize();  // Points opposite view direction
        self.u = self.vup.cross(self.w).normalize();         // Points to camera right
        self.v = self.w.cross(self.u);                       // Points to camera up

        // Calculate the vectors across the horizontal and down the vertical viewport edges
        let viewport_u = viewport_width * self.u;    // Vector across viewport horizontal edge
        let viewport_v = viewport_height * -self.v;  // Vector down viewport vertical edge (negative v)

        // Calculate the horizontal and vertical delta vectors from pixel to pixel
        self.pixel_delta_u = viewport_u / self.image_width as f32;
        self.pixel_delta_v = viewport_v / self.image_height as f32;

        // Calculate the location of the upper left pixel
        let viewport_upper_left = self.center - (self.focus_dist * self.w) - viewport_u / 2.0 - viewport_v / 2.0;
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v);

        // Calculate the camera defocus disk basis vectors
        let defocus_radius = self.focus_dist * (self.defocus_angle.to_radians() / 2.0).tan();
        self.defocus_disk_u = self.u * defocus_radius;
        self.defocus_disk_v = self.v * defocus_radius;

        self.initialized = true;
    }

    /// Generate a ray through a pixel with random sampling.
    ///
    /// Uses random sampling within the pixel for anti-aliasing and optionally
    /// samples from the defocus disk for depth-of-field blur.
    fn get_ray(&self, i: u32, j: u32) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + ((i as f32 + offset.x) * self.pixel_delta_u)
            + ((j as f32 + offset.y) * self.pixel_delta_v);

        let ray_origin = if self.defocus_angle <= 0.0 {
            self.center
        } else {
            self.defocus_disk_sample()
        };
        let ray_direction = pixel_sample - ray_origin;

        Ray::new(ray_origin, ray_direction)
    }

    /// Generate random offset within [-0.5, 0.5] square for pixel sampling.
    fn sample_square(&self) -> Vec3A {
        Vec3A::new(
            random::random_f32() - 0.5,
            random::random_f32() - 0.5,
            0.0
        )
    }

    /// Sample random point on the defocus disk for depth-of-field blur.
    fn defocus_disk_sample(&self) -> Vec3A {
        let p = random::random_in_unit_disk();
        self.center + (p.x * self.defocus_disk_u) + (p.y * self.defocus_disk_v)
    }

    /// Trace a ray and compute its color contribution.
    ///
    /// Recursively follows ray bounces through the scene, sampling materials
    /// to determine color and next ray direction. Returns sky color if no hit.
    fn ray_color(&self, r: &Ray, world: &dyn Hittable, depth: u32) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered
        if depth <= 0 {
            return Color::ZERO;
        }
        
        let mut rec = HitRecord {
            p: Vec3A::ZERO,
            normal: Vec3A::ZERO,
            t: 0.0,
            front_face: false,
            material: crate::material::MaterialType::Lambertian { albedo: Vec3A::ZERO },
        };
        
        // Test if ray hits any object in the world
        if world.hit(r, Interval::new(0.001, f32::INFINITY), &mut rec) {
            let mut attenuation = Color::ZERO;
            let mut scattered = Ray::new(Vec3A::ZERO, Vec3A::ZERO);
            
            if rec.material.scatter(r, &rec, &mut attenuation, &mut scattered) {
                return attenuation * self.ray_color(&scattered, world, depth - 1);
            }
            return Color::ZERO;
        }

        // No hit - render sky gradient
        let unit_direction = r.direction.normalize();
        // Create a blend factor based on Y component of ray direction
        // Y = -1 (down) gives a = 0, Y = 1 (up) gives a = 1
        let a = 0.5 * (unit_direction.y + 1.0);
        
        // Linear interpolation between white and light blue
        (1.0 - a) * Color::new(1.0, 1.0, 1.0) + a * Color::new(0.5, 0.7, 1.0)
    }
}