//! # Output Module
//! 
//! This module provides functionality for outputting rendered images in various formats:
//! - Real-time visualization via TEV (The EXR Viewer)
//! - PNG file export with HDR to LDR conversion
//! 
//! ## TEV Integration
//! 
//! TEV is a high-performance image viewer designed for HDR images. This module handles:
//! - Network connection to TEV client
//! - Image format conversion (interleaved RGB to channel-wise layout)
//! - Real-time image updates for progressive rendering
//! 
//! ## PNG Export
//! 
//! Converts f32 HDR images to standard u8 PNG format with proper tone mapping:
//! - Clamps values to [0.0, 1.0] range
//! - Linear scaling from floating-point to 8-bit integers
//! - Error handling for file I/O operations

use log::{info, warn, debug};
use image::{ImageBuffer, Rgb};
use tev_client::{TevClient, PacketCreateImage, PacketUpdateImage};
use std::net::TcpStream;
use exr::prelude::*;

/// Send an f32 RGB image to TEV for real-time visualization
/// 
/// This function establishes a TCP connection to a TEV client and sends the provided
/// image data for real-time display. It handles the complete workflow:
/// 
/// 1. Network connection with TCP_NODELAY for reduced latency
/// 2. TEV image creation with proper channel configuration
/// 3. Data format conversion from interleaved RGB to channel-wise layout
/// 4. Optimized data transmission with performance timing
/// 
/// # Arguments
/// 
/// * `image` - f32 RGB image buffer with values typically in [0.0, 1.0] range
/// * `tev_address` - TEV server address (IP:port or just IP, defaults to port 14158)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// 
/// # Performance
/// 
/// The function includes timing information and reports:
/// - Data preparation time (typically < 50ms)
/// - Network transmission time (depends on image size and network speed)
/// - Total data size in MB
/// 
/// For a 512×512 image (~3.1MB), expect 1-3 seconds transmission time depending
/// on TEV processing speed and network conditions.
/// 
/// # Examples
/// 
/// ```ignore
/// use image::{ImageBuffer, Rgb};
/// 
/// let image: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(512, 512);
/// send_image_to_tev(&image, "localhost:14158", 512, 512);
/// send_image_to_tev(&image, "192.168.1.100", 512, 512); // Uses default port 14158
/// ```
pub fn send_image_to_tev(image: &ImageBuffer<Rgb<f32>, Vec<f32>>, tev_address: &str, width: u32, height: u32) {
    // Add default port if not specified
    let tev_address = if tev_address.contains(':') {
        tev_address.to_string()
    } else {
        format!("{}:14158", tev_address)
    };
    
    debug!("Attempting to connect to TEV at {}", tev_address);
    
    match TcpStream::connect(&tev_address) {
        Ok(stream) => {
            // Configure TCP socket for optimal performance
            if let Err(e) = stream.set_nodelay(true) {
                debug!("Failed to set TCP_NODELAY: {}", e);
            }
            
            debug!("TCP connection established successfully");
            let mut client = TevClient::wrap(stream);
            
            // Create image in TEV
            let create_packet = PacketCreateImage {
                image_name: "chromapath_output",
                width,
                height,
                channel_names: &["R", "G", "B"],
                grab_focus: true,
            };
            
            match client.send(create_packet) {
                Ok(_) => debug!("Image created in TEV successfully"),
                Err(e) => {
                    warn!("Failed to create image in TEV: {}", e);
                    return;
                }
            }
            
            // Convert image data from interleaved (RGBRGB...) to planar (RRR...GGG...BBB...) for TEV
            let data_prep_start = std::time::Instant::now();
            let pixel_count = (width * height) as usize;
            let mut rgb_data = Vec::with_capacity(pixel_count * 3);
            
            // First pass: collect all R values
            for pixel in image.pixels() {
                rgb_data.push(pixel[0]);
            }
            // Second pass: collect all G values  
            for pixel in image.pixels() {
                rgb_data.push(pixel[1]);
            }
            // Third pass: collect all B values
            for pixel in image.pixels() {
                rgb_data.push(pixel[2]);
            }
            
            debug!("Data preparation completed in {:.2?}", data_prep_start.elapsed());
            debug!("Sending {} pixels to TEV ({:.1} MB)", rgb_data.len() / 3, rgb_data.len() as f32 * 4.0 / 1_000_000.0);
            let start_time = std::time::Instant::now();
            
            // Update image with pixel data
            let update_packet = PacketUpdateImage {
                image_name: "chromapath_output",
                grab_focus: false,
                channel_names: &["R", "G", "B"],
                x: 0,
                y: 0,
                width,
                height,
                channel_offsets: &[0, (width * height) as u64, (2 * width * height) as u64],
                channel_strides: &[1, 1, 1],
                data: &rgb_data,
            };
            
            match client.send(update_packet) {
                Ok(_) => {
                    let elapsed = start_time.elapsed();
                    info!("Image data sent to TEV at {} successfully in {:.2?}", tev_address, elapsed);
                },
                Err(e) => warn!("Failed to send image data to TEV: {}", e),
            }
        },
        Err(e) => warn!("Failed to connect to TEV on {}: {}", tev_address, e),
    }
}

/// Save an f32 RGB image as PNG with HDR to LDR tone mapping and gamma correction
/// 
/// This function converts a high dynamic range (HDR) f32 image to a standard
/// 8-bit PNG file. The conversion process includes:
/// 
/// 1. Value clamping to [0.0, 1.0] range to handle out-of-gamut values
/// 2. Gamma correction (linear to gamma 2.0) for proper display
/// 3. Linear scaling from floating-point to 8-bit integers [0, 255]
/// 4. PNG encoding and file I/O with error handling
/// 
/// # Arguments
/// 
/// * `image` - f32 RGB image buffer, typically containing HDR values
/// * `output_path` - File path for the output PNG (should include .png extension)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// 
/// # Tone Mapping and Gamma Correction
/// 
/// The implementation applies sRGB gamma correction for proper PNG display:
/// - Values > 1.0 are clamped to 1.0 (overexposed areas become white)
/// - Values < 0.0 are clamped to 0.0 (underexposed areas become black)
/// - sRGB gamma correction with linear portion for dark values (< 0.0031308)
/// - Power curve: `1.055 * linear^(1/2.4) - 0.055` for brighter values
/// - Linear scaling: `output = gamma_corrected * 255.0`
/// 
/// # Examples
/// 
/// ```ignore
/// use image::{ImageBuffer, Rgb};
/// 
/// let image: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(512, 512);
/// save_image_as_png(&image, "output.png", 512, 512);
/// save_image_as_png(&image, "render_001.png", 512, 512);
/// ```
/// 
/// # Errors
/// 
/// Logs warnings for I/O errors but does not panic. Common error causes:
/// - Invalid file path or insufficient permissions
/// - Disk space issues
/// - Invalid image dimensions
pub fn save_image_as_png(image: &ImageBuffer<Rgb<f32>, Vec<f32>>, output_path: &str, width: u32, height: u32) {
    let u8_image: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = image.get_pixel(x, y);
        
        // Apply gamma correction for proper PNG display
        // sRGB standard gamma correction with linear portion for dark values
        let linear_to_gamma = |linear: f32| -> f32 {
            if linear <= 0.0 {
                0.0
            } else if linear <= 0.0031308 {
                // Linear portion for very dark values
                12.92 * linear
            } else {
                // Gamma curve: 1.055 * linear^(1/2.4) - 0.055
                1.055 * linear.powf(1.0 / 2.4) - 0.055
            }
        };
        
        Rgb([
            (linear_to_gamma(pixel[0].clamp(0.0, 1.0)) * 255.0) as u8,
            (linear_to_gamma(pixel[1].clamp(0.0, 1.0)) * 255.0) as u8,
            (linear_to_gamma(pixel[2].clamp(0.0, 1.0)) * 255.0) as u8,
        ])
    });
    
    match u8_image.save(output_path) {
        Ok(_) => info!("Image saved as {}", output_path),
        Err(e) => warn!("Failed to save image: {}", e),
    }
}

/// Save an f32 RGB image as EXR with full HDR precision
/// 
/// This function saves a high dynamic range (HDR) f32 image to OpenEXR format,
/// preserving the full linear light values without any tone mapping or gamma correction.
/// This is ideal for:
/// 
/// - Professional workflows requiring HDR data
/// - Viewing with TEV (The EXR Viewer) which handles display transforms
/// - Post-processing with tone mapping, color grading, or compositing
/// - Archival storage of ray-traced images with full dynamic range
/// 
/// # Arguments
/// 
/// * `image` - f32 RGB image buffer containing linear HDR values
/// * `output_path` - File path for the output EXR (should include .exr extension)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// 
/// # Format Details
/// 
/// The EXR file is saved with:
/// - 32-bit floating-point precision per channel
/// - RGB channels (Red, Green, Blue)
/// - Linear light values (no gamma correction)
/// - ZIP compression for reasonable file sizes
/// - Standard OpenEXR headers for compatibility
/// 
/// # Examples
/// 
/// ```ignore
/// use image::{ImageBuffer, Rgb};
/// 
/// let image: ImageBuffer<Rgb<f32>, Vec<f32>> = ImageBuffer::new(512, 512);
/// save_image_as_exr(&image, "output.exr", 512, 512);
/// save_image_as_exr(&image, "render_hdr.exr", 512, 512);
/// ```
/// 
/// # Errors
/// 
/// Logs warnings for I/O errors but does not panic. Common error causes:
/// - Invalid file path or insufficient permissions
/// - Disk space issues
/// - Invalid image dimensions
/// - EXR library internal errors
pub fn save_image_as_exr(image: &ImageBuffer<Rgb<f32>, Vec<f32>>, output_path: &str, width: u32, height: u32) {
    // Create RGB pixels vector
    let pixels = image.pixels()
        .map(|rgb| (rgb[0], rgb[1], rgb[2]))
        .collect::<Vec<(f32, f32, f32)>>();
    
    // Create EXR image using simple RGB API
    let result = write_rgb_file(
        output_path,
        width as usize, height as usize,
        |x, y| {
            let index = y * (width as usize) + x;
            pixels[index]
        }
    );
    
    match result {
        Ok(_) => info!("HDR image saved as EXR: {}", output_path),
        Err(e) => warn!("Failed to save EXR image: {}", e),
    }
}