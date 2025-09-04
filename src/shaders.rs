/// Precompiled shaders module
/// 
/// All shaders are compiled to SPIR-V at build time using build.rs and glslc.
/// This provides zero runtime compilation overhead while supporting both compute
/// and ray tracing shaders with full extension support.

use std::sync::Arc;
use vulkano::{
    device::Device,
    shader::{ShaderModule, ShaderModuleCreateInfo},
};

/// Compute shaders
pub mod compute {
    use super::*;
    
    /// Ray tracing compute shader SPIR-V bytecode (compiled by build.rs)
    const RAYTRACING_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/compute/raytracing.spv"));
    
    /// Load the precompiled ray tracing compute shader
    pub fn load_raytracing(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        load_shader_from_spirv(device, RAYTRACING_SPIRV, "raytracing compute")
    }
}

/// Ray tracing shaders
pub mod rt {
    use super::*;
    
    /// Ray generation shader SPIR-V bytecode (compiled by build.rs)
    const RAYGEN_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt/raygen.spv"));
    
    /// Closest hit shader SPIR-V bytecode (compiled by build.rs)
    const CLOSESTHIT_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt/closesthit.spv"));
    
    /// Miss shader SPIR-V bytecode (compiled by build.rs)
    const MISS_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt/miss.spv"));
    
    /// Intersection shader SPIR-V bytecode (compiled by build.rs)
    const INTERSECTION_SPIRV: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders/rt/intersection.spv"));
    
    /// Load the precompiled ray generation shader
    pub fn load_raygen(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        load_shader_from_spirv(device, RAYGEN_SPIRV, "raygen")
    }
    
    /// Load the precompiled closest hit shader
    pub fn load_closesthit(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        load_shader_from_spirv(device, CLOSESTHIT_SPIRV, "closesthit")
    }
    
    /// Load the precompiled miss shader
    pub fn load_miss(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        load_shader_from_spirv(device, MISS_SPIRV, "miss")
    }
    
    /// Load the precompiled intersection shader
    pub fn load_intersection(device: Arc<Device>) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
        load_shader_from_spirv(device, INTERSECTION_SPIRV, "intersection")
    }
}

/// Helper function to load a shader from SPIR-V bytecode
fn load_shader_from_spirv(
    device: Arc<Device>, 
    spirv_bytes: &[u8], 
    shader_name: &str
) -> Result<Arc<ShaderModule>, Box<dyn std::error::Error>> {
    // Convert bytes to u32 words (SPIR-V format)
    if spirv_bytes.len() % 4 != 0 {
        return Err(format!("Invalid SPIR-V bytecode for {}: length not multiple of 4", shader_name).into());
    }
    
    let spirv_words: Vec<u32> = spirv_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    log::info!("Loading precompiled {} shader ({} bytes SPIR-V)", shader_name, spirv_bytes.len());
    
    // SAFETY: ShaderModule::new is safe with valid SPIR-V bytecode that we've compiled
    let shader = unsafe {
        ShaderModule::new(
            device,
            ShaderModuleCreateInfo::new(&spirv_words),
        )?
    };
    
    Ok(shader)
}