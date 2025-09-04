use std::process::Command;
use std::path::Path;
use std::fs;

fn main() {
    // Capture Git values during compilation (not at runtime)
    let git_hash = std::process::Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let git_date = std::process::Command::new("git")
        .args(&["log", "-1", "--format=%ci"])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Embed these values as constants in the binary
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=GIT_DATE={}", git_date);
    
    // Compile shaders
    compile_shaders();
}

fn compile_shaders() {
    println!("cargo:rerun-if-changed=shaders/");
    
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shader_out_dir = Path::new(&out_dir).join("shaders");
    
    // Create output directories
    fs::create_dir_all(shader_out_dir.join("compute")).expect("Failed to create compute output dir");
    fs::create_dir_all(shader_out_dir.join("rt")).expect("Failed to create rt output dir");
    
    // Compile compute shader
    compile_shader(
        "shaders/compute/raytracing.glsl",
        &shader_out_dir.join("compute/raytracing.spv").to_string_lossy(),
        "compute"
    );
    
    // Compile ray tracing shaders (using correct glslc stage names)
    compile_shader(
        "shaders/rt/raygen.glsl", 
        &shader_out_dir.join("rt/raygen.spv").to_string_lossy(),
        "rgen"
    );
    
    compile_shader(
        "shaders/rt/closesthit.glsl",
        &shader_out_dir.join("rt/closesthit.spv").to_string_lossy(), 
        "rchit"
    );
    
    compile_shader(
        "shaders/rt/miss.glsl",
        &shader_out_dir.join("rt/miss.spv").to_string_lossy(),
        "rmiss"
    );
    
    compile_shader(
        "shaders/rt/intersection.glsl",
        &shader_out_dir.join("rt/intersection.spv").to_string_lossy(),
        "rint"
    );
    
    println!("✅ All shaders compiled successfully to {}", shader_out_dir.display());
}

fn compile_shader(input: &str, output: &str, stage: &str) {
    if !Path::new(input).exists() {
        panic!("Shader source file not found: {}", input);
    }
    
    println!("🔧 Compiling {} shader: {} -> {}", stage, input, output);
    
    let result = Command::new("glslc")
        .arg(&format!("-fshader-stage={}", stage))
        .arg("-O") // Optimize
        .arg("--target-env=vulkan1.2")
        .arg("--target-spv=spv1.5")
        .arg(input)
        .arg("-o")
        .arg(output)
        .status()
        .expect("Failed to execute glslc. Make sure it's installed and in PATH.");
    
    if !result.success() {
        panic!("Shader compilation failed for {}", input);
    }
    
    // Verify output file was created
    if !Path::new(output).exists() {
        panic!("Expected output file not created: {}", output);
    }
    
    let size = fs::metadata(output).unwrap().len();
    println!("   ✅ {} compiled ({} bytes)", output, size);
}