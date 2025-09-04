use clap::{Parser, ValueEnum};
use log::LevelFilter;

/// Custom enum for log levels that can be used with clap's ValueEnum
#[derive(Debug, Clone, ValueEnum)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Convert our custom LogLevel enum to log crate's LevelFilter
impl From<LogLevel> for LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Error => LevelFilter::Error,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Info => LevelFilter::Info,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Trace => LevelFilter::Trace,
        }
    }
}

/// Command line arguments structure using clap derive macros
#[derive(Parser)]
#[command(name = "chromapath")]
#[command(about = "A simple path tracer in Rust")]
pub struct Args {
    #[arg(short, long)]
    pub verbose: bool,
    
    #[arg(short, long, default_value = "config.toml")]
    pub config: String,
    
    /// Set the logging level (defaults to "info")
    #[arg(long, default_value = "info", help = "Set the logging level")]
    pub debug_level: LogLevel,
    
    /// Image width in pixels
    #[arg(long, default_value = "800", help = "Image width in pixels")]
    pub width: u32,
    
    /// Image height in pixels
    #[arg(long, default_value = "600", help = "Image height in pixels")]
    pub height: u32,
    
    /// Number of samples per pixel
    #[arg(long, short = 's', default_value = "100", help = "Number of samples per pixel")]
    pub samples_per_pixel: u32,
    
    /// Send image to TEV for real-time visualization
    #[arg(long, help = "Send image to TEV for real-time visualization")]
    pub tev: bool,
    
    /// TEV client IP address and port (automatically enables --tev)
    #[arg(long, help = "TEV client IP address and port (automatically enables --tev)")]
    pub tev_address: Option<String>,
    
    /// Output file path (.png for 8-bit with gamma correction, .exr for HDR linear)
    #[arg(short, long, default_value = "output.png", help = "Output file path (.png for 8-bit with gamma correction, .exr for HDR linear)")]
    pub output: String,
    
    /// Use GPU compute shaders for rendering (Vulkan-based)
    #[arg(long = "compute", help = "Use GPU compute shaders for rendering (Vulkan-based)")]
    pub gpu: bool,
    
    /// Test hardware ray tracing acceleration structure support
    #[arg(long, help = "Test hardware ray tracing acceleration structure support")]
    pub test_rt: bool,
    
    /// Use hardware-accelerated ray tracing (requires RT-capable GPU: RTX, RX 6000+, Arc)
    #[arg(long = "hw-rt", short = 'r', help = "Use hardware-accelerated ray tracing (requires RT-capable GPU: RTX, RX 6000+, Arc)")]
    pub hardware_rt: bool,
    
    /// Run benchmark comparing CPU, Compute, and Hardware RT (500 samples, 3 output images)
    #[arg(long, help = "Run benchmark comparing CPU, Compute, and Hardware RT (500 samples, 3 output images)")]
    pub bench: bool,
    
    pub files: Vec<String>,
}