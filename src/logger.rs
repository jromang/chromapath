use log::LevelFilter;

/// Initialize the logger with the specified level
pub fn init_logger(level: LevelFilter) {
    env_logger::Builder::from_default_env()
        .filter_level(level)
        .init();
}