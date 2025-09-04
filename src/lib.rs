//! ChromaPath path tracer
//!
//! Supports three rendering backends: CPU, GPU compute, and hardware ray tracing.
//! Outputs PNG and EXR formats with optional TEV viewer integration.

#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod ray;
pub mod sphere;
pub mod hittable;
pub mod interval;
pub mod camera;
pub mod random;
pub mod material;