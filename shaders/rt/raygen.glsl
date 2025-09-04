#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

// Complete bindings for full ray tracing
layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0, rgba32f) uniform image2D image;

layout(binding = 2, set = 0) uniform RenderParams {
    uint imageWidth;
    uint imageHeight;
    uint samplesPerPixel;
    uint maxDepth;
    uint frameNumber;
    uint sphereCount;
    uint padding1;
    uint padding2;
    
    // Camera parameters (vec4 for better alignment)
    vec4 cameraOrigin;       // xyz + padding
    vec4 cameraLowerLeft;    // xyz + padding
    vec4 cameraHorizontal;   // xyz + padding
    vec4 cameraVertical;     // xyz + padding
    
    // Defocus parameters  
    float lensRadius;
    float focusDist;
    float padding3;
    float padding4;
    vec4 defocusDiskU;       // xyz + padding
    vec4 defocusDiskV;       // xyz + padding
} params;

struct RayPayload {
    vec3 color;
    uint depth;
    uint seed;
};

layout(location = 0) rayPayloadEXT RayPayload payload;

// Random number generation (simple hash-based PRNG)
uint hash(uint x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

float random(inout uint seed) {
    seed = hash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

vec3 randomInUnitSphere(inout uint seed) {
    // Use rejection sampling with safety limit to avoid infinite loops
    vec3 p;
    for (int i = 0; i < 100; i++) {
        p = 2.0 * vec3(random(seed), random(seed), random(seed)) - vec3(1.0);
        if (dot(p, p) < 1.0) {
            return p;
        }
    }
    // Fallback if we couldn't find a point (should rarely happen)
    return vec3(0.0, 0.0, 0.0);
}

vec3 randomInUnitDisk(inout uint seed) {
    // Use rejection sampling with safety limit to avoid infinite loops
    vec2 p;
    for (int i = 0; i < 100; i++) {
        p = 2.0 * vec2(random(seed), random(seed)) - vec2(1.0);
        if (dot(p, p) < 1.0) {
            return vec3(p, 0.0);
        }
    }
    // Fallback if we couldn't find a point (should rarely happen)
    return vec3(0.0);
}

void main() {
    // Get pixel coordinates with anti-aliasing
    const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + vec2(0.5);
    const vec2 inUV = pixelCenter/vec2(gl_LaunchSizeEXT.xy);
    
    // Initialize random seed
    uint seed = uint(gl_LaunchIDEXT.x) * 1973u + uint(gl_LaunchIDEXT.y) * 9277u + params.frameNumber * 26699u;
    
    vec3 color = vec3(0.0);
    
    // Multiple samples per pixel for anti-aliasing
    for (uint s = 0u; s < params.samplesPerPixel; s++) {
        // Generate random offset for anti-aliasing
        vec2 uv = inUV + vec2(random(seed) - 0.5, random(seed) - 0.5) / vec2(gl_LaunchSizeEXT.xy);
        
        // Flip Y coordinate to match correct orientation
        uv.y = 1.0 - uv.y;
        
        // Calculate pixel position on focal plane
        vec3 target = params.cameraLowerLeft.xyz + 
                      uv.x * params.cameraHorizontal.xyz + 
                      uv.y * params.cameraVertical.xyz;
        
        // Apply defocus blur if lens radius > 0
        vec3 origin;
        if (params.lensRadius > 0.0) {
            // Sample random point on defocus disk
            vec3 rd = randomInUnitDisk(seed);
            vec3 offset = params.defocusDiskU.xyz * rd.x + params.defocusDiskV.xyz * rd.y;
            origin = params.cameraOrigin.xyz + offset;
        } else {
            // No defocus blur - ray originates from camera center
            origin = params.cameraOrigin.xyz;
        }
        
        vec3 direction = normalize(target - origin);
        
        // Initialize payload for primary ray
        payload.color = vec3(0.0);
        payload.depth = 0u;
        payload.seed = seed;
        
        // Trace primary ray
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsNoneEXT,
            0xff,
            0,
            0,
            0,
            origin,
            0.001,
            direction,
            10000.0,
            0
        );
        
        color += payload.color;
        seed = payload.seed; // Update seed for next sample
    }
    
    // Average the samples
    color /= float(params.samplesPerPixel);
    
    // Clamp values to prevent overflow but keep in linear space
    // The gamma correction will be applied when saving to PNG
    color = clamp(color, 0.0, 10.0);  // Allow HDR values up to 10.0
    
    // Write final color to output image in linear space
    // Gamma correction should be handled by the output stage
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(color, 1.0));
}