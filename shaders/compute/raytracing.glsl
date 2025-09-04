#version 450

layout(local_size_x = 16, local_size_y = 16) in;

struct RenderParams {
    uint image_width;
    uint image_height;
    uint samples_per_pixel;
    uint max_depth;
    uint sphere_count;
    uint batch_offset;        // Offset for random seed variation between batches
    uint padding1;
    uint padding2;
    
    // Camera position and orientation (vec3 + padding for alignment)
    vec4 camera_center;       // Camera position (lookfrom)
    vec4 camera_w;            // Camera look direction (-w vector)
    vec4 camera_u;            // Camera right direction (u vector)
    vec4 camera_v;            // Camera up direction (v vector)
    
    // Viewport and projection parameters
    vec4 pixel00_loc;         // First pixel location (pixel00_loc)
    vec4 pixel_delta_u;       // Pixel horizontal delta (pixel_delta_u)
    vec4 pixel_delta_v;       // Pixel vertical delta (pixel_delta_v)
    
    // Defocus blur parameters
    vec4 defocus_disk_u;      // Defocus disk horizontal radius vector
    vec4 defocus_disk_v;      // Defocus disk vertical radius vector
    float defocus_angle;      // Defocus angle (for enabling/disabling blur)
    float focus_dist;         // Focus distance
    float padding3;           // Additional padding to maintain alignment
    float padding4;           // Final padding to maintain alignment
};

struct Sphere {
    vec4 center;
    uint material_type;
    uint padding1[3];
    vec4 material_data;
};

layout(set = 0, binding = 0, std430) readonly buffer RenderParamsBuffer {
    RenderParams render_params;
};

layout(set = 0, binding = 1, std430) readonly buffer SphereBuffer {
    Sphere spheres[];
};

layout(set = 0, binding = 2, std430) writeonly buffer OutputBuffer {
    vec4 output_image[];
};

// Ray representation for ray tracing
struct Ray {
    vec3 origin;              // Ray starting point
    vec3 direction;           // Ray direction (should be normalized)
};

// Ray-surface intersection result
struct HitRecord {
    vec3 point;               // Intersection point in world space
    vec3 normal;              // Surface normal at intersection
    float t;                  // Distance along ray to intersection
    bool front_face;          // Whether ray hits front face of surface
    uint material_type;       // Material type at intersection
    vec4 material_data;       // Material properties at intersection
};

// Camera ray generation
// Creates rays from camera through pixels for primary ray casting

// Random number generation for GPU shaders
// Hash function based PRNG suitable for parallel execution
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

uint hash_combine(uint x, uint y) {
    return hash(x ^ (y + 0x9e3779b9u + (x << 6u) + (x >> 2u)));
}

float random_float(uint seed) {
    return float(hash(seed)) / float(0xFFFFFFFFu);
}

vec2 random_in_unit_disk(uint seed) {
    for (int i = 0; i < 8; i++) { // Limit iterations to prevent infinite loop
        vec2 p = 2.0 * vec2(random_float(seed + uint(i * 2)), random_float(seed + uint(i * 2 + 1))) - 1.0;
        if (dot(p, p) < 1.0) {
            return p;
        }
    }
    return vec2(0.0); // Fallback
}

/// Generate ray through pixel with sub-pixel jittering and defocus blur
/// @param i: pixel x coordinate
/// @param j: pixel y coordinate  
/// @param sample_idx: sample index for random seed variation
Ray get_ray_jittered(uint i, uint j, uint sample_idx) {
    // Create unique seed for this pixel and sample
    uint pixel_seed = hash_combine(hash_combine(i, j), sample_idx + render_params.batch_offset);
    
    // Sub-pixel jittering for anti-aliasing
    float jitter_x = random_float(pixel_seed) - 0.5;
    float jitter_y = random_float(pixel_seed + 1u) - 0.5;
    
    // Calculate pixel sample position using camera parameters
    vec3 pixel_sample = render_params.pixel00_loc.xyz
                      + ((float(i) + jitter_x) * render_params.pixel_delta_u.xyz)
                      + ((float(j) + jitter_y) * render_params.pixel_delta_v.xyz);
    
    // Ray origin with defocus blur
    vec3 ray_origin;
    if (render_params.defocus_angle <= 0.0) {
        // No defocus blur - rays from camera center
        ray_origin = render_params.camera_center.xyz;
    } else {
        // Defocus blur - sample from defocus disk
        vec2 disk_sample = random_in_unit_disk(pixel_seed + 2u);
        ray_origin = render_params.camera_center.xyz 
                   + (disk_sample.x * render_params.defocus_disk_u.xyz) 
                   + (disk_sample.y * render_params.defocus_disk_v.xyz);
    }
    
    // Create ray from origin to pixel sample
    Ray ray;
    ray.origin = ray_origin;
    ray.direction = normalize(pixel_sample - ray_origin);
    return ray;
}

// Geometry intersection functions
// Ray-sphere intersection using quadratic formula with numerical stability improvements

/// Test ray-sphere intersection and return hit information
/// @param sphere: sphere to test intersection with
/// @param ray: ray to test
/// @param t_min: minimum valid intersection distance
/// @param t_max: maximum valid intersection distance
/// @return: HitRecord with intersection details (t=-1 if no hit)
HitRecord hit_sphere(Sphere sphere, Ray ray, float t_min, float t_max) {
    vec3 center = sphere.center.xyz;
    float radius = sphere.center.w;
    vec3 oc = center - ray.origin;  // Vector from ray origin to sphere center
    
    // Quadratic equation coefficients for ray-sphere intersection
    // |ray.origin + t*ray.direction - center|² = radius²
    float a = dot(ray.direction, ray.direction);  // Should be 1.0 for normalized rays
    float h = dot(ray.direction, oc);            // Half-b for numerical stability  
    float c = dot(oc, oc) - radius * radius;
    
    float discriminant = h * h - a * c;
    
    HitRecord hit_record;
    hit_record.t = -1.0; // Invalid hit marker
    
    // No intersection if discriminant < 0
    if (discriminant < 0.0) {
        return hit_record;
    }
    
    // Find nearest valid intersection point
    float sqrtd = sqrt(discriminant);
    float root = (h - sqrtd) / a;  // Try nearest intersection first
    if (root < t_min || root > t_max) {
        root = (h + sqrtd) / a;  // Try farther intersection
        if (root < t_min || root > t_max) {
            return hit_record;   // No valid intersection in range
        }
    }
    
    // Fill hit record with intersection details
    hit_record.t = root;
    hit_record.point = ray.origin + root * ray.direction;
    vec3 outward_normal = (hit_record.point - center) / radius;
    
    // Determine which side of surface was hit (important for refraction)
    bool front_face = dot(ray.direction, outward_normal) < 0.0;
    hit_record.front_face = front_face;
    hit_record.normal = normalize(front_face ? outward_normal : -outward_normal);
    hit_record.material_type = sphere.material_type;
    hit_record.material_data = sphere.material_data;
    
    return hit_record;
}

HitRecord hit_world(Ray ray, float t_min, float t_max) {
    HitRecord closest_hit;
    closest_hit.t = -1.0;
    float closest_t = t_max;
    
    for (uint i = 0u; i < render_params.sphere_count; i++) {
        HitRecord hit = hit_sphere(spheres[i], ray, t_min, closest_t);
        if (hit.t > 0.0) {
            closest_hit = hit;
            closest_t = hit.t;
        }
    }
    
    return closest_hit;
}

// Hash functions for pseudo-random number generation
float hash11(float p) {
    uint p_int = floatBitsToUint(p);
    p_int = (p_int << 13u) ^ p_int;
    p_int = p_int * (p_int * p_int * 15731u + 789221u) + 1376312589u;
    return float(p_int & 0x7fffffffu) / float(0x7fffffff);
}

vec3 hash33(vec3 p) {
    uvec3 q = uvec3(floatBitsToUint(p.x), floatBitsToUint(p.y), floatBitsToUint(p.z));
    q *= uvec3(1597334673u, 3812015801u, 2798796415u);
    q = (q.zxy ^ q.yzx) * uvec3(1597334673u, 3812015801u, 2798796415u);
    return vec3(q & uvec3(0x7fffffffu)) / float(0x7fffffff);
}

Ray scatter_lambertian(HitRecord hit, uint bounce) {
    // Use hash-based randomness
    vec3 seed = vec3(hit.point.x + float(bounce) * 0.1, hit.point.y + float(bounce) * 0.2, hit.point.z + float(bounce) * 0.3);
    vec3 random_vec = hash33(seed) * 2.0 - 1.0; // [-1,1]
    vec3 scatter_direction = hit.normal + normalize(random_vec);
    
    Ray scattered;
    scattered.origin = hit.point; // No offset like CPU
    scattered.direction = normalize(length(scatter_direction) > 0.001 ? scatter_direction : hit.normal);
    return scattered;
}

Ray scatter_metal(HitRecord hit, vec3 ray_direction, float fuzz, uint bounce) {
    vec3 reflected = reflect(normalize(ray_direction), hit.normal);
    
    // Hash-based fuzz randomness
    vec3 seed = vec3(hit.point.x + float(bounce) * 0.5, hit.point.y + float(bounce) * 0.7, hit.point.z + float(bounce) * 0.9);
    vec3 random_fuzz = normalize(hash33(seed) * 2.0 - 1.0);
    
    Ray scattered;
    scattered.origin = hit.point; // No offset like CPU
    scattered.direction = normalize(reflected + fuzz * random_fuzz);
    return scattered;
}

float reflectance(float cosine, float refraction_index) {
    // Schlick's approximation for reflectance
    float r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

vec3 refract_vec(vec3 incident, vec3 normal, float etai_over_etat) {
    float cos_theta = min(dot(-incident, normal), 1.0);
    vec3 r_out_perp = etai_over_etat * (incident + cos_theta * normal);
    vec3 r_out_parallel = -sqrt(abs(1.0 - dot(r_out_perp, r_out_perp))) * normal;
    return r_out_perp + r_out_parallel;
}

Ray scatter_dielectric(HitRecord hit, vec3 ray_direction, float refraction_index, uint bounce) {
    vec3 unit_direction = normalize(ray_direction);
    
    float refraction_ratio = hit.front_face ? (1.0 / refraction_index) : refraction_index;
    
    float cos_theta = min(dot(-unit_direction, hit.normal), 1.0);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    
    // Hash for random reflectance decision
    vec3 seed = vec3(hit.point.x + float(bounce) * 0.3, hit.point.y + float(bounce) * 0.6, hit.point.z + float(bounce) * 0.9);
    float random_val = hash11(seed.x + seed.y + seed.z);
    
    Ray scattered;
    
    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_val) {
        // Reflect - no offset like CPU
        scattered.direction = reflect(unit_direction, hit.normal);
        scattered.origin = hit.point;
    } else {
        // Refract - no offset like CPU
        scattered.direction = refract_vec(unit_direction, hit.normal, refraction_ratio);
        scattered.origin = hit.point;
    }
    
    return scattered;
}

vec3 ray_color(Ray initial_ray) {
    Ray ray = initial_ray;
    vec3 absorption = vec3(1.0, 1.0, 1.0);
    
    // Iterative ray bouncing (max 50 bounces) - match CPU depth
    for (int bounce = 0; bounce <= 50; bounce++) {
        // If we've exceeded the ray bounce limit without hitting a light source, no light is gathered
        if (bounce == 50) {
            return vec3(0.0, 0.0, 0.0);
        }
        
        HitRecord hit = hit_world(ray, 0.001, 1000.0);
        
        if (hit.t > 0.0) {
            // Material scattering
            vec3 attenuation;
            Ray scattered;
            
            if (hit.material_type == 0u) {
                // Lambertian
                attenuation = hit.material_data.rgb;
                scattered = scatter_lambertian(hit, uint(bounce));
            } else if (hit.material_type == 1u) {
                // Metal
                attenuation = hit.material_data.rgb;
                scattered = scatter_metal(hit, ray.direction, hit.material_data.w, uint(bounce));
            } else {
                // Dielectric - proper refraction/reflection
                attenuation = vec3(1.0, 1.0, 1.0); // Glass doesn't absorb light
                scattered = scatter_dielectric(hit, ray.direction, hit.material_data.w, uint(bounce));
            }
            
            // Apply material attenuation
            absorption = absorption * attenuation;
            
            // Continue with scattered ray
            ray = scattered;
            
            // No Russian roulette - continue with next bounce
        } else {
            // Sky gradient - ray missed all objects
            vec3 unit_direction = normalize(ray.direction);
            float t = 0.5 * (unit_direction.y + 1.0);
            vec3 sky_color = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            
            return absorption * sky_color;
        }
    }
    
    // If we exhausted bounces without hitting sky, return black
    return vec3(0.0, 0.0, 0.0);
}

// Main compute shader entry point
// Each invocation processes one pixel of the output image
// Uses 16x16 workgroups for optimal GPU utilization

/// Compute shader main function - renders one pixel
void main() {
    uint pixel_x = gl_GlobalInvocationID.x;
    uint pixel_y = gl_GlobalInvocationID.y;
    
    // Bounds check - don't process pixels outside image
    if (pixel_x >= render_params.image_width || pixel_y >= render_params.image_height) {
        return;
    }
    
    // Convert 2D pixel coordinates to 1D array index
    uint pixel_index = pixel_y * render_params.image_width + pixel_x;
    
    // Multi-sampling anti-aliasing (MSAA)
    // Accumulate color from multiple sub-pixel samples
    vec3 pixel_color = vec3(0.0, 0.0, 0.0);
    
    for (uint s = 0u; s < render_params.samples_per_pixel; s++) {
        // Cast ray through pixel with jittering and defocus blur
        Ray ray = get_ray_jittered(pixel_x, pixel_y, s);
        
        // Trace ray through scene and accumulate color
        pixel_color = pixel_color + ray_color(ray);
    }
    
    // Average all samples for final pixel color
    pixel_color = pixel_color / float(render_params.samples_per_pixel);
    
    // Store final color in output buffer (RGB + alpha=1.0)
    output_image[pixel_index] = vec4(pixel_color.r, pixel_color.g, pixel_color.b, 1.0);
}