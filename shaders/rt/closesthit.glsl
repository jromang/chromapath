#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

struct RayPayload {
    vec3 color;
    uint depth;
    uint seed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

hitAttributeEXT vec2 attribs;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 3, set = 0) readonly buffer SphereBuffer {
    // Sphere data: center.xyz, radius, materialType, materialData.xyzw
    float sphereData[];
};

// Random number generation
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

vec3 randomUnitVector(inout uint seed) {
    return normalize(randomInUnitSphere(seed));
}

vec3 reflectRay(vec3 v, vec3 n) {
    return v - 2.0 * dot(v, n) * n;
}

vec3 refract_vec(vec3 uv, vec3 n, float etai_over_etat) {
    float cos_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float r_out_parallel_length_sq = 1.0 - dot(r_out_perp, r_out_perp);
    // Clamp to avoid negative sqrt (numerical errors near critical angle)
    vec3 r_out_parallel = -sqrt(max(0.0, r_out_parallel_length_sq)) * n;
    return r_out_perp + r_out_parallel;
}

float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance
    float r0 = (1.0 - ref_idx) / (1.0 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
}

void main() {
    // Check recursion depth to prevent infinite recursion
    if (payload.depth >= 10u) {
        payload.color = vec3(0.0);
        return;
    }
    
    // Get sphere data - gl_PrimitiveID is the AABB index in the BLAS
    uint sphereIndex = gl_PrimitiveID;
    
    // Safety check to avoid out-of-bounds access  
    if (sphereIndex >= 500u) {  // Safe limit for ~500 spheres
        payload.color = vec3(1.0, 0.0, 1.0); // Magenta for error
        return;
    }
    
    uint dataOffset = sphereIndex * 10u; // 10 floats per sphere
    
    // Read complete sphere data for shading
    vec3 sphereCenter = vec3(0.0);
    float sphereRadius = 1.0;
    uint materialType = 0u;
    vec3 materialData = vec3(0.5, 0.5, 0.5);
    float materialParam = 0.0;
    
    // Read all sphere data with bounds checking
    if (dataOffset + 9u < 5000u) {  // Increased limit for safety
        sphereCenter = vec3(sphereData[dataOffset], sphereData[dataOffset + 1u], sphereData[dataOffset + 2u]);
        sphereRadius = sphereData[dataOffset + 3u];
        materialType = uint(sphereData[dataOffset + 4u]);
        // skip padding at dataOffset + 5
        materialData = vec3(sphereData[dataOffset + 6u], sphereData[dataOffset + 7u], sphereData[dataOffset + 8u]);
        materialParam = sphereData[dataOffset + 9u];
    }
    
    // Calculate hit point and normal
    vec3 hitPoint = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3 normal = normalize(hitPoint - sphereCenter);
    
    // Check if we're hitting from inside
    bool frontFace = dot(gl_WorldRayDirectionEXT, normal) < 0.0;
    if (!frontFace) {
        normal = -normal;
    }
    
    // Use the seed from the payload
    uint seed = payload.seed;
    
    vec3 scattered;
    vec3 attenuation = materialData;
    bool shouldScatter = true;
    
    // Handle different material types
    if (materialType == 0u) {
        // Lambertian material
        vec3 scatterDirection = normal + randomUnitVector(seed);
        if (length(scatterDirection) < 1e-8) {
            scatterDirection = normal;
        }
        scattered = normalize(scatterDirection);
        attenuation = materialData;
    } else if (materialType == 1u) {
        // Metal material
        vec3 reflected = reflectRay(normalize(gl_WorldRayDirectionEXT), normal);
        float fuzz = materialParam; // fuzz parameter
        scattered = normalize(reflected + fuzz * randomInUnitSphere(seed));
        attenuation = materialData;
        shouldScatter = dot(scattered, normal) > 0.0;
    } else if (materialType == 2u) {
        // Dielectric material
        float refractionIndex = materialParam;
        float refractionRatio = frontFace ? (1.0 / refractionIndex) : refractionIndex;
        
        vec3 unitDirection = normalize(gl_WorldRayDirectionEXT);
        float cosTheta = min(dot(-unitDirection, normal), 1.0);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta)); // Ensure non-negative
        
        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        
        if (cannotRefract || reflectance(cosTheta, refractionRatio) > random(seed)) {
            // Reflect
            scattered = reflectRay(unitDirection, normal);
            // Ensure scattered ray is normalized and valid
            if (length(scattered) < 0.001) {
                scattered = normal; // Fallback to normal if reflection fails
            } else {
                scattered = normalize(scattered);
            }
        } else {
            // Refract
            scattered = normalize(refract_vec(unitDirection, normal, refractionRatio));
        }
        // Allow colored glass through materialData, or default to clear glass
        attenuation = length(materialData) > 0.001 ? materialData : vec3(1.0, 1.0, 1.0);
    } else {
        // Unknown material, default to Lambert
        vec3 scatterDirection = normal + randomUnitVector(seed);
        if (length(scatterDirection) < 1e-8) {
            scatterDirection = normal;
        }
        scattered = normalize(scatterDirection);
        attenuation = materialData;
    }
    
    // Trace bounce ray if we have depth left and material scatters
    if (payload.depth < 8u && shouldScatter) {
        // Update payload for bounce ray
        payload.depth = payload.depth + 1u;
        payload.seed = seed;
        
        // Calculate proper offset direction based on scattered ray
        // For dielectrics, the scattered ray might go in opposite direction to normal
        vec3 offsetDir = dot(scattered, normal) > 0.0 ? normal : -normal;
        
        // Trace bounce ray (this will modify payload.color)
        traceRayEXT(
            topLevelAS,
            gl_RayFlagsNoneEXT,
            0xff,
            0,
            0,
            0,
            hitPoint + offsetDir * 0.001,  // Offset in correct direction
            0.001,
            scattered,
            10000.0,
            0  // Use same payload location
        );
        
        // Combine bounce color with material attenuation
        payload.color = attenuation * payload.color;
    } else {
        // Max depth reached or material doesn't scatter, return dark
        payload.color = vec3(0.0, 0.0, 0.0);
    }
    
    // Update seed
    payload.seed = seed;
}