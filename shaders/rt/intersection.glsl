#version 460
#extension GL_EXT_ray_tracing : require

hitAttributeEXT vec2 attribs;

layout(binding = 2, set = 0) uniform RenderParams {
    uint imageWidth;
    uint imageHeight;
    uint samplesPerPixel;
    uint maxDepth;
    uint frameNumber;
    uint sphereCount;
    uint padding1;
    uint padding2;
} params;

layout(binding = 3, set = 0) readonly buffer SphereBuffer {
    // Sphere data: center.xyz, radius, materialType, padding, materialData.xyzw
    float sphereData[];
};

void main() {
    // Get sphere data for this primitive
    uint sphereIndex = gl_PrimitiveID;
    
    // Safety check to avoid out-of-bounds access
    if (sphereIndex >= params.sphereCount) {
        return; // Don't report intersection for invalid indices
    }
    
    uint dataOffset = sphereIndex * 10u; // 10 floats per sphere
    
    vec3 sphereCenter = vec3(sphereData[dataOffset], sphereData[dataOffset + 1u], sphereData[dataOffset + 2u]);
    float sphereRadius = sphereData[dataOffset + 3u];
    
    // Ray-sphere intersection
    vec3 oc = gl_WorldRayOriginEXT - sphereCenter;
    float a = dot(gl_WorldRayDirectionEXT, gl_WorldRayDirectionEXT);
    float b = 2.0 * dot(oc, gl_WorldRayDirectionEXT);
    float c = dot(oc, oc) - sphereRadius * sphereRadius;
    
    float discriminant = b * b - 4.0 * a * c;
    
    if (discriminant >= 0.0) {
        float sqrtDiscriminant = sqrt(discriminant);
        float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
        float t2 = (-b + sqrtDiscriminant) / (2.0 * a);
        
        // Find the closest valid intersection
        float t = -1.0;
        if (t1 > gl_RayTminEXT && t1 < gl_RayTmaxEXT) {
            t = t1;
        } else if (t2 > gl_RayTminEXT && t2 < gl_RayTmaxEXT) {
            t = t2;
        }
        
        if (t > 0.0) {
            // Calculate hit attributes (barycentric coordinates for sphere surface)
            vec3 hitPoint = gl_WorldRayOriginEXT + t * gl_WorldRayDirectionEXT;
            vec3 normal = normalize(hitPoint - sphereCenter);
            
            // Convert normal to UV coordinates for attributes
            attribs.x = 0.5 + atan(normal.z, normal.x) / (2.0 * 3.14159265);
            attribs.y = 0.5 - asin(normal.y) / 3.14159265;
            
            // Report intersection
            reportIntersectionEXT(t, 0u);
        }
    }
}