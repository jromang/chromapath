#version 460
#extension GL_EXT_ray_tracing : require

struct RayPayload {
    vec3 color;
    uint depth;
    uint seed;
};

layout(location = 0) rayPayloadInEXT RayPayload payload;

void main() {
    // Sky gradient based on ray direction
    vec3 rayDir = normalize(gl_WorldRayDirectionEXT);
    float t = 0.5 * (rayDir.y + 1.0);
    vec3 skyColor = (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    payload.color = skyColor;
}