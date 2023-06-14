// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: spopov@google.com (Stefan Popov)

//Renders a scene with triangle geometry
#version 430

// The view projection matrix
uniform mat4 view_projection_matrix;

// Describes a material
struct Material
{
// The ambient coefficient (in .rgb). The .w component is ignored.
    vec4 ambient;

// The .rgb components contain the diffuse color. The .w component contains
// the diffuse texture index, or -1 if no texture is associated with this
// material.
    vec4 diffuse_and_texture;


// The .rgb components contain the specular coefficient, while .w contains
// the specular power.
    vec4 specular_shininess;
};

// Whether shading normals are supplied in the "normal" buffer. If false, the
// shading normal will be set to the geometric normal.
uniform bool has_normals = false;

// Whether texture coordinates are supplied in the "texcoord" buffer. If false,
// all texture coordinates will be zeros.
uniform bool has_texcoords = false;

// Whether vertex colors are supplied in the "vertex_colors" buffer.
uniform bool has_vertex_colors = false;

// Whether to cull back-facing triangles.
uniform bool cull_backfacing = true;

// The scene geometry, given as a sequence of consecutive triangles.
// Each triangle is specified with 9 consecutiveness floats -- the (x,y,z)
// positions of each of its 3 vertices.
layout(binding=0) buffer mesh { float mesh_buffer[]; };

// The shading normals, 9 floats per triangle -- x,y,z for each of the 3
// vertices. If has_normals is false, this buffer can be left unset.
layout(binding=1) buffer normals { float normal_buffer[]; };

// Vertex colors, 9 floats per triangle -- r,g,b for each of the 3
// vertices. If has_vertex_colors is false, this buffer can be left unset.
layout(binding=2) buffer vertex_colors { float vertex_colors_buffer[]; };

// The normalized texture coordinate, 6 floats per triangle -- u,v for each of
// the 3 vertices. If has_texcoords is false, this buffer can be left unset.
layout(binding=3) buffer texcoords { float texcoord_buffer[]; };

// The material ids, 1 integer per triangle. Each value in this buffer
// is an index into the "materials" buffer.
layout(binding=4) buffer material_ids { int material_id_buffer[]; };

// The materials used in the scene.
layout(std430, binding=5) buffer materials { Material material_buffer[]; };

// layout(binding=5) buffer debug { float debug_buffer []; };

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
out layout(location = 3) vec3 out_vertex_color;
out layout(location = 4) float depth;
out layout(location = 6) flat Material material;

layout(points) in;
layout(triangle_strip, max_vertices=12) out;

vec3 get_position(int i) {
    int o = gl_PrimitiveIDIn * 9 + i * 3;
    return vec3(mesh_buffer[o + 0], mesh_buffer[o + 1], mesh_buffer[o + 2]);
}

vec3 get_normal(int i) {
    int o = gl_PrimitiveIDIn * 9 + i * 3;
    return normalize(vec3(normal_buffer[o + 0], normal_buffer[o + 1], normal_buffer[o + 2]));
}

vec3 get_vertex_color(int i) {
    int o = gl_PrimitiveIDIn * 9 + i * 3;
    if(!has_vertex_colors) return vec3(1.0, 1.0, 1.0);
    return vec3(vertex_colors_buffer[o + 0], vertex_colors_buffer[o + 1], vertex_colors_buffer[o + 2]);
}

vec2 get_texcoord(int i) {
    int o = gl_PrimitiveIDIn * 6 + i * 2;
    return vec2(texcoord_buffer[o + 0], texcoord_buffer[o + 1]);
}

Material get_material(int i) {
    if (i < 0)
    {
        Material default_material;
        default_material.ambient = vec4(0.0);
        default_material.diffuse_and_texture = vec4(0.8, 0.2, 0.8, -1.0);
        default_material.specular_shininess = vec4(0.0);
        return default_material;
    }

    return material_buffer[i];
}

bool is_back_facing(vec3 v0, vec3 v1, vec3 v2) {
    vec4 tv0 = view_projection_matrix * vec4(v0, 1.0);
    vec4 tv1 = view_projection_matrix * vec4(v1, 1.0);
    vec4 tv2 = view_projection_matrix * vec4(v2, 1.0);
    tv0 /= tv0.w;
    tv1 /= tv1.w;
    tv2 /= tv2.w;
    vec2 a = (tv1.xy - tv0.xy);
    vec2 b = (tv2.xy - tv0.xy);
    return (a.x * b.y - b.x * a.y) <= 0;
}

void main() {
    vec3 v0 = get_position(0);
    vec3 v1 = get_position(1);
    vec3 v2 = get_position(2);

    if (cull_backfacing && is_back_facing(v0, v1, v2)) {
        return;
    }

    vec3 geometric_normal = normalize(
    cross(normalize(v1 - v0), normalize(v2 - v0)));

    material = material_buffer[material_id_buffer[gl_PrimitiveIDIn]];

    vec3 positions[3] = {v0, v1, v2};
    for (int i = 0; i < 3; i++) {
        gl_Position = view_projection_matrix * vec4(positions[i], 1);
        position = positions[i];
        normal = has_normals ? get_normal(i) : geometric_normal;
        texcoord = has_texcoords ? get_texcoord(i) : vec2(0.0);
        out_vertex_color = get_vertex_color(i);
        depth = gl_Position.w;

        EmitVertex();
    }
    EndPrimitive();
}
