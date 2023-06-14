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

// Fragment shader to compute point-light illumination
#version 420

// Light source position
uniform vec3 light_position;

// Camera position. Used to compute the lighting.
uniform vec3 camera_position;

// Light color
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);

// Ambient light color
uniform vec3 ambient_light_color = vec3(0.1, 0.1, 0.1);

// Diffuse textures, specified as an array.
uniform sampler2DArray textures;

struct Material
{
    vec4 ambient;
    vec4 diffuse_and_texture;
    vec4 specular_shininess;
};

in layout(location = 0) vec3 position;
in layout(location = 1) vec3 normal;
in layout(location = 2) vec2 texcoord;
in layout(location = 3) vec3 vertex_color;
in layout(location = 4) float depth;
in layout(location = 6) flat Material material;

out vec4 output_color;

void main() {
    vec3 ambient = material.ambient.xyz;
    vec3 diffuse = material.diffuse_and_texture.xyz;
    float diffuse_texture_layer = material.diffuse_and_texture.w;
    vec3 specular = material.specular_shininess.xyz;
    float shininess = material.specular_shininess.w;

    diffuse *= vertex_color;
    if (diffuse_texture_layer >= 0.0) {
        diffuse *= texture(
            textures, vec3(texcoord.xy, diffuse_texture_layer)).xyz;
    }

    vec3 N = normalize(normal);
    vec3 L = normalize(light_position - position);
    vec3 V = -normalize(camera_position - position);
    vec3 R = reflect(L, N);
    float dotNL = dot(L, N);
    float dotRV = dot(R, V);
    vec3 A = ambient;
    vec3 D = diffuse * abs(dotNL) * light_color + ambient_light_color * diffuse;
    vec3 S = specular * light_color * pow(max(dotRV, 0), shininess);

    output_color = vec4(A + D + S, depth);
}
