#!/usr/bin/env python3
"""Render a normalized OBJ with a fixed PartNet-style canonical camera."""

import argparse
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


def add_area_light(name, location, energy, size):
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    point_camera(light, (0.0, 0.0, 0.0))


def main():
    args = parse_args()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.obj(filepath=str(args.input))

    material = bpy.data.materials.new("neutral")
    material.diffuse_color = (0.72, 0.78, 0.82, 1.0)
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        obj.data.materials.clear()
        obj.data.materials.append(material)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (4.0, -6.0, 3.2)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.8
    point_camera(camera, (0.0, 0.0, 0.0))
    bpy.context.scene.camera = camera

    add_area_light("Key", (4.0, -4.0, 6.0), 650.0, 5.0)
    add_area_light("Fill", (-4.0, -2.0, 3.0), 350.0, 4.0)
    add_area_light("Rim", (0.0, 5.0, 5.0), 450.0, 4.0)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.2
    scene.world.color = (1.0, 1.0, 1.0)
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.display.shading.light = "STUDIO"
    scene.camera.data.lens = 50
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(args.output)
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
