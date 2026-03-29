from __future__ import annotations

import csv
import struct
import zlib
from pathlib import Path

from .geometry import Vector2
from .models import StepTrace


def export_trace_csv(path: str | Path, trace: list[StepTrace]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step_index",
                "simulation_time_s",
                "virtual_x",
                "virtual_y",
                "virtual_heading_deg",
                "physical_x",
                "physical_y",
                "physical_heading_deg",
                "observed_virtual_x",
                "observed_virtual_y",
                "observed_virtual_heading_deg",
                "observed_physical_x",
                "observed_physical_y",
                "observed_physical_heading_deg",
                "observed_tracking_x",
                "observed_tracking_y",
                "observed_tracking_heading_deg",
                "in_reset",
                "translation_injection_x",
                "translation_injection_y",
                "rotation_injection_deg",
                "curvature_injection_deg",
                "translation_gain",
                "rotation_gain",
                "curvature_gain",
                "total_force_x",
                "total_force_y",
                "priority",
            ]
        )
        for row in trace:
            writer.writerow(
                [
                    row.step_index,
                    row.simulation_time_s,
                    row.virtual_x,
                    row.virtual_y,
                    row.virtual_heading_deg,
                    row.physical_x,
                    row.physical_y,
                    row.physical_heading_deg,
                    row.observed_virtual_x,
                    row.observed_virtual_y,
                    row.observed_virtual_heading_deg,
                    row.observed_physical_x,
                    row.observed_physical_y,
                    row.observed_physical_heading_deg,
                    row.observed_tracking_x,
                    row.observed_tracking_y,
                    row.observed_tracking_heading_deg,
                    int(row.in_reset),
                    row.translation_injection_x,
                    row.translation_injection_y,
                    row.rotation_injection_deg,
                    row.curvature_injection_deg,
                    row.translation_gain,
                    row.rotation_gain,
                    row.curvature_gain,
                    row.total_force_x,
                    row.total_force_y,
                    row.priority,
                ]
            )


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _blend_rgb(color: tuple[int, int, int], background: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
    return (
        _clamp_byte(weight * color[0] + (1.0 - weight) * background[0]),
        _clamp_byte(weight * color[1] + (1.0 - weight) * background[1]),
        _clamp_byte(weight * color[2] + (1.0 - weight) * background[2]),
    )


class RasterCanvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int]) -> None:
        self.width = width
        self.height = height
        self.pixels = bytearray(width * height * 3)
        self.fill(background)

    def fill(self, color: tuple[int, int, int]) -> None:
        row = bytes(color) * self.width
        self.pixels = bytearray(row * self.height)

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int], thickness: int = 1) -> None:
        radius = max(0, thickness // 2)
        for yy in range(y - radius, y + radius + 1):
            if yy < 0 or yy >= self.height:
                continue
            for xx in range(x - radius, x + radius + 1):
                if xx < 0 or xx >= self.width:
                    continue
                offset = (yy * self.width + xx) * 3
                self.pixels[offset : offset + 3] = bytes(color)

    def draw_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        color_start: tuple[int, int, int],
        color_end: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        steps = max(abs(dx), abs(dy), 1)
        color_end = color_end or color_start
        for step in range(steps + 1):
            t = step / steps
            x = int(round(x0 + dx * t))
            y = int(round(y0 + dy * t))
            color = (
                _clamp_byte(color_start[0] + (color_end[0] - color_start[0]) * t),
                _clamp_byte(color_start[1] + (color_end[1] - color_start[1]) * t),
                _clamp_byte(color_start[2] + (color_end[2] - color_start[2]) * t),
            )
            self.set_pixel(x, y, color, thickness)

    def draw_polygon(self, polygon: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 1) -> None:
        if len(polygon) < 2:
            return
        for index, point in enumerate(polygon):
            self.draw_line(point, polygon[(index + 1) % len(polygon)], color, thickness=thickness)

    def save_png(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for row in range(self.height):
            raw.append(0)
            offset = row * stride
            raw.extend(self.pixels[offset : offset + stride])
        compressed = zlib.compress(bytes(raw), level=9)

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        ihdr = struct.pack("!IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
        target.write_bytes(png)


def _world_to_pixel(point: Vector2, side_length: float, resolution: int) -> tuple[int, int]:
    x = (point.x + side_length / 2.0) / side_length * (resolution - 1)
    y = (point.y + side_length / 2.0) / side_length * (resolution - 1)
    return int(round(x)), int(round((resolution - 1) - y))


def export_real_path_graph_png(
    path: str | Path,
    *,
    tracking_space: list[Vector2],
    obstacles: list[list[Vector2]],
    user_real_paths: list[list[Vector2]],
    avatar_colors: list[tuple[int, int, int]] | None = None,
    resolution: int = 1024,
    side_length: float | None = None,
    border_thickness: int = 3,
    path_thickness: int = 2,
) -> None:
    if side_length is None:
        all_points = [*tracking_space, *(point for polygon in obstacles for point in polygon), *(point for path_points in user_real_paths for point in path_points)]
        max_extent = max((max(abs(point.x), abs(point.y)) for point in all_points), default=1.0)
        side_length = max(1.0, max_extent * 2.2)

    background = (255, 255, 255)
    tracking_color = (0, 0, 0)
    obstacle_color = (180, 80, 60)
    default_avatar_colors = [
        (220, 72, 67),
        (42, 112, 194),
        (37, 168, 76),
        (244, 160, 45),
    ]
    colors = avatar_colors or default_avatar_colors

    canvas = RasterCanvas(resolution, resolution, background)
    canvas.draw_polygon([_world_to_pixel(point, side_length, resolution) for point in tracking_space], tracking_color, thickness=border_thickness)
    for polygon in obstacles:
        canvas.draw_polygon([_world_to_pixel(point, side_length, resolution) for point in polygon], obstacle_color, thickness=border_thickness)

    for path_index, real_path in enumerate(user_real_paths):
        if len(real_path) < 2:
            continue
        color = colors[path_index % len(colors)]
        begin_weight = 0.1
        delta_weight = (1.0 - begin_weight) / max(len(real_path), 1)
        for segment_index in range(len(real_path) - 1):
            w0 = begin_weight + delta_weight * segment_index
            w1 = begin_weight + delta_weight * (segment_index + 1)
            color0 = _blend_rgb(color, background, w0)
            color1 = _blend_rgb(color, background, w1)
            canvas.draw_line(
                _world_to_pixel(real_path[segment_index], side_length, resolution),
                _world_to_pixel(real_path[segment_index + 1], side_length, resolution),
                color0,
                color1,
                thickness=path_thickness,
            )
    canvas.save_png(path)
