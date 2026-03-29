from __future__ import annotations

import json
import os
import threading
import time
import webbrowser
from errno import EADDRINUSE
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .exporters import export_trace_csv
from .factory import (
    PATH_OPTIONS,
    REDIRECTOR_OPTIONS,
    RESETTER_OPTIONS,
    TRACKING_SPACE_OPTIONS,
    SimulationConfig,
    build_scheduler,
)
from .geometry import Vector2, heading_to_vector, vector_to_heading
from .experiments import run_command_file as run_experiment_command_file


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Python OpenRDW</title>
  <style>
    :root { --bg:#f3efe6; --panel:#fffaf2; --line:#bda98f; --ink:#1f1a16; --accent:#0f5f9a; --accent2:#b04a2d; --virt:#d94841; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:ui-sans-serif, system-ui, sans-serif; background:linear-gradient(135deg,#f3efe6,#e6f0fb); color:var(--ink); }
    .layout { display:grid; grid-template-columns:320px 1fr; min-height:100vh; }
    .controls { padding:18px; background:var(--panel); border-right:1px solid var(--line); overflow:auto; }
    .viewer { padding:18px; }
    h1 { margin:0 0 8px; font-size:22px; }
    .sub { margin:0 0 16px; color:#5d5347; font-size:13px; line-height:1.4; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px 12px; }
    label { display:block; font-size:12px; font-weight:700; margin-bottom:4px; text-transform:uppercase; letter-spacing:.04em; }
    input, select, button { width:100%; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:white; }
    button { background:var(--accent); color:white; border:none; font-weight:700; cursor:pointer; }
    button.secondary { background:#5e6b75; }
    button.warn { background:var(--accent2); }
    .button-row { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
    .status { margin-top:14px; padding:12px; border:1px solid var(--line); border-radius:12px; background:#f8f2e8; min-height:54px; }
    .card { position:relative; background:rgba(255,255,255,.72); border:1px solid rgba(120,100,80,.25); border-radius:18px; padding:14px; backdrop-filter:blur(6px); }
    .title { font-weight:800; margin-bottom:8px; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:#4f463d; margin-bottom:10px; }
    .telemetry { margin-top:12px; display:grid; gap:10px; }
    .telemetry-card { border:1px solid rgba(120,100,80,.25); border-radius:12px; background:rgba(255,255,255,.85); padding:10px 12px; }
    .telemetry-title { font-weight:800; margin-bottom:6px; }
    .telemetry-line { font-size:13px; color:#463d35; line-height:1.45; }
    .legend span::before { content:""; display:inline-block; width:12px; height:12px; border-radius:999px; margin-right:6px; vertical-align:-1px; }
    .legend .virt::before { background:var(--virt); }
    .legend .phys::before { background:#1768ac; }
    .legend .buffer::before { background:#f0b429; }
    .legend .target::before { background:#1f9d55; }
    .legend .trail::before { background:#6c63ff; }
    .toggle { margin-top:12px; padding:10px 12px; border:1px solid var(--line); border-radius:10px; background:white; display:flex; gap:10px; align-items:center; }
    .toggle input { width:auto; margin:0; }
    .section-title { margin:14px 0 8px; font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.05em; color:#5d5347; }
    .obstacle-actions { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:8px; }
    .obstacle-list { display:grid; gap:8px; margin-top:8px; }
    .obstacle-row { border:1px solid var(--line); border-radius:12px; background:white; padding:10px; display:grid; grid-template-columns:1fr 1fr; gap:8px; }
    .obstacle-row .row-title { grid-column:1 / -1; font-size:12px; font-weight:800; color:#5d5347; }
    .reset-banner { position:absolute; top:50px; right:14px; padding:8px 12px; border-radius:12px; background:#b04a2d; color:white; font-weight:800; display:none; width:max-content; z-index:2; pointer-events:none; }
    .completion-panel { margin-top:14px; padding:12px; border:1px solid #d6b88a; border-radius:12px; background:#fff3df; display:none; }
    .completion-panel.active { display:block; }
    .completion-panel .panel-title { font-weight:900; margin-bottom:6px; }
    .completion-panel .panel-line { font-size:13px; color:#5a4732; line-height:1.45; }
    canvas { width:100%; aspect-ratio:1/1; background:white; border:1px solid var(--line); border-radius:12px; }
  </style>
</head>
<body>
  <div class="layout">
    <div class="controls">
      <h1>Python OpenRDW</h1>
      <p class="sub">Single-frame overlay view. The virtual space stays fixed. The physical tracking space is transformed relative to it so you can see how it slides and rotates against the virtual world.</p>
      <div class="grid">
        <div><label>Redirector</label><select id="redirector"></select></div>
        <div><label>Resetter</label><select id="resetter"></select></div>
        <div><label>Controller</label><select id="movement_controller"></select></div>
        <div><label>Path</label><select id="path_mode"></select></div>
        <div><label>Tracking Space</label><select id="tracking_space_shape"></select></div>
        <div><label>Agents</label><input id="agent_count" type="number" min="1" max="4" value="1"></div>
        <div><label>Physical Width</label><input id="physical_width" type="number" step="0.5" value="5"></div>
        <div><label>Physical Height</label><input id="physical_height" type="number" step="0.5" value="5"></div>
        <div><label>Virtual Width</label><input id="virtual_width" type="number" step="0.5" value="20"></div>
        <div><label>Virtual Height</label><input id="virtual_height" type="number" step="0.5" value="20"></div>
        <div><label>Physical Obstacles</label><input id="physical_obstacle_count" type="number" min="0" max="24" value="0" onchange="syncPhysicalObstacleCount()"></div>
        <div><label>Virtual Obstacles</label><input id="virtual_obstacle_count" type="number" min="0" max="12" value="0"></div>
        <div><label>Physical Space Buffer</label><input id="physical_space_buffer" type="number" step="0.1" min="0" value="0.4"></div>
        <div><label>Obstacle Buffer</label><input id="obstacle_buffer" type="number" step="0.1" min="0" value="0.4"></div>
        <div><label>Body Diameter</label><input id="body_collider_diameter" type="number" step="0.05" min="0" value="0.1"></div>
        <div><label>Path Length</label><input id="total_path_length" type="number" step="1" value="30"></div>
        <div><label>Sampling Hz</label><input id="sampling_frequency" type="number" step="1" min="1" value="10"></div>
        <div><label>dt</label><input id="time_step" type="number" step="0.01" value="0.0333"></div>
        <div><label>Move Speed</label><input id="translation_speed" type="number" step="0.1" value="1.2"></div>
        <div><label>Rot Speed</label><input id="rotation_speed" type="number" step="1" value="100"></div>
        <div><label>Trail Time</label><input id="trail_visual_time" type="number" step="1" value="-1"></div>
        <div><label>Seed</label><input id="seed" type="number" step="1" value="3041"></div>
        <div><label>Tracking File</label><input id="tracking_space_file" type="text" value=""></div>
        <div><label>Waypoints File</label><input id="waypoints_file" type="text" value=""></div>
        <div><label>Sampling File</label><input id="sampling_intervals_file" type="text" value=""></div>
      </div>
      <label class="toggle"><input id="use_custom_sampling_frequency" type="checkbox" onchange="refreshState()">Use custom position sampling frequency</label>
      <label class="toggle"><input id="draw_real_trail" type="checkbox" checked onchange="refreshState()">Draw physical trail</label>
      <label class="toggle"><input id="draw_virtual_trail" type="checkbox" checked onchange="refreshState()">Draw virtual trail</label>
      <label class="toggle"><input id="virtual_world_visible" type="checkbox" checked onchange="refreshState()">Show virtual world</label>
      <label class="toggle"><input id="tracking_space_visible" type="checkbox" checked onchange="refreshState()">Show tracking space</label>
      <div class="section-title">Command File Runner</div>
      <div class="grid">
        <div><label>Command File / Dir</label><input id="command_file" type="text" value=""></div>
        <div><label>Output Dir</label><input id="command_output_dir" type="text" value="/Users/yph/Desktop/My_project/python_openrdw/out/command_runs"></div>
      </div>
      <div class="button-row" style="margin-top:10px;">
        <button class="secondary" onclick="runCommandFile()">Run Command File</button>
        <button class="secondary" onclick="refreshState()">Refresh Status</button>
      </div>
      <div class="section-title">Physical Obstacle Editor</div>
      <div class="obstacle-actions">
        <button type="button" onclick="addPhysicalObstacle()">Add Obstacle</button>
        <button type="button" class="secondary" onclick="clearPhysicalObstacles()">Clear</button>
      </div>
      <div class="obstacle-list" id="physical_obstacle_list"></div>
      <div class="button-row">
        <button onclick="buildScene()">Build Scene</button>
        <button class="secondary" onclick="stepOnce()">Step</button>
      </div>
      <div class="button-row">
        <button onclick="startRun()">Start</button>
        <button class="secondary" onclick="pauseRun()">Pause</button>
      </div>
      <div class="button-row">
        <button class="warn" onclick="resetScene()">Reset</button>
        <button class="secondary" onclick="exportCsv()">Export CSV</button>
      </div>
      <label class="toggle"><input id="show_buffer" type="checkbox" checked onchange="refreshState()">Show physical buffer and reset boundary</label>
      <div class="status" id="status">Ready</div>
      <div class="completion-panel" id="completion_panel">
        <div class="panel-title" id="completion_title">Experiment Complete</div>
        <div class="panel-line" id="completion_line_1"></div>
        <div class="panel-line" id="completion_line_2"></div>
        <div class="panel-line" id="completion_line_3"></div>
      </div>
    </div>
    <div class="viewer">
      <div class="card">
        <div class="title">Combined View</div>
        <div class="reset-banner" id="reset_banner">RESET</div>
        <div class="legend">
          <span class="virt">Virtual space (fixed)</span>
          <span class="phys">Physical tracking space and user heading</span>
          <span class="buffer">Physical buffer / reset trigger boundary</span>
          <span class="target">Current waypoint ball</span>
          <span class="trail">Virtual / physical trail history</span>
        </div>
        <canvas id="overlay" width="900" height="900"></canvas>
        <div class="telemetry" id="telemetry"></div>
      </div>
    </div>
  </div>
<script>
const redirectorOptions = __REDIRECTORS__;
const resetterOptions = __RESETTERS__;
const pathOptions = __PATHS__;
const trackingOptions = __TRACKING_SPACES__;
const movementOptions = ['autopilot', 'keyboard'];
const colors = ['#d94841','#2f7d32','#1768ac','#c17900'];
const TRAIL_MIN_DIST = 0.1;
let timer = null;
let running = false;
let stepInFlight = false;
let commandPollTimer = null;
const pressedKeys = { w:false, a:false, s:false, d:false, left:false, right:false };
let physicalObstacleSpecs = [];

function fillSelect(id, values, selected) {
  const el = document.getElementById(id);
  for (const value of values) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = value;
    if (value === selected) opt.selected = true;
    el.appendChild(opt);
  }
}
fillSelect('redirector', redirectorOptions, 's2c');
fillSelect('resetter', resetterOptions, 'two_one_turn');
fillSelect('movement_controller', movementOptions, 'autopilot');
fillSelect('path_mode', pathOptions, 'random_turn');
fillSelect('tracking_space_shape', trackingOptions, 'rectangle');

function defaultPhysicalObstacleSpec() {
  return { shape: 'square', x: 0, y: 0, size: 0.8, width: 1.0, height: 1.0, radius: 0.5 };
}

function collectPhysicalObstacleSpecs() {
  const rows = [...document.querySelectorAll('.obstacle-row')];
  return rows.map((row) => ({
    shape: row.querySelector('.shape').value,
    x: Number(row.querySelector('.x').value),
    y: Number(row.querySelector('.y').value),
    size: Number(row.querySelector('.size').value),
    width: Number(row.querySelector('.width').value),
    height: Number(row.querySelector('.height').value),
    radius: Number(row.querySelector('.radius').value)
  }));
}

function renderPhysicalObstacleEditor() {
  const root = document.getElementById('physical_obstacle_list');
  root.innerHTML = '';
  physicalObstacleSpecs.forEach((spec, index) => {
    const row = document.createElement('div');
    row.className = 'obstacle-row';
    row.innerHTML = `
      <div class="row-title">Obstacle ${index + 1}</div>
      <div><label>Shape</label><select class="shape">
        <option value="square" ${spec.shape === 'square' ? 'selected' : ''}>square</option>
        <option value="rectangle" ${spec.shape === 'rectangle' ? 'selected' : ''}>rectangle</option>
        <option value="triangle" ${spec.shape === 'triangle' ? 'selected' : ''}>triangle</option>
        <option value="circle" ${spec.shape === 'circle' ? 'selected' : ''}>circle</option>
      </select></div>
      <div><label>Position X</label><input class="x" type="number" step="0.1" value="${spec.x}"></div>
      <div><label>Position Y</label><input class="y" type="number" step="0.1" value="${spec.y}"></div>
      <div><label>Square Size</label><input class="size" type="number" step="0.1" min="0.1" value="${spec.size}"></div>
      <div><label>Rect Width</label><input class="width" type="number" step="0.1" min="0.1" value="${spec.width}"></div>
      <div><label>Rect Height</label><input class="height" type="number" step="0.1" min="0.1" value="${spec.height}"></div>
      <div><label>Circle Radius</label><input class="radius" type="number" step="0.1" min="0.1" value="${spec.radius}"></div>
      <div style="grid-column:1 / -1;"><button type="button" class="warn" onclick="removePhysicalObstacle(${index})">Remove</button></div>
    `;
    root.appendChild(row);
  });
  document.getElementById('physical_obstacle_count').value = physicalObstacleSpecs.length;
}

function syncPhysicalObstacleCount() {
  physicalObstacleSpecs = collectPhysicalObstacleSpecs();
  const targetCount = Number(document.getElementById('physical_obstacle_count').value);
  while (physicalObstacleSpecs.length < targetCount) physicalObstacleSpecs.push(defaultPhysicalObstacleSpec());
  while (physicalObstacleSpecs.length > targetCount) physicalObstacleSpecs.pop();
  renderPhysicalObstacleEditor();
}

function addPhysicalObstacle() {
  physicalObstacleSpecs = collectPhysicalObstacleSpecs();
  physicalObstacleSpecs.push(defaultPhysicalObstacleSpec());
  renderPhysicalObstacleEditor();
}

function removePhysicalObstacle(index) {
  physicalObstacleSpecs = collectPhysicalObstacleSpecs();
  physicalObstacleSpecs.splice(index, 1);
  renderPhysicalObstacleEditor();
}

function clearPhysicalObstacles() {
  physicalObstacleSpecs = [];
  renderPhysicalObstacleEditor();
}

function setStatus(text) {
  document.getElementById('status').textContent = text;
}

function currentConfig() {
  return {
    redirector: document.getElementById('redirector').value,
    resetter: document.getElementById('resetter').value,
    movement_controller: document.getElementById('movement_controller').value,
    path_mode: document.getElementById('path_mode').value,
    tracking_space_shape: document.getElementById('tracking_space_shape').value,
    physical_width: Number(document.getElementById('physical_width').value),
    physical_height: Number(document.getElementById('physical_height').value),
    virtual_width: Number(document.getElementById('virtual_width').value),
    virtual_height: Number(document.getElementById('virtual_height').value),
    physical_obstacle_count: Number(document.getElementById('physical_obstacle_count').value),
    physical_obstacle_specs: collectPhysicalObstacleSpecs(),
    virtual_obstacle_count: Number(document.getElementById('virtual_obstacle_count').value),
    physical_space_buffer: Number(document.getElementById('physical_space_buffer').value),
    obstacle_buffer: Number(document.getElementById('obstacle_buffer').value),
    body_collider_diameter: Number(document.getElementById('body_collider_diameter').value),
    agent_count: Number(document.getElementById('agent_count').value),
    total_path_length: Number(document.getElementById('total_path_length').value),
    sampling_frequency: Number(document.getElementById('sampling_frequency').value),
    use_custom_sampling_frequency: document.getElementById('use_custom_sampling_frequency').checked,
    time_step: Number(document.getElementById('time_step').value),
    translation_speed: Number(document.getElementById('translation_speed').value),
    rotation_speed: Number(document.getElementById('rotation_speed').value),
    draw_real_trail: document.getElementById('draw_real_trail').checked,
    draw_virtual_trail: document.getElementById('draw_virtual_trail').checked,
    trail_visual_time: Number(document.getElementById('trail_visual_time').value),
    virtual_world_visible: document.getElementById('virtual_world_visible').checked,
    tracking_space_visible: document.getElementById('tracking_space_visible').checked,
    buffer_visible: document.getElementById('show_buffer').checked,
    seed: Number(document.getElementById('seed').value),
    tracking_space_file: document.getElementById('tracking_space_file').value || null,
    waypoints_file: document.getElementById('waypoints_file').value || null,
    sampling_intervals_file: document.getElementById('sampling_intervals_file').value || null
  };
}

function currentCommandFilePayload() {
  return {
    command_file: document.getElementById('command_file').value || null,
    output_dir: document.getElementById('command_output_dir').value || null,
    sampling_frequency: Number(document.getElementById('sampling_frequency').value)
  };
}

function showBuffer() {
  return document.getElementById('show_buffer').checked && (window.__lastStateConfig ? window.__lastStateConfig.buffer_visible : true);
}

function trackingBoundaryInset(state) {
  const bodyDiameter = Number(state.gains.body_collider_diameter || 0);
  const buffer = Number(state.gains.physical_space_buffer || 0);
  return buffer + bodyDiameter / 2;
}

async function api(path, payload) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });
  return await response.json();
}

function rotatePoint(point, degrees) {
  const r = -degrees * Math.PI / 180;
  const cos = Math.cos(r);
  const sin = Math.sin(r);
  return { x: point.x * cos - point.y * sin, y: point.x * sin + point.y * cos };
}

function transformPhysicalPoint(point, trackingPose) {
  const rotated = rotatePoint(point, trackingPose.heading_deg);
  return { x: trackingPose.position.x + rotated.x, y: trackingPose.position.y + rotated.y };
}

function transformPhysicalDirection(vector, trackingPose) {
  return rotatePoint(vector, trackingPose.heading_deg);
}

function renderTrackingPose(agentState) {
  return agentState.observed_tracking_space_pose || agentState.tracking_space_pose;
}

function renderPhysicalPose(agentState) {
  return agentState.observed_physical_pose || {
    position: agentState.physical_position,
    heading_deg: agentState.physical_heading_deg,
  };
}

function headingFromVector(vector) {
  if (!vector || Math.hypot(vector.x, vector.y) < 1e-6) return 0;
  return Math.atan2(vector.x, vector.y) * 180 / Math.PI;
}

function polygonSignedArea(polygon) {
  if (!polygon || polygon.length < 3) return 0;
  let area = 0;
  for (let i = 0; i < polygon.length; i++) {
    const p = polygon[i];
    const q = polygon[(i + 1) % polygon.length];
    area += p.x * q.y - q.x * p.y;
  }
  return area / 2;
}

function lineIntersection(p1, p2, p3, p4) {
  const x1 = p1.x, y1 = p1.y, x2 = p2.x, y2 = p2.y;
  const x3 = p3.x, y3 = p3.y, x4 = p4.x, y4 = p4.y;
  const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(denom) < 1e-8) return null;
  const px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
  const py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
  return { x: px, y: py };
}

function offsetPolygon(polygon, buffer, inward) {
  if (!polygon || polygon.length < 3 || buffer <= 1e-8) return polygon || [];
  const area = polygonSignedArea(polygon);
  if (Math.abs(area) < 1e-8) return polygon;
  const ccw = area > 0;
  const direction = inward ? 1 : -1;
  const offsetSign = ccw ? direction : -direction;
  const shiftedLines = [];

  for (let i = 0; i < polygon.length; i++) {
    const p = polygon[i];
    const q = polygon[(i + 1) % polygon.length];
    const edge = { x: q.x - p.x, y: q.y - p.y };
    const length = Math.hypot(edge.x, edge.y);
    if (length < 1e-8) continue;
    const inwardNormal = ccw
      ? { x: -edge.y / length, y: edge.x / length }
      : { x: edge.y / length, y: -edge.x / length };
    const shift = {
      x: inwardNormal.x * buffer * offsetSign,
      y: inwardNormal.y * buffer * offsetSign,
    };
    shiftedLines.push([
      { x: p.x + shift.x, y: p.y + shift.y },
      { x: q.x + shift.x, y: q.y + shift.y },
    ]);
  }
  if (shiftedLines.length < 3) return polygon;

  const buffered = [];
  for (let i = 0; i < shiftedLines.length; i++) {
    const prev = shiftedLines[(i - 1 + shiftedLines.length) % shiftedLines.length];
    const curr = shiftedLines[i];
    const point = lineIntersection(prev[0], prev[1], curr[0], curr[1]);
    if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      return polygon;
    }
    buffered.push(point);
  }
  return buffered;
}

function buildStableBounds(state) {
  const allPoints = [];
  const pushPoint = (point) => {
    if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) allPoints.push(point);
  };
  const pushPoints = (points) => {
    for (const point of points || []) pushPoint(point);
  };
  const pushPolygons = (polygons) => {
    for (const polygon of polygons || []) pushPoints(polygon);
  };

  pushPoints(state.virtual.boundary);
  pushPolygons(state.virtual.obstacles);
  pushPoints(state.virtual.targets);
  for (const trace of state.virtual.traces || []) pushPoints(clipTrail(trace, state));

  (state.agent_states || []).forEach((agentState, idx) => {
    const trackingPose = renderTrackingPose(agentState);
    if (!trackingPose) return;
    pushPoints((state.physical.boundary || []).map(p => transformPhysicalPoint(p, trackingPose)));
    if (showBuffer()) {
      pushPoints(
        offsetPolygon(state.physical.boundary || [], trackingBoundaryInset(state), true)
          .map(p => transformPhysicalPoint(p, trackingPose))
      );
    }
    for (const obstacle of state.physical.obstacles || []) {
      pushPoints(obstacle.map(p => transformPhysicalPoint(p, trackingPose)));
      if (showBuffer()) {
        pushPoints(
          offsetPolygon(obstacle, state.gains.obstacle_buffer || 0, false)
            .map(p => transformPhysicalPoint(p, trackingPose))
        );
      }
    }
    const trace = state.physical.traces && state.physical.traces[idx] ? state.physical.traces[idx] : [];
    pushPoints(clipTrail(trace, state).map(p => transformPhysicalPoint(p, trackingPose)));
  });

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of allPoints) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  if (!Number.isFinite(minX)) {
    minX = -10; maxX = 10; minY = -10; maxY = 10;
  }
  const pad = Math.max(2.0, Math.max(state.gains.physical_space_buffer || 0, state.gains.obstacle_buffer || 0) + 1.0);
  return {
    minX: minX - pad,
    maxX: maxX + pad,
    minY: minY - pad,
    maxY: maxY + pad,
  };
}

function buildMapper(state, canvas) {
  const bounds = buildStableBounds(state);
  const minX = bounds.minX;
  const maxX = bounds.maxX;
  const minY = bounds.minY;
  const maxY = bounds.maxY;
  const width = Math.max(maxX - minX, 1e-6);
  const height = Math.max(maxY - minY, 1e-6);
  const margin = 34;
  return function(point) {
    const x = margin + ((point.x - minX) / width) * (canvas.width - 2 * margin);
    const y = canvas.height - (margin + ((point.y - minY) / height) * (canvas.height - 2 * margin));
    return [x, y];
  };
}

function drawPolygon(ctx, polygon, mapper, fill, stroke, lineWidth = 2, dash = []) {
  if (!polygon || polygon.length === 0) return;
  ctx.save();
  ctx.setLineDash(dash);
  ctx.beginPath();
  polygon.forEach((point, idx) => {
    const [x, y] = mapper(point);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
  ctx.restore();
}

function drawAgent(ctx, mapper, agent, color) {
  const [x, y] = mapper(agent.position);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.fill();
  const dx = 14 * Math.sin(agent.heading_deg * Math.PI / 180);
  const dy = -14 * Math.cos(agent.heading_deg * Math.PI / 180);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + dx, y + dy);
  ctx.stroke();
}

function drawTargetBall(ctx, mapper, point) {
  if (!point) return;
  const [x, y] = mapper(point);
  ctx.fillStyle = '#1f9d55';
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#0d5c31';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 10, 0, Math.PI * 2);
  ctx.stroke();
}

function drawPolyline(ctx, points, mapper, stroke, lineWidth = 2, dash = []) {
  if (!points || points.length < 2) return;
  ctx.save();
  ctx.setLineDash(dash);
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.beginPath();
  points.forEach((point, idx) => {
    const [x, y] = mapper(point);
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.restore();
}

function clipTrail(points, state) {
  if (!points || points.length === 0) return [];
  const trailTime = Number(state.config.trail_visual_time ?? -1);
  if (trailTime < 0 || points[0].t === undefined) return points;
  const latest = points[points.length - 1].t;
  return points.filter((point) => latest - point.t <= trailTime + 1e-6);
}

function formatPoint(point) {
  if (!point) return 'none';
  return `(${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
}

function renderTelemetry(state) {
  const root = document.getElementById('telemetry');
  root.innerHTML = '';
  const agentStates = state.agent_states || [];
  agentStates.forEach((agent, idx) => {
    const card = document.createElement('div');
    card.className = 'telemetry-card';
    card.style.borderLeft = `6px solid ${colors[idx % colors.length]}`;
    card.innerHTML = `
      <div class="telemetry-title">Agent ${agent.agent_id}</div>
      <div class="telemetry-line">Virtual Pos: ${formatPoint(agent.virtual_position)}</div>
      <div class="telemetry-line">Physical Pos: ${formatPoint(agent.physical_position)}</div>
      <div class="telemetry-line">Virtual Heading: ${agent.virtual_heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Physical Heading: ${agent.physical_heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Overlay Heading: ${agent.display_heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Root Pos: ${formatPoint(agent.root_pose.position)}</div>
      <div class="telemetry-line">Root Heading: ${agent.root_pose.heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Tracking Space Pos: ${formatPoint(agent.tracking_space_pose.position)}</div>
      <div class="telemetry-line">Tracking Space Heading: ${agent.tracking_space_pose.heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Tracking Space Local Pos: ${formatPoint(agent.tracking_space_local_position)}</div>
      <div class="telemetry-line">Tracking Space Local Heading: ${agent.tracking_space_local_heading_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Waypoint: #${agent.current_waypoint} @ ${formatPoint(agent.active_waypoint)}</div>
      <div class="telemetry-line">Gains: g_t=${agent.translation_gain.toFixed(3)} g_r=${agent.rotation_gain.toFixed(3)} g_c=${agent.curvature_gain.toFixed(3)}</div>
      <div class="telemetry-line">Delta Virtual: ${formatPoint(agent.delta_virtual_translation)} / ${agent.delta_virtual_rotation_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Delta Physical: ${formatPoint(agent.delta_physical_translation)} / ${agent.delta_physical_rotation_deg.toFixed(2)} deg</div>
      <div class="telemetry-line">Same Pos Time: ${agent.same_pos_time.toFixed(2)} s</div>
      <div class="telemetry-line">In Reset: ${agent.in_reset ? 'yes' : 'no'}</div>
      <div class="telemetry-line">Mission Complete: ${agent.mission_complete ? 'yes' : 'no'}</div>
    `;
    root.appendChild(card);
  });
}

function drawOverlay(state) {
  const canvas = document.getElementById('overlay');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const mapPoint = buildMapper(state, canvas);

  if (state.config.virtual_world_visible) {
    drawPolygon(ctx, state.virtual.boundary, mapPoint, 'rgba(217,72,65,0.05)', '#d94841', 2, []);
    for (const obstacle of state.virtual.obstacles) {
      drawPolygon(ctx, obstacle, mapPoint, 'rgba(217,72,65,0.10)', '#d94841', 1.5, []);
    }
  }

  state.physical.agents.forEach((physicalAgent, idx) => {
    const trackingPose = renderTrackingPose(state.agent_states[idx]);
    if (state.config.tracking_space_visible) {
      const transformedPhysicalBoundary = state.physical.boundary.map(p => transformPhysicalPoint(p, trackingPose));
      drawPolygon(ctx, transformedPhysicalBoundary, mapPoint, 'rgba(23,104,172,0.04)', '#1768ac', 2, [8, 6]);
      if (showBuffer()) {
        const bufferBoundary = offsetPolygon(state.physical.boundary, trackingBoundaryInset(state), true)
          .map(p => transformPhysicalPoint(p, trackingPose));
        drawPolygon(ctx, bufferBoundary, mapPoint, 'rgba(240,180,41,0.10)', '#f0b429', 2, [4, 4]);
      }
    }
    if (state.config.tracking_space_visible) {
      for (const obstacle of state.physical.obstacles) {
        const transformedObstacle = obstacle.map(p => transformPhysicalPoint(p, trackingPose));
        drawPolygon(ctx, transformedObstacle, mapPoint, 'rgba(23,104,172,0.10)', '#1768ac', 1.5, []);
        if (showBuffer()) {
          const obstacleBuffer = offsetPolygon(obstacle, state.gains.obstacle_buffer, false)
            .map(p => transformPhysicalPoint(p, trackingPose));
          drawPolygon(ctx, obstacleBuffer, mapPoint, 'rgba(240,180,41,0.14)', '#f0b429', 1.5, [4, 4]);
        }
      }
    }
  });
  if (state.config.draw_virtual_trail && state.virtual.traces) {
    state.virtual.traces.forEach((trace) => drawPolyline(ctx, clipTrail(trace, state), mapPoint, '#d94841', 2, []));
  }
  if (state.config.draw_real_trail && state.physical.traces) {
    state.physical.traces.forEach((trace, idx) => {
      const trackingPose = renderTrackingPose(state.agent_states[idx]);
      const transformedTrace = clipTrail(trace, state).map(p => transformPhysicalPoint(p, trackingPose));
      drawPolyline(ctx, transformedTrace, mapPoint, '#1768ac', 2, []);
    });
  }
  state.virtual.agents.forEach((agent, idx) => {
    const physicalAgentState = state.agent_states && state.agent_states[idx] ? state.agent_states[idx] : null;
    const physicalPose = physicalAgentState ? renderPhysicalPose(physicalAgentState) : null;
    const trackingPose = physicalAgentState ? renderTrackingPose(physicalAgentState) : null;
    drawAgent(
      ctx,
      mapPoint,
      {
        position: physicalAgentState ? transformPhysicalPoint(physicalPose.position, trackingPose) : agent.position,
        heading_deg: physicalAgentState ? physicalAgentState.display_heading_deg : agent.heading_deg,
      },
      '#1768ac'
    );
  });
  if (state.config.virtual_world_visible && state.virtual.targets) {
    state.virtual.targets.forEach((target) => drawTargetBall(ctx, mapPoint, target));
  }
}

function renderResetBanner(state) {
  const banner = document.getElementById('reset_banner');
  const inReset = (state.agent_states || []).some(agent => agent.in_reset);
  banner.style.display = inReset ? 'block' : 'none';
}

function render(state) {
  window.__lastRenderedState = state;
  window.__lastStateConfig = state.config;
  drawOverlay(state);
  renderResetBanner(state);
  renderTelemetry(state);
  renderCompletionPanel(state);
  if (state.all_mission_complete) {
    setStatus(`step=${state.step_index} mission complete`);
  } else {
    setStatus(`step=${state.step_index} agents=${state.config.agent_count} redirector=${state.config.redirector} resetter=${state.config.resetter}`);
  }
}

function renderCompletionPanel(state) {
  const panel = document.getElementById('completion_panel');
  const title = document.getElementById('completion_title');
  const line1 = document.getElementById('completion_line_1');
  const line2 = document.getElementById('completion_line_2');
  const line3 = document.getElementById('completion_line_3');
  const job = state.command_job || {};
  if (state.all_mission_complete && !['queued', 'running', 'completed', 'error'].includes(job.status)) {
    panel.classList.add('active');
    title.textContent = 'Mission Complete';
    line1.textContent = `Redirector: ${state.config.redirector}`;
    line2.textContent = `Resetter: ${state.config.resetter}`;
    line3.textContent = `Steps: ${state.step_index}`;
    return;
  }
  const completed = job.status === 'completed';
  const error = job.status === 'error';
  panel.classList.toggle('active', completed || error);
  if (completed) {
    title.textContent = 'Experiment Complete';
    line1.textContent = `Command file: ${job.command_file || 'n/a'}`;
    line2.textContent = `Output dir: ${job.output_dir || 'n/a'}`;
    line3.textContent = `Trials: ${job.trial_count ?? 'n/a'} Status: ${job.message || 'completed'}`;
  } else if (error) {
    title.textContent = 'Experiment Failed';
    line1.textContent = `Command file: ${job.command_file || 'n/a'}`;
    line2.textContent = `Output dir: ${job.output_dir || 'n/a'}`;
    line3.textContent = job.error || 'Unknown error';
  } else if (job.status === 'running' || job.status === 'queued') {
    title.textContent = 'Experiment Running';
    line1.textContent = `Command file: ${job.command_file || 'n/a'}`;
    line2.textContent = `Output dir: ${job.output_dir || 'n/a'}`;
    line3.textContent = job.message || 'Running...';
  } else {
    panel.classList.remove('active');
  }
}

async function refreshState() {
  const response = await fetch('/api/state');
  const data = await response.json();
  render(data);
}

async function buildScene() {
  pauseRun();
  physicalObstacleSpecs = collectPhysicalObstacleSpecs();
  const data = await api('/api/build', currentConfig());
  render(data);
}

async function stepOnce() {
  if (stepInFlight) return;
  stepInFlight = true;
  const payload = {};
  if (document.getElementById('movement_controller').value === 'keyboard') {
    payload.manual_inputs = { '0': pressedKeys };
  }
  try {
    const data = await api('/api/step', payload);
    render(data);
    if (data.all_mission_complete) {
      pauseRun();
      setStatus(`step=${data.step_index} mission complete`);
    }
  } finally {
    stepInFlight = false;
  }
}

async function runLoop() {
  if (!running) return;
  await stepOnce();
  if (!running) return;
  timer = setTimeout(runLoop, 33);
}

function startRun() {
  if (running) return;
  if (window.__lastRenderedState && window.__lastRenderedState.all_mission_complete) {
    setStatus(`step=${window.__lastRenderedState.step_index} mission complete`);
    return;
  }
  running = true;
  setStatus('Running');
  runLoop();
}

function pauseRun() {
  running = false;
  if (timer) {
    clearTimeout(timer);
    timer = null;
  }
}

async function resetScene() {
  await buildScene();
  setStatus('Reset');
}

async function exportCsv() {
  const data = await api('/api/export', {});
  setStatus(data.message);
}

async function runCommandFile() {
  pauseRun();
  const data = await api('/api/run_command_file', currentCommandFilePayload());
  render(data);
  if (commandPollTimer) clearInterval(commandPollTimer);
  commandPollTimer = setInterval(async () => {
    const refresh = await fetch('/api/state');
    const state = await refresh.json();
    render(state);
    const job = state.command_job || {};
    if (job.status !== 'running' && job.status !== 'queued') {
      clearInterval(commandPollTimer);
      commandPollTimer = null;
    }
  }, 1000);
}

renderPhysicalObstacleEditor();
buildScene();

window.addEventListener('keydown', (event) => {
  if (event.key === 'w' || event.key === 'W') pressedKeys.w = true;
  if (event.key === 'a' || event.key === 'A') pressedKeys.a = true;
  if (event.key === 's' || event.key === 'S') pressedKeys.s = true;
  if (event.key === 'd' || event.key === 'D') pressedKeys.d = true;
  if (event.key === 'ArrowLeft') pressedKeys.left = true;
  if (event.key === 'ArrowRight') pressedKeys.right = true;
});

window.addEventListener('keyup', (event) => {
  if (event.key === 'w' || event.key === 'W') pressedKeys.w = false;
  if (event.key === 'a' || event.key === 'A') pressedKeys.a = false;
  if (event.key === 's' || event.key === 'S') pressedKeys.s = false;
  if (event.key === 'd' || event.key === 'D') pressedKeys.d = false;
  if (event.key === 'ArrowLeft') pressedKeys.left = false;
  if (event.key === 'ArrowRight') pressedKeys.right = false;
});
</script>
</body>
</html>
"""


def build_index_html() -> str:
    return (
        INDEX_HTML.replace("__REDIRECTORS__", json.dumps(sorted(REDIRECTOR_OPTIONS.keys())))
        .replace("__RESETTERS__", json.dumps(sorted(RESETTER_OPTIONS.keys())))
        .replace("__PATHS__", json.dumps(sorted(PATH_OPTIONS.keys())))
        .replace("__TRACKING_SPACES__", json.dumps(sorted([*TRACKING_SPACE_OPTIONS.keys(), "file_path"])))
    )


class SimulationSession:
    def __init__(self) -> None:
        self.scheduler = None
        self.config = SimulationConfig()
        self.step_index = 0
        self.command_job = {
            "status": "idle",
            "message": "No command file job running.",
        }
        self._command_job_lock = threading.Lock()

    def build(self, payload: dict) -> dict:
        self.config = SimulationConfig(**payload)
        self.scheduler = build_scheduler(self.config)
        self.step_index = 0
        return self.snapshot()

    def step(self, payload: dict | None = None) -> dict:
        if self.scheduler is None:
            self.scheduler = build_scheduler(self.config)
        if all(agent.state.mission_complete and not agent.state.in_reset for agent in self.scheduler.agents):
            return self.snapshot()
        manual_inputs = None if not payload else payload.get("manual_inputs")
        self.scheduler.step(self.step_index, manual_inputs=manual_inputs)
        self.step_index += 1
        return self.snapshot()

    def export(self) -> dict:
        if self.scheduler is None:
            self.scheduler = build_scheduler(self.config)
        output_dir = Path("python_openrdw/out")
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for agent in self.scheduler.agents:
            export_trace_csv(output_dir / f"ui_export_{timestamp}_agent_{agent.agent_id}.csv", agent.trace)
        return {"message": f"Exported traces to {output_dir}"}

    def run_command_file(self, payload: dict) -> dict:
        command_file = payload.get("command_file")
        output_dir = payload.get("output_dir")
        if not command_file:
            with self._command_job_lock:
                self.command_job = {
                    "status": "error",
                    "message": "command_file is required.",
                    "error": "command_file is required",
                }
            return self.snapshot()
        if not output_dir:
            output_dir = "python_openrdw/out/command_runs"
        with self._command_job_lock:
            if self.command_job.get("status") in {"running", "queued"}:
                return self.snapshot()
            self.command_job = {
                "status": "queued",
                "message": "Command file job queued.",
                "command_file": command_file,
                "output_dir": output_dir,
                "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        thread = threading.Thread(
            target=self._run_command_file_job,
            args=(str(command_file), str(output_dir), payload.get("max_steps", 30000), payload.get("sampling_frequency", 10.0)),
            daemon=True,
        )
        thread.start()
        return self.snapshot()

    def _run_command_file_job(self, command_file: str, output_dir: str, max_steps: int, sampling_frequency: float) -> None:
        with self._command_job_lock:
            self.command_job = {
                **self.command_job,
                "status": "running",
                "message": "Running command file experiment.",
                "command_file": command_file,
                "output_dir": output_dir,
            }
        try:
            summaries = run_experiment_command_file(
                command_file,
                output_dir=output_dir,
                max_steps=max_steps,
                sampling_frequency=sampling_frequency,
            )
            with self._command_job_lock:
                self.command_job = {
                    **self.command_job,
                    "status": "completed",
                    "message": "Experiment complete.",
                    "trial_count": len(summaries),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
        except Exception as exc:  # pragma: no cover - surfaced in UI
            with self._command_job_lock:
                self.command_job = {
                    **self.command_job,
                    "status": "error",
                    "message": "Experiment failed.",
                    "error": str(exc),
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

    def snapshot(self) -> dict:
        if self.scheduler is None:
            self.scheduler = build_scheduler(self.config)
        environment = self.scheduler.agents[0].environment
        with self._command_job_lock:
            command_job = dict(self.command_job)

        def decimated_trace(samples: list[dict[str, float]]) -> list[dict[str, float]]:
            decimated: list[dict[str, float]] = []
            for sample in samples:
                if not decimated:
                    decimated.append(sample)
                    continue
                last = decimated[-1]
                if ((sample["x"] - last["x"]) ** 2 + (sample["y"] - last["y"]) ** 2) ** 0.5 > 0.1:
                    decimated.append(sample)
            return decimated

        def virtual_trace_points(trace_rows) -> list[dict[str, float]]:
            return decimated_trace(
                [
                    {
                        "x": row.virtual_x,
                        "y": row.virtual_y,
                        "t": row.simulation_time_s,
                    }
                    for row in trace_rows
                ]
            )

        def physical_trace_points(trace_rows) -> list[dict[str, float]]:
            return decimated_trace(
                [
                    {
                        "x": row.observed_physical_x if row.observed_physical_x is not None else row.physical_x,
                        "y": row.observed_physical_y if row.observed_physical_y is not None else row.physical_y,
                        "t": row.simulation_time_s,
                    }
                    for row in trace_rows
                ]
            )

        return {
            "step_index": self.step_index,
            "all_mission_complete": all(agent.state.mission_complete and not agent.state.in_reset for agent in self.scheduler.agents),
            "config": self.config.__dict__,
            "gains": {
                "body_collider_diameter": self.scheduler.agents[0].gains.body_collider_diameter,
                "physical_space_buffer": self.scheduler.agents[0].gains.physical_space_buffer,
                "obstacle_buffer": self.scheduler.agents[0].gains.obstacle_buffer,
            },
            "physical": {
                "boundary": [point.__dict__ for point in environment.tracking_space],
                "obstacles": [[point.__dict__ for point in obstacle] for obstacle in environment.obstacles],
                "agents": [
                    {
                        "position": agent.state.physical_pose.position.__dict__,
                        "heading_deg": agent.state.physical_pose.heading_deg,
                    }
                    for agent in self.scheduler.agents
                ],
                "traces": [
                    physical_trace_points(agent.trace)
                    for agent in self.scheduler.agents
                ],
            },
            "virtual": {
                "boundary": [point.__dict__ for point in environment.all_virtual_polygons[0]],
                "obstacles": [[point.__dict__ for point in obstacle] for obstacle in environment.all_virtual_polygons[1:]],
                "agents": [
                    {
                        "position": agent.state.virtual_pose.position.__dict__,
                        "heading_deg": agent.state.virtual_pose.heading_deg,
                    }
                    for agent in self.scheduler.agents
                ],
                "targets": [
                    agent.state.active_waypoint.__dict__ if agent.state.active_waypoint is not None else None
                    for agent in self.scheduler.agents
                ],
                "traces": [
                    virtual_trace_points(agent.trace)
                    for agent in self.scheduler.agents
                ],
            },
            "agent_states": [
                {
                    "agent_id": agent.agent_id,
                    "virtual_position": agent.state.virtual_pose.position.__dict__,
                    "physical_position": agent.state.physical_pose.position.__dict__,
                    "virtual_heading_deg": agent.state.virtual_pose.heading_deg,
                    "physical_heading_deg": agent.state.physical_pose.heading_deg,
                    "root_pose": {
                        "position": agent.state.root_pose.position.__dict__,
                        "heading_deg": agent.state.root_pose.heading_deg,
                    },
                    "prev_virtual_pose": {
                        "position": agent.state.prev_virtual_pose.position.__dict__,
                        "heading_deg": agent.state.prev_virtual_pose.heading_deg,
                    } if agent.state.prev_virtual_pose is not None else None,
                    "prev_physical_pose": {
                        "position": agent.state.prev_physical_pose.position.__dict__,
                        "heading_deg": agent.state.prev_physical_pose.heading_deg,
                    } if agent.state.prev_physical_pose is not None else None,
                    "prev_root_pose": {
                        "position": agent.state.prev_root_pose.position.__dict__,
                        "heading_deg": agent.state.prev_root_pose.heading_deg,
                    } if agent.state.prev_root_pose is not None else None,
                    "tracking_space_pose": {
                        "position": agent.state.tracking_space_pose.position.__dict__,
                        "heading_deg": agent.state.tracking_space_pose.heading_deg,
                    },
                    "observed_tracking_space_pose": {
                        "position": agent.state.observed_tracking_space_pose.position.__dict__,
                        "heading_deg": agent.state.observed_tracking_space_pose.heading_deg,
                    } if agent.state.observed_tracking_space_pose is not None else None,
                    "tracking_space_local_position": agent.state.tracking_space_local_position.__dict__,
                    "tracking_space_local_heading_deg": agent.state.tracking_space_local_heading_deg,
                    "display_heading_deg": self._display_heading_deg(agent.state),
                    "observed_physical_pose": {
                        "position": agent.state.observed_physical_pose.position.__dict__,
                        "heading_deg": agent.state.observed_physical_pose.heading_deg,
                    } if agent.state.observed_physical_pose is not None else None,
                    "physical_delta_translation": agent.state.physical_delta_translation.__dict__,
                    "delta_virtual_translation": agent.state.delta_virtual_translation.__dict__,
                    "delta_virtual_rotation_deg": agent.state.delta_virtual_rotation_deg,
                    "delta_physical_translation": agent.state.delta_physical_translation.__dict__,
                    "delta_physical_rotation_deg": agent.state.delta_physical_rotation_deg,
                    "curr_pos": agent.state.curr_pos.__dict__,
                    "curr_pos_real": agent.state.curr_pos_real.__dict__,
                    "prev_pos": agent.state.prev_pos.__dict__,
                    "prev_pos_real": agent.state.prev_pos_real.__dict__,
                    "delta_pos": agent.state.delta_pos.__dict__,
                    "delta_dir": agent.state.delta_dir,
                    "same_pos_time": agent.state.same_pos_time,
                    "current_waypoint": agent.state.current_waypoint,
                    "active_waypoint": agent.state.active_waypoint.__dict__ if agent.state.active_waypoint is not None else None,
                    "translation_gain": agent.state.last_command.translation_gain,
                    "rotation_gain": agent.state.last_command.rotation_gain,
                    "curvature_gain": agent.state.last_command.curvature_gain,
                    "in_reset": agent.state.in_reset,
                    "mission_complete": agent.state.mission_complete,
                }
                for agent in self.scheduler.agents
            ],
            "command_job": command_job,
        }

    def _display_heading_deg(self, state) -> float:
        tracking_pose = state.observed_tracking_space_pose or state.tracking_space_pose
        tracking_heading = tracking_pose.heading_deg if tracking_pose is not None else (
            state.virtual_pose.heading_deg - state.physical_pose.heading_deg
        )
        if state.physical_delta_translation.magnitude > 1e-6:
            return vector_to_heading(state.physical_delta_translation.rotate(tracking_heading))
        return vector_to_heading(heading_to_vector(state.physical_pose.heading_deg).rotate(tracking_heading))


class OpenRDWHandler(BaseHTTPRequestHandler):
    session = SimulationSession()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            html = build_index_html()
            self._send_text(html, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(self.session.snapshot())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if parsed.path == "/api/build":
            self._send_json(self.session.build(payload))
            return
        if parsed.path == "/api/run_command_file":
            self._send_json(self.session.run_command_file(payload))
            return
        if parsed.path == "/api/step":
            self._send_json(self.session.step(payload))
            return
        if parsed.path == "/api/export":
            self._send_json(self.session.export())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, body: str, content_type: str) -> None:
        data = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    max_port_tries: int = 20,
) -> tuple[ThreadingHTTPServer, int]:
    last_error: OSError | None = None
    for candidate_port in range(port, port + max_port_tries):
        try:
            server = ThreadingHTTPServer((host, candidate_port), OpenRDWHandler)
            if open_browser:
                threading.Timer(0.5, lambda: webbrowser.open(f"http://{host}:{candidate_port}/")).start()
            return server, candidate_port
        except OSError as exc:
            last_error = exc
            if exc.errno != EADDRINUSE:
                raise
    assert last_error is not None
    raise last_error


def main() -> None:
    host = os.environ.get("OPENRDW_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("OPENRDW_UI_PORT", "8765"))
    server, actual_port = serve(host=host, port=port)
    print(f"OpenRDW UI available at http://{host}:{actual_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
