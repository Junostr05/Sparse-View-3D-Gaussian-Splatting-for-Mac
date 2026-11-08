import Foundation

/// The HTML template of the exported web page.
///
/// The page is one file with no external references. It has four parts, filled in by `WebViewerExporter`:
/// - `%%CONFIG%%`: the JSON configuration (title, counts, up axis, initial camera, background),
/// - `%%DATA%%`: the gzip-compressed `.splat` records, base64-encoded,
/// - `%%DATA_RAW%%`: an optional uncompressed copy of the records, or nothing,
/// - `%%TITLE%%`: the escaped page title. It is filled in last.
///
/// Rendering (WebGL 2, written from the algorithm, not from any other project's code):
/// - Each Gaussian is one instanced quad. Its textures hold the centre and opacity, the six entries of its
///   3D covariance, and its colour. The covariance is Σ = (R·S)(R·S)ᵀ with S = diag(scale).
/// - The vertex shader projects the covariance with the camera Jacobian (EWA splatting). The lateral
///   position used in the Jacobian is clamped to 1.3 × the field of view, and 0.3 px is added to the
///   diagonal of the 2D covariance, as the app's rasterizer does. It outputs the conic (inverse 2D covariance).
/// - The fragment shader evaluates exp(−½ dᵀ Σ⁻¹ d) times the opacity, discards values below 1/255, and blends
///   premultiplied colour with `ONE, ONE_MINUS_SRC_ALPHA`, drawn back to front.
/// - The draw order is a counting sort on quantised depth along the view direction, run in a Web Worker
///   built from a Blob. The last order is kept until the next one arrives. Without Worker support the sort
///   runs on the main thread.
///
/// Controls: left drag orbits, right drag or Shift + left drag pans, the wheel zooms, one finger orbits, a
/// pinch zooms and two fingers pan on touch screens, a double click turns the view to the point under the
/// pointer, R resets the view, and F shows the frame rate.
enum WebViewerPage {
    static func render(title: String, config: String, dataBase64: String, rawBlock: String) -> String {
        template
            .replacingOccurrences(of: "%%CONFIG%%", with: config)
            .replacingOccurrences(of: "%%DATA%%", with: dataBase64)
            .replacingOccurrences(of: "%%DATA_RAW%%", with: rawBlock)
            .replacingOccurrences(of: "%%TITLE%%", with: title)
    }

    static let template: String = #"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #000; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
  [hidden] { display: none !important; }
  #scene { position: fixed; left: 0; top: 0; width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; outline: none; }
  #scene:active { cursor: grabbing; }
  #overlay { position: fixed; left: 16px; bottom: 14px; color: #fff; font-size: 12px; line-height: 1.45; opacity: 0.75; pointer-events: none; user-select: none; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6); }
  #overlay .title { font-size: 13px; font-weight: 600; }
  body.light #overlay { color: #111; text-shadow: 0 1px 2px rgba(255, 255, 255, 0.6); }
  #fps { position: fixed; right: 16px; top: 12px; color: #fff; font-size: 12px; font-variant-numeric: tabular-nums; opacity: 0.75; pointer-events: none; text-shadow: 0 1px 2px rgba(0, 0, 0, 0.6); }
  body.light #fps { color: #111; text-shadow: 0 1px 2px rgba(255, 255, 255, 0.6); }
  #message { position: fixed; left: 0; top: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; padding: 16px; box-sizing: border-box; text-align: center; color: #fff; font-size: 15px; line-height: 1.5; }
</style>
</head>
<body>
<canvas id="scene" tabindex="0" aria-label="Gaussian splat scene. Drag to orbit, right-drag or Shift-drag to pan, scroll to zoom."></canvas>
<div id="overlay">
  <div class="title" id="title"></div>
  <div id="stats"></div>
  <div>R: reset &middot; drag: orbit &middot; scroll: zoom</div>
</div>
<div id="fps" hidden></div>
<div id="message" hidden></div>
<script id="config" type="application/json">%%CONFIG%%</script>
<script id="data" type="application/octet-stream">%%DATA%%</script>
%%DATA_RAW%%
<script>
(function () {
  'use strict';

  var TEXTURE_WIDTH = 2048;
  var RECORD_BYTES = 32;
  var SORT_COS = Math.cos(0.5 * Math.PI / 180);
  var ORBIT_SPEED = 0.006;
  var FPS_INTERVAL_MS = 500;

  var canvas = document.getElementById('scene');
  var messageEl = document.getElementById('message');
  var fpsEl = document.getElementById('fps');
  var statsEl = document.getElementById('stats');

  function showMessage(text) {
    messageEl.textContent = text;
    messageEl.hidden = false;
  }

  var configEl = document.getElementById('config');
  var config;
  try {
    config = JSON.parse(configEl.textContent);
  } catch (error) {
    showMessage('The page configuration is damaged.');
    return;
  }

  document.title = config.title;
  document.getElementById('title').textContent = config.title;
  var background = config.background;
  var luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2];
  if (luminance > 0.5) {
    document.body.classList.add('light');
  }
  document.body.style.background = 'rgb(' + Math.round(background[0] * 255) + ',' +
    Math.round(background[1] * 255) + ',' + Math.round(background[2] * 255) + ')';

  // ---- Small vector helpers. Vectors are plain arrays of three numbers.

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function length(a) { return Math.sqrt(dot(a, a)); }
  function normalize(a) {
    var n = length(a);
    return n > 0 ? scale(a, 1 / n) : [0, 0, 0];
  }
  function clamp(value, low, high) { return Math.min(Math.max(value, low), high); }
  // Rotates v about a unit axis by angle (Rodrigues' formula).
  function rotate(v, axis, angle) {
    var c = Math.cos(angle);
    var s = Math.sin(angle);
    return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
  }

  var up = normalize(config.up);
  var fovY = config.camera.fovY;

  // ---- Camera. The eye is target - forward * distance. The axes follow the viewer: right = forward x up,
  // down = forward x right (OpenCV: x right, y down, z forward).

  function rightOf(c) { return normalize(cross(c.forward, up)); }
  function downOf(c, r) { return normalize(cross(c.forward, r)); }
  function eyeOf(c) { return sub(c.target, scale(c.forward, c.distance)); }
  function copyCamera(c) { return { target: c.target.slice(), distance: c.distance, forward: c.forward.slice() }; }

  // World-to-camera transform as the shader wants it: a column-major 3x3 rotation whose rows are right, down and
  // forward, and the translation -R·eye.
  function viewOf(c) {
    var eye = eyeOf(c);
    var r = rightOf(c);
    var d = downOf(c, r);
    var f = c.forward;
    return {
      rotation: [r[0], d[0], f[0], r[1], d[1], f[1], r[2], d[2], f[2]],
      translation: [-dot(r, eye), -dot(d, eye), -dot(f, eye)],
      right: r,
      down: d,
      eye: eye
    };
  }

  // ---- Scene decoding.

  function base64ToBytes(text) {
    var binary = atob(text.replace(/\s+/g, ''));
    var bytes = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes;
  }

  function gunzip(bytes) {
    var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).arrayBuffer();
  }

  function loadRecords() {
    if (typeof DecompressionStream === 'function') {
      return gunzip(base64ToBytes(document.getElementById('data').textContent));
    }
    var raw = document.getElementById('data-raw');
    if (raw) {
      return Promise.resolve(base64ToBytes(raw.textContent).buffer);
    }
    return Promise.reject(new Error('This browser cannot decompress the scene. Use a current version of Safari, Chrome or Firefox.'));
  }

  // Turns 32-byte records into texel arrays. Layout per record: position (3 x float32), linear scale (3 x float32),
  // RGBA (4 x uint8, alpha is the opacity), rotation w, x, y, z (4 x uint8, value = q * 128 + 128).
  function decodeRecords(buffer) {
    var count = Math.floor(buffer.byteLength / RECORD_BYTES);
    if (count < 1) {
      throw new Error('The scene contains no Gaussians.');
    }
    var view = new DataView(buffer);
    var bytes = new Uint8Array(buffer);
    var width = Math.min(TEXTURE_WIDTH, count);
    var height = Math.ceil(count / width);
    var texels = width * height;
    var centers = new Float32Array(texels * 4);
    var cov0 = new Float32Array(texels * 4);
    var cov1 = new Float32Array(texels * 4);
    var colors = new Uint8Array(texels * 4);
    var lower = [Infinity, Infinity, Infinity];
    var upper = [-Infinity, -Infinity, -Infinity];

    for (var i = 0; i < count; i++) {
      var o = i * RECORD_BYTES;
      var t = i * 4;
      var x = view.getFloat32(o, true);
      var y = view.getFloat32(o + 4, true);
      var z = view.getFloat32(o + 8, true);
      var sx = view.getFloat32(o + 12, true);
      var sy = view.getFloat32(o + 16, true);
      var sz = view.getFloat32(o + 20, true);

      var qw = (bytes[o + 28] - 128) / 128;
      var qx = (bytes[o + 29] - 128) / 128;
      var qy = (bytes[o + 30] - 128) / 128;
      var qz = (bytes[o + 31] - 128) / 128;
      var qn = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
      if (qn > 0) {
        qw /= qn; qx /= qn; qy /= qn; qz /= qn;
      } else {
        qw = 1; qx = 0; qy = 0; qz = 0;
      }

      // Rotation matrix of the unit quaternion, entries r[row][col].
      var r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
      var r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
      var r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);

      // Sigma = sum over k of s_k^2 · column_k(R) · column_k(R)ᵀ.
      var a0 = sx * sx, a1 = sy * sy, a2 = sz * sz;
      var s00 = a0 * r00 * r00 + a1 * r01 * r01 + a2 * r02 * r02;
      var s01 = a0 * r00 * r10 + a1 * r01 * r11 + a2 * r02 * r12;
      var s02 = a0 * r00 * r20 + a1 * r01 * r21 + a2 * r02 * r22;
      var s11 = a0 * r10 * r10 + a1 * r11 * r11 + a2 * r12 * r12;
      var s12 = a0 * r10 * r20 + a1 * r11 * r21 + a2 * r12 * r22;
      var s22 = a0 * r20 * r20 + a1 * r21 * r21 + a2 * r22 * r22;

      centers[t] = x; centers[t + 1] = y; centers[t + 2] = z; centers[t + 3] = 0;
      cov0[t] = s00; cov0[t + 1] = s01; cov0[t + 2] = s02; cov0[t + 3] = s11;
      cov1[t] = s12; cov1[t + 1] = s22; cov1[t + 2] = 0; cov1[t + 3] = 0;
      colors[t] = bytes[o + 24];
      colors[t + 1] = bytes[o + 25];
      colors[t + 2] = bytes[o + 26];
      colors[t + 3] = bytes[o + 27];

      var p = [x, y, z];
      for (var c = 0; c < 3; c++) {
        if (p[c] < lower[c]) lower[c] = p[c];
        if (p[c] > upper[c]) upper[c] = p[c];
      }
    }

    var center = [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, (lower[2] + upper[2]) / 2];
    var radius = length(sub(upper, lower)) / 2;
    return {
      count: count, width: width, height: height,
      centers: centers, cov0: cov0, cov1: cov1, colors: colors,
      center: center, radius: radius > 0 ? radius : 1
    };
  }

  // ---- Depth sorting. This function is also the body of the worker, so it must not use outer variables.

  function sortBackToFront(centers, count, forward, scratch) {
    var bins = 65536;
    if (!scratch.counts) {
      scratch.counts = new Uint32Array(bins);
    }
    if (!scratch.depth || scratch.depth.length !== count) {
      scratch.depth = new Float32Array(count);
      scratch.bin = new Uint32Array(count);
    }
    var depth = scratch.depth;
    var bin = scratch.bin;
    var counts = scratch.counts;
    var fx = forward[0], fy = forward[1], fz = forward[2];
    var lo = Infinity, hi = -Infinity;
    var i;
    for (i = 0; i < count; i++) {
      var d = fx * centers[4 * i] + fy * centers[4 * i + 1] + fz * centers[4 * i + 2];
      depth[i] = d;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    var factor = hi > lo ? (bins - 1) / (hi - lo) : 0;
    counts.fill(0);
    for (i = 0; i < count; i++) {
      var q = Math.floor((depth[i] - lo) * factor);
      if (!(q >= 0)) {
        q = 0;
      } else if (q > bins - 1) {
        q = bins - 1;
      }
      bin[i] = q;
      counts[q]++;
    }
    // Largest depth first: the far splats are drawn before the near ones.
    var next = 0;
    for (var k = bins - 1; k >= 0; k--) {
      var c = counts[k];
      counts[k] = next;
      next += c;
    }
    var order = new Uint32Array(count);
    for (i = 0; i < count; i++) {
      order[counts[bin[i]]++] = i;
    }
    return order;
  }

  function sortWorkerMain() {
    var centers = null;
    var count = 0;
    var scratch = {};
    self.onmessage = function (event) {
      var message = event.data;
      if (message.type === 'init') {
        centers = message.centers;
        count = message.count;
        scratch = {};
        return;
      }
      var order = sortBackToFront(centers, count, message.forward, scratch);
      self.postMessage({ type: 'sorted', order: order }, [order.buffer]);
    };
  }

  // ---- WebGL 2.

  var gl = canvas.getContext('webgl2', {
    alpha: false, antialias: false, depth: false, stencil: false, premultipliedAlpha: true
  });
  if (!gl) {
    showMessage('This browser does not support WebGL 2. Use a current version of Safari, Chrome or Firefox.');
    return;
  }

  var VERTEX_SOURCE = [
    '#version 300 es',
    'precision highp float;',
    'precision highp int;',
    'precision highp sampler2D;',
    'uniform sampler2D uCenter;',
    'uniform sampler2D uCov0;',
    'uniform sampler2D uCov1;',
    'uniform sampler2D uColor;',
    'uniform int uWidth;',
    'uniform mat3 uView;',
    'uniform vec3 uTranslate;',
    'uniform vec2 uFocal;',
    'uniform vec2 uCenterPx;',
    'uniform vec2 uViewport;',
    'in uint aIndex;',
    'out vec2 vDelta;',
    'out vec3 vConic;',
    'out vec3 vColor;',
    'out float vAlpha;',
    'void main() {',
    '  int index = int(aIndex);',
    '  ivec2 texel = ivec2(index % uWidth, index / uWidth);',
    '  vec3 p = texelFetch(uCenter, texel, 0).xyz;',
    '  vec3 t = uView * p + uTranslate;',
    '  if (t.z < 0.05) {',
    '    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);',
    '    return;',
    '  }',
    '  vec2 mean = uFocal * t.xy / t.z + uCenterPx;',
    '  vec2 limit = 1.3 * uCenterPx / uFocal;',
    '  vec2 lateral = clamp(t.xy / t.z, -limit, limit) * t.z;',
    '  vec3 j0 = vec3(uFocal.x / t.z, 0.0, -uFocal.x * lateral.x / (t.z * t.z));',
    '  vec3 j1 = vec3(0.0, uFocal.y / t.z, -uFocal.y * lateral.y / (t.z * t.z));',
    '  vec3 t0 = transpose(uView) * j0;',
    '  vec3 t1 = transpose(uView) * j1;',
    '  vec4 q0 = texelFetch(uCov0, texel, 0);',
    '  vec4 q1 = texelFetch(uCov1, texel, 0);',
    '  mat3 sigma = mat3(vec3(q0.x, q0.y, q0.z), vec3(q0.y, q0.w, q1.x), vec3(q0.z, q1.x, q1.y));',
    '  vec3 s0 = sigma * t0;',
    '  vec3 s1 = sigma * t1;',
    '  float a = dot(t0, s0) + 0.3;',
    '  float b = dot(t0, s1);',
    '  float c = dot(t1, s1) + 0.3;',
    '  float det = a * c - b * b;',
    '  if (det <= 0.0) {',
    '    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);',
    '    return;',
    '  }',
    '  float mid = 0.5 * (a + c);',
    '  float lambda = mid + sqrt(max(0.1, mid * mid - det));',
    '  float radius = ceil(3.0 * sqrt(lambda));',
    '  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1)) * 2.0 - 1.0;',
    '  vec2 offset = corner * radius;',
    '  vec2 pixel = mean + offset;',
    '  vDelta = offset;',
    '  vConic = vec3(c, -b, a) / det;',
    '  vec4 color = texelFetch(uColor, texel, 0);',
    '  vColor = color.rgb;',
    '  vAlpha = color.a;',
    '  gl_Position = vec4(pixel.x / uViewport.x * 2.0 - 1.0, 1.0 - pixel.y / uViewport.y * 2.0, 0.0, 1.0);',
    '}'
  ].join('\n');

  var FRAGMENT_SOURCE = [
    '#version 300 es',
    'precision highp float;',
    'in vec2 vDelta;',
    'in vec3 vConic;',
    'in vec3 vColor;',
    'in float vAlpha;',
    'out vec4 fragColor;',
    'void main() {',
    '  float power = -0.5 * (vConic.x * vDelta.x * vDelta.x + 2.0 * vConic.y * vDelta.x * vDelta.y + vConic.z * vDelta.y * vDelta.y);',
    '  if (power > 0.0) discard;',
    '  float alpha = min(0.99, vAlpha * exp(power));',
    '  if (alpha < 1.0 / 255.0) discard;',
    '  fragColor = vec4(vColor * alpha, alpha);',
    '}'
  ].join('\n');

  function compileShader(type, source) {
    var shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  var program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, VERTEX_SOURCE));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, FRAGMENT_SOURCE));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(program));
  }

  var uniforms = {};
  ['uCenter', 'uCov0', 'uCov1', 'uColor', 'uWidth', 'uView', 'uTranslate', 'uFocal', 'uCenterPx', 'uViewport'].forEach(function (name) {
    uniforms[name] = gl.getUniformLocation(program, name);
  });

  var vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  var indexBuffer = gl.createBuffer();
  var indexLocation = gl.getAttribLocation(program, 'aIndex');
  gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
  gl.enableVertexAttribArray(indexLocation);
  gl.vertexAttribIPointer(indexLocation, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(indexLocation, 1);
  gl.bindVertexArray(null);

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  var textures = {};

  function uploadTextures(scene) {
    var names = ['center', 'cov0', 'cov1', 'color'];
    var sources = [scene.centers, scene.cov0, scene.cov1, scene.colors];
    var units = [gl.TEXTURE0, gl.TEXTURE1, gl.TEXTURE2, gl.TEXTURE3];
    for (var i = 0; i < names.length; i++) {
      var texture = gl.createTexture();
      gl.activeTexture(units[i]);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      if (names[i] === 'color') {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, scene.width, scene.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, sources[i]);
      } else {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, scene.width, scene.height, 0, gl.RGBA, gl.FLOAT, sources[i]);
      }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      textures[names[i]] = { texture: texture, unit: i };
    }
    var identity = new Uint32Array(scene.count);
    for (var k = 0; k < scene.count; k++) {
      identity[k] = k;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, identity, gl.DYNAMIC_DRAW);
  }

  // ---- State shared by the renderer and the input handlers.

  var scene = null;
  var home = null;
  var cam = null;
  var minDistance = 1;
  var maxDistance = 1;
  var worker = null;
  var sortPending = false;
  var sortForward = null;
  var inlineScratch = {};

  function cameraFromConfig(c) {
    var target = c.camera.target;
    var offset = sub(target, c.camera.position);
    var distance = length(offset);
    var forward;
    if (distance > 1e-9 && isFinite(distance)) {
      forward = normalize(offset);
    } else {
      distance = scene.radius * 2;
      forward = normalize(cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]));
    }
    if (Math.abs(dot(forward, up)) > 0.999) {
      forward = normalize(cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]));
    }
    return { target: target.slice(), distance: distance, forward: forward };
  }

  function applySortedOrder(order) {
    sortPending = false;
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, order);
    requestSort();
  }

  // Starts a sort when the view direction has moved by more than half a degree since the last one. The order
  // depends only on the view direction. At most one sort is in flight, and the old order stays in use until it arrives.
  function requestSort() {
    if (sortPending || !scene) return;
    if (sortForward && dot(cam.forward, sortForward) > SORT_COS) return;
    sortForward = cam.forward.slice();
    if (worker) {
      sortPending = true;
      worker.postMessage({ type: 'sort', forward: sortForward });
    } else {
      applySortedOrder(sortBackToFront(scene.centers, scene.count, sortForward, inlineScratch));
    }
  }

  function startWorker() {
    try {
      var source = sortBackToFront.toString() + '\n(' + sortWorkerMain.toString() + ')();';
      var url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
      var created = new Worker(url);
      created.onmessage = function (event) {
        applySortedOrder(event.data.order);
      };
      created.onerror = function () {
        worker = null;
        sortPending = false;
      };
      var copy = scene.centers.slice();
      created.postMessage({ type: 'init', centers: copy, count: scene.count }, [copy.buffer]);
      worker = created;
    } catch (error) {
      worker = null;
    }
  }

  function draw() {
    var width = canvas.width;
    var height = canvas.height;
    gl.viewport(0, 0, width, height);
    gl.clearColor(background[0], background[1], background[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!scene) return;

    var v = viewOf(cam);
    var focal = (height / 2) / Math.tan(fovY / 2);
    gl.useProgram(program);
    gl.bindVertexArray(vao);
    gl.uniformMatrix3fv(uniforms.uView, false, v.rotation);
    gl.uniform3f(uniforms.uTranslate, v.translation[0], v.translation[1], v.translation[2]);
    gl.uniform2f(uniforms.uFocal, focal, focal);
    gl.uniform2f(uniforms.uCenterPx, width / 2, height / 2);
    gl.uniform2f(uniforms.uViewport, width, height);
    gl.uniform1i(uniforms.uWidth, scene.width);
    ['center', 'cov0', 'cov1', 'color'].forEach(function (name, index) {
      gl.activeTexture(gl.TEXTURE0 + index);
      gl.bindTexture(gl.TEXTURE_2D, textures[name].texture);
    });
    gl.uniform1i(uniforms.uCenter, 0);
    gl.uniform1i(uniforms.uCov0, 1);
    gl.uniform1i(uniforms.uCov1, 2);
    gl.uniform1i(uniforms.uColor, 3);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, scene.count);
    gl.bindVertexArray(null);
  }

  function resize() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    var width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    var height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  }

  // ---- Camera motion.

  function orbit(dx, dy) {
    // Dragging right turns the scene right (the camera moves left around the target). Dragging down looks down.
    cam.forward = normalize(rotate(cam.forward, up, -dx * ORBIT_SPEED));
    var r = rightOf(cam);
    var tilted = rotate(cam.forward, r, -dy * ORBIT_SPEED);
    if (Math.abs(dot(tilted, up)) <= 0.995) {
      cam.forward = normalize(tilted);
    }
  }

  function pan(dx, dy) {
    var r = rightOf(cam);
    var d = downOf(cam, r);
    var worldPerPixel = 2 * cam.distance * Math.tan(fovY / 2) / canvas.clientHeight;
    cam.target = add(cam.target, add(scale(r, -dx * worldPerPixel), scale(d, -dy * worldPerPixel)));
  }

  function zoom(factor) {
    cam.distance = clamp(cam.distance * factor, minDistance, maxDistance);
  }

  function resetView() {
    cam = copyCamera(home);
  }

  // Turns the view to the splat under the pointer, keeping the eye where it is.
  function refocus(clientX, clientY) {
    var rect = canvas.getBoundingClientRect();
    var px = (clientX - rect.left) * (canvas.width / rect.width);
    var py = (clientY - rect.top) * (canvas.height / rect.height);
    var focal = (canvas.height / 2) / Math.tan(fovY / 2);
    var halfW = canvas.width / 2;
    var halfH = canvas.height / 2;
    var pickRadius = 16 * (canvas.width / rect.width);
    var eye = eyeOf(cam);
    var r = rightOf(cam);
    var d = downOf(cam, r);
    var f = cam.forward;
    var best = -1;
    var bestDepth = Infinity;
    var limit = pickRadius * pickRadius;
    for (var i = 0; i < scene.count; i++) {
      var rel = [scene.centers[4 * i] - eye[0], scene.centers[4 * i + 1] - eye[1], scene.centers[4 * i + 2] - eye[2]];
      var cz = dot(f, rel);
      if (cz <= 0.01) continue;
      var sx = focal * dot(r, rel) / cz + halfW;
      var sy = focal * dot(d, rel) / cz + halfH;
      var dx = sx - px;
      var dy = sy - py;
      if (dx * dx + dy * dy <= limit && cz < bestDepth) {
        best = i;
        bestDepth = cz;
      }
    }
    if (best < 0) return;
    var point = [scene.centers[4 * best], scene.centers[4 * best + 1], scene.centers[4 * best + 2]];
    var toPoint = sub(point, eye);
    var distance = length(toPoint);
    if (!(distance > 0)) return;
    cam.target = point;
    cam.distance = clamp(distance, minDistance, maxDistance);
    cam.forward = normalize(toPoint);
  }

  // ---- Input.

  var pointers = new Map();
  var panning = false;
  var pinch = null;

  canvas.addEventListener('contextmenu', function (event) { event.preventDefault(); });

  canvas.addEventListener('pointerdown', function (event) {
    if (!cam) return;
    canvas.setPointerCapture(event.pointerId);
    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    panning = event.button === 1 || event.button === 2 || event.shiftKey;
    pinch = null;
  });

  canvas.addEventListener('pointermove', function (event) {
    var pointer = pointers.get(event.pointerId);
    if (!pointer || !cam) return;
    var dx = event.clientX - pointer.x;
    var dy = event.clientY - pointer.y;
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    if (pointers.size === 1) {
      if (panning) {
        pan(dx, dy);
      } else {
        orbit(dx, dy);
      }
    } else if (pointers.size === 2) {
      var pair = Array.from(pointers.values());
      var cx = (pair[0].x + pair[1].x) / 2;
      var cy = (pair[0].y + pair[1].y) / 2;
      var spread = Math.hypot(pair[0].x - pair[1].x, pair[0].y - pair[1].y);
      if (pinch && pinch.spread > 0 && spread > 0) {
        zoom(pinch.spread / spread);
        pan(cx - pinch.cx, cy - pinch.cy);
      }
      pinch = { spread: spread, cx: cx, cy: cy };
    }
  });

  function endPointer(event) {
    pointers.delete(event.pointerId);
    pinch = null;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  canvas.addEventListener('wheel', function (event) {
    event.preventDefault();
    if (!cam) return;
    var amount = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
    zoom(Math.exp(amount * 0.0015));
  }, { passive: false });

  canvas.addEventListener('dblclick', function (event) {
    if (scene && cam) {
      refocus(event.clientX, event.clientY);
    }
  });

  window.addEventListener('keydown', function (event) {
    if (!event.key || event.metaKey || event.ctrlKey || event.altKey) return;
    var key = event.key.toLowerCase();
    if (key === 'r' && cam) {
      resetView();
    } else if (key === 'f') {
      fpsEl.hidden = !fpsEl.hidden;
    }
  });

  window.addEventListener('resize', resize);

  // ---- Frame loop.

  var frames = 0;
  var lastFpsTime = 0;

  function frame(now) {
    resize();
    if (scene && cam) {
      requestSort();
      draw();
    }
    frames += 1;
    if (now - lastFpsTime >= FPS_INTERVAL_MS) {
      if (lastFpsTime > 0 && !fpsEl.hidden) {
        fpsEl.textContent = Math.round(frames * 1000 / (now - lastFpsTime)) + ' fps';
      }
      frames = 0;
      lastFpsTime = now;
    }
    window.requestAnimationFrame(frame);
  }

  // ---- Start.

  showMessage('Loading the scene…');
  loadRecords().then(function (buffer) {
    scene = decodeRecords(buffer);
    minDistance = scene.radius * 0.02;
    maxDistance = scene.radius * 50;
    uploadTextures(scene);
    home = cameraFromConfig({ camera: config.camera, up: config.up });
    cam = copyCamera(home);
    var shown = scene.count.toLocaleString();
    var stats = shown + ' Gaussians';
    if (config.total > config.count) {
      stats += ' (the most important of ' + config.total.toLocaleString() + ')';
    }
    statsEl.textContent = stats;
    startWorker();
    messageEl.hidden = true;
    window.requestAnimationFrame(frame);
  }).catch(function (error) {
    showMessage(error && error.message ? error.message : 'The scene could not be loaded.');
  });
})();
</script>
</body>
</html>

"""#
}
