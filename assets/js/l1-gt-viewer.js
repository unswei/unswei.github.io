async function loadL1GtViewer(root) {
  const canvas = root.querySelector('canvas');
  const status = root.querySelector('.robot-status');
  const sliders = [...root.querySelectorAll('[data-joint]')];
  const viewport = root.querySelector('.robot-viewport');
  const poster = root.querySelector('.robot-poster');
  const controls = root.querySelector('[data-viewer-controls]');

  try {
    status.hidden = false;
    const gl = canvas.getContext('webgl2', {alpha:false, antialias:true});
    if (!gl) throw new Error('This viewer requires WebGL 2.');
    const shaders = [
      [gl.VERTEX_SHADER, `#version 300 es
        in vec3 position;
        uniform vec3 origin, axis, target, right, up, backward;
        uniform vec2 halfSize;
        uniform float angle;
        out vec3 world;
        void main() {
          float c = cos(angle), s = sin(angle);
          world = origin + c*position + s*cross(axis,position) + (1.0-c)*dot(axis,position)*axis;
          vec3 relative = world - target;
          gl_Position = vec4(dot(right,relative)/halfSize.x, dot(up,relative)/halfSize.y,
                            -0.65*dot(backward,relative), 1.0);
        }`],
      [gl.FRAGMENT_SHADER, `#version 300 es
        precision highp float;
        uniform vec3 colour, backward;
        uniform bool unlit;
        in vec3 world;
        out vec4 pixel;
        void main() {
          if (unlit) { pixel = vec4(colour, 1.0); return; }
          vec3 n = normalize(cross(dFdx(world),dFdy(world)));
          vec3 light = normalize(vec3(0.55,-0.4,0.9));
          float key = max(dot(n,light),0.0);
          float fill = max(dot(n,normalize(vec3(-0.5,0.7,0.2))),0.0);
          float shine = pow(max(dot(n,normalize(light+backward)),0.0),25.0);
          pixel = vec4(colour*(0.56+0.44*key+0.14*fill)+0.12*shine,1.0);
        }`]
    ];
    const program = gl.createProgram();
    for (const [kind, source] of shaders) {
      const shader = gl.createShader(kind);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
      gl.deleteShader(shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    gl.enable(gl.DEPTH_TEST);
    const position = gl.getAttribLocation(program, 'position');
    const uniform = Object.fromEntries(['origin','axis','target','right','up','backward','halfSize','angle','colour','unlit']
      .map(name => [name, gl.getUniformLocation(program, name)]));
    const glVector = (name, values) => gl.uniform3fv(uniform[name], values);
    const buffer = (kind, data) => {
      const handle = gl.createBuffer();
      gl.bindBuffer(kind, handle);
      gl.bufferData(kind, data, gl.STATIC_DRAW);
      return handle;
    };
    const response = await fetch(root.dataset.l1GtViewer);
    if (!response.ok) throw new Error(`Mesh request failed (${response.status}).`);
    const asset = await response.arrayBuffer();
    // The asset starts with a little-endian metadata length and UTF-8 JSON,
    // followed by a gzip-compressed buffer for each visual mesh.
    const metadataLength = new DataView(asset).getUint32(0, true);
    const payload = JSON.parse(new TextDecoder().decode(new Uint8Array(asset, 4, metadataLength)));
    let offset = 4 + metadataLength;
    const meshes = await Promise.all(payload.map(async mesh => {
      const compressed = new Uint8Array(asset, offset, mesh.byteLength);
      offset += mesh.byteLength;
      const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
      const raw = await new Response(stream).arrayBuffer();
      const deltas = new Int16Array(raw, 0, mesh.vertexCount * 3);
      const faces = new Uint16Array(raw, deltas.byteLength);
      const vertices = new Float32Array(deltas.length);
      const previous = [0,0,0];
      for (let i=0; i<deltas.length; i++) {
        const axis = i % 3;
        previous[axis] += deltas[i];
        vertices[i] = previous[axis] * 0.0001;
      }
      let first = 0;
      for (let i=0; i<faces.length; i+=3) {
        first += faces[i];
        faces[i] = first;
        faces[i+1] += first;
        faces[i+2] += first;
      }
      const colour = mesh.finish === 'tread' ? 'tread'
        : mesh.link.includes('_wheel_') ? 'rim'
        : mesh.link === 'left_arm_link' ? 'left'
        : mesh.link === 'right_arm_link' ? 'right' : 'body';
      return {origin:mesh.origin, axis:mesh.axis, joint:mesh.joint, colour,
              vertexBuffer:buffer(gl.ARRAY_BUFFER,vertices), indexBuffer:buffer(gl.ELEMENT_ARRAY_BUFFER,faces), count:faces.length};
    }));
    const grid = [];
    for (let i=-6; i<=6; i++) {
      const a = i*0.075, edge = 0.45, z = -0.097;
      grid.push(a,-edge,z, a,edge,z, -edge,a,z, edge,a,z);
    }
    const gridBuffer = buffer(gl.ARRAY_BUFFER, new Float32Array(grid));
    const paletteContext = document.createElement('canvas').getContext('2d', {willReadFrequently:true});
    const colours = Object.fromEntries([...root.querySelectorAll('[data-colour]')].map(el => [el.dataset.colour, el]));
    const palette = {};
    function readColours() {
      for (const [key, element] of Object.entries(colours)) {
        paletteContext.clearRect(0,0,1,1);
        paletteContext.fillStyle = getComputedStyle(element).color;
        paletteContext.fillRect(0,0,1,1);
        palette[key] = Array.from(paletteContext.getImageData(0,0,1,1).data.subarray(0,3), value => value/255);
      }
      scheduleDraw();
    }
    const angles = Object.fromEntries(sliders.map(input => [input.dataset.joint, 0]));
    const camera = {yaw:-0.93, pitch:0.4, size:0.425, target:[0,0,0.185]};
    const clamp = (n, min, max) => Math.min(max, Math.max(min, n));
    function basis() {
      const cy=Math.cos(camera.yaw), sy=Math.sin(camera.yaw);
      const ct=Math.cos(camera.pitch), st=Math.sin(camera.pitch);
      return {right:[-sy,cy,0], up:[-st*cy,-st*sy,ct], backward:[ct*cy,ct*sy,st]};
    }
    let redraw = null;
    function scheduleDraw() {
      if (redraw === null) redraw = requestAnimationFrame(draw);
    }
    function bindVertices(handle) {
      gl.bindBuffer(gl.ARRAY_BUFFER, handle);
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position,3,gl.FLOAT,false,0,0);
    }
    function draw() {
      redraw = null;
      if (canvas.clientWidth === 0 || canvas.clientHeight === 0) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.round(canvas.clientWidth*dpr), height = Math.round(canvas.clientHeight*dpr);
      if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
      const aspect = width/height;
      const halfHeight = camera.size * Math.max(1, 1/aspect);
      gl.viewport(0,0,width,height);
      gl.clearColor(...palette.background, 1);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      for (const [name, vector] of Object.entries(basis())) glVector(name,vector);
      glVector('target', camera.target);
      gl.uniform2f(uniform.halfSize, halfHeight*aspect, halfHeight);
      glVector('origin',[0,0,0]);
      glVector('axis',[0,1,0]);
      gl.uniform1f(uniform.angle,0);
      gl.uniform1i(uniform.unlit,1);
      glVector('colour',palette.grid);
      bindVertices(gridBuffer);
      gl.drawArrays(gl.LINES,0,grid.length/3);
      gl.uniform1i(uniform.unlit,0);
      for (const mesh of meshes) {
        glVector('origin',mesh.origin);
        glVector('axis',mesh.axis);
        glVector('colour',palette[mesh.colour]);
        gl.uniform1f(uniform.angle,(angles[mesh.joint] ?? 0)*Math.PI/180);
        bindVertices(mesh.vertexBuffer);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,mesh.indexBuffer);
        gl.drawElements(gl.TRIANGLES,mesh.count,gl.UNSIGNED_SHORT,0);
      }
    }
    function setAngle(input) {
      angles[input.dataset.joint] = input.valueAsNumber;
      const value = `${input.value}°`;
      root.querySelector(`output[for="${input.id}"]`).value = value;
      input.setAttribute('aria-valuetext',`${input.value} degrees`);
      scheduleDraw();
    }
    for (const input of sliders) {
      input.addEventListener('input', () => setAngle(input));
      setAngle(input);
    }
    root.querySelector('[data-reset-pose]').addEventListener('click', () => {
      for (const input of sliders) {
        input.value = 0;
        setAngle(input);
      }
    });
    const pointers = new Map();
    function pan(dx, dy) {
      const {right, up} = basis();
      const pixelScale = 2*camera.size/Math.min(canvas.clientWidth,canvas.clientHeight);
      camera.target = camera.target.map((value,i) => value + pixelScale*(-dx*right[i] + dy*up[i]));
    }
    function zoom(factor) {
      camera.size = clamp(camera.size*factor,0.045,1.6);
    }
    canvas.addEventListener('pointerdown', event => {
      if (!event.isPrimary) return;
      if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId,[event.clientX,event.clientY]);
    });
    canvas.addEventListener('pointermove', event => {
      const old = pointers.get(event.pointerId);
      if (!old) return;
      const point = [event.clientX,event.clientY];
      const dx = point[0]-old[0], dy = point[1]-old[1];
      pointers.set(event.pointerId,point);
      if (event.shiftKey || (event.buttons & 6)) {
        pan(dx, dy);
      } else {
        orbit(-dx * 0.009, dy * 0.009);
      }
      scheduleDraw();
    });
    canvas.addEventListener('lostpointercapture', event => pointers.delete(event.pointerId));
    canvas.addEventListener('pointercancel', event => pointers.delete(event.pointerId));
    canvas.addEventListener('contextmenu', event => event.preventDefault());
    function orbit(yaw, pitch) {
      camera.yaw += yaw;
      camera.pitch = clamp(camera.pitch + pitch, -1.54, 1.54);
    }
    for (const button of root.querySelectorAll('[data-zoom]')) {
      button.addEventListener('click', () => {
        zoom(Number(button.dataset.zoom));
        scheduleDraw();
      });
    }
    viewport.addEventListener('keydown', event => {
      const step = Math.PI / 24;
      switch (event.key) {
        case 'ArrowLeft': orbit(step, 0); break;
        case 'ArrowRight': orbit(-step, 0); break;
        case 'ArrowUp': orbit(0, step); break;
        case 'ArrowDown': orbit(0, -step); break;
        case '+': case '=': zoom(0.8); break;
        case '-': zoom(1.25); break;
        default: return;
      }
      event.preventDefault();
      scheduleDraw();
    });
    canvas.addEventListener('webglcontextlost', event => {
      event.preventDefault();
      status.textContent = '3D rendering stopped. Reload the page to restart it.';
      status.hidden = false;
      canvas.hidden = true;
      poster.hidden = false;
      controls.disabled = true;
      viewport.removeAttribute('tabindex');
    });
    new ResizeObserver(scheduleDraw).observe(canvas);
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change',readColours);
    const observer = new MutationObserver(readColours);
    for (const element of [document.documentElement, document.body]) {
      observer.observe(element, {attributes: true, attributeFilter: ['data-theme']});
    }
    readColours();
    canvas.hidden = false;
    poster.hidden = true;
    status.hidden = true;
    controls.hidden = false;
    controls.disabled = false;
    root.dataset.ready = '';
    viewport.tabIndex = 0;
  } catch (error) {
    console.error(error);
    status.hidden = false;
    status.textContent = '3D preview unavailable. The image shows the L1-GT design.';
  }
}

const viewer = document.querySelector('[data-l1-gt-viewer]');
if (viewer) {
  const visibility = new IntersectionObserver(entries => {
    if (!entries.some(entry => entry.isIntersecting)) return;
    visibility.disconnect();
    loadL1GtViewer(viewer);
  }, {rootMargin: '160px'});
  visibility.observe(viewer);
}
