/**
 * Three.js 3D Visualizer for ML Pipeline & Decision Spaces
 * Interactive WebGL scene featuring glowing data nodes, animated particle streams,
 * rotating hyperplanes, and multi-mode 3D inspection.
 */

class Pipeline3DScene {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) return;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.particles = null;
    this.pipelineNodes = [];
    this.pipelineLinks = [];
    this.decisionObjects = [];
    this.currentMode = 'pipeline'; // 'pipeline' | 'decision' | 'neural'
    
    this.mouseX = 0;
    this.mouseY = 0;
    this.targetRotationX = 0;
    this.targetRotationY = 0;
    this.isDragging = false;
    this.prevMousePos = { x: 0, y: 0 };
    this.clock = new THREE.Clock();

    this.init();
    this.buildPipelineScene();
    this.addEventListeners();
    this.animate();
  }

  init() {
    const width = this.container.clientWidth || 1200;
    const height = this.container.clientHeight || 480;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x070913, 0.015);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 1000);
    this.camera.position.set(0, 8, 38);
    this.camera.lookAt(0, 0, 0);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    // 4. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    const cyanPoint = new THREE.PointLight(0x00f2fe, 3.5, 60);
    cyanPoint.position.set(-15, 12, 10);
    this.scene.add(cyanPoint);

    const purplePoint = new THREE.PointLight(0xa855f7, 3.5, 60);
    purplePoint.position.set(15, -8, 10);
    this.scene.add(purplePoint);

    const emeraldPoint = new THREE.PointLight(0x10b981, 2.5, 40);
    emeraldPoint.position.set(0, 15, -10);
    this.scene.add(emeraldPoint);

    // 5. Ambient Grid & Particle System
    this.createBackgroundGrid();
    this.createBackgroundParticles();
  }

  createBackgroundGrid() {
    const grid = new THREE.GridHelper(80, 40, 0x00f2fe, 0x131b36);
    grid.position.y = -10;
    grid.material.opacity = 0.35;
    grid.material.transparent = true;
    this.scene.add(grid);
  }

  createBackgroundParticles() {
    const count = 700;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);

    const color1 = new THREE.Color(0x00f2fe);
    const color2 = new THREE.Color(0xa855f7);

    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 100;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 60;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 80;

      const mixed = color1.clone().lerp(color2, Math.random());
      colors[i * 3] = mixed.r;
      colors[i * 3 + 1] = mixed.g;
      colors[i * 3 + 2] = mixed.b;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.35,
      vertexColors: true,
      transparent: true,
      opacity: 0.65,
      blending: THREE.AdditiveBlending
    });

    this.particles = new THREE.Points(geometry, material);
    this.scene.add(this.particles);
  }

  buildPipelineScene() {
    this.clearDynamicObjects();

    const nodeData = [
      { id: 'raw', name: 'Raw Ingestion', color: 0x00f2fe, x: -22, y: 0, z: 0, sub: '15.7K Raw Stream' },
      { id: 'clean', name: 'Cleansing & Backfill', color: 0x3b82f6, x: -11, y: 3, z: -2, sub: 'Dedupe + Catalog Join' },
      { id: 'marts', name: '5 Star Data Marts', color: 0xa855f7, x: 0, y: -2, z: 0, sub: 'Daily / Cat / Region / LTV' },
      { id: 'ltv', name: 'Customer LTV Engine', color: 0xf59e0b, x: 11, y: 3, z: 2, sub: 'ML Churn & Spend Risk' },
      { id: 'dash', name: 'Streamlit & 3D Serving', color: 0x10b981, x: 22, y: 0, z: 0, sub: 'Real-Time Analytics' }
    ];


    const nodesGroup = new THREE.Group();
    const linksGroup = new THREE.Group();

    nodeData.forEach((data, index) => {
      // 1. Core glowing geometry (Rounded Octahedron / Box)
      const geom = new THREE.OctahedronGeometry(2.2, 2);
      const mat = new THREE.MeshStandardMaterial({
        color: data.color,
        emissive: data.color,
        emissiveIntensity: 0.6,
        roughness: 0.2,
        metalness: 0.8,
        wireframe: false
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(data.x, data.y, data.z);
      mesh.userData = data;

      // 2. Wireframe shell for futuristic sci-fi effect
      const wireGeom = new THREE.OctahedronGeometry(2.8, 1);
      const wireMat = new THREE.MeshBasicMaterial({
        color: data.color,
        wireframe: true,
        transparent: true,
        opacity: 0.45
      });
      const wireMesh = new THREE.Mesh(wireGeom, wireMat);
      mesh.add(wireMesh);

      // 3. Ring orbit
      const ringGeom = new THREE.TorusGeometry(3.6, 0.08, 8, 32);
      const ringMat = new THREE.MeshBasicMaterial({
        color: data.color,
        transparent: true,
        opacity: 0.5
      });
      const ringMesh = new THREE.Mesh(ringGeom, ringMat);
      ringMesh.rotation.x = Math.PI / 2;
      mesh.add(ringMesh);

      nodesGroup.add(mesh);
      this.pipelineNodes.push(mesh);

      // Connection Tubes between consecutive nodes
      if (index > 0) {
        const prev = nodeData[index - 1];
        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(prev.x, prev.y, prev.z),
          new THREE.Vector3((prev.x + data.x) / 2, (prev.y + data.y) / 2 + 2, (prev.z + data.z) / 2),
          new THREE.Vector3(data.x, data.y, data.z)
        ]);

        const tubeGeom = new THREE.TubeGeometry(curve, 32, 0.15, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({
          color: 0x00f2fe,
          transparent: true,
          opacity: 0.4
        });
        const tubeMesh = new THREE.Mesh(tubeGeom, tubeMat);
        linksGroup.add(tubeMesh);
        this.pipelineLinks.push({ curve, tubeMesh, progress: Math.random() });
      }
    });

    this.scene.add(nodesGroup);
    this.scene.add(linksGroup);
    this.dynamicGroup = nodesGroup;
    this.dynamicLinksGroup = linksGroup;

    // Moving data packets along curves
    this.createDataPackets();
  }

  createDataPackets() {
    this.packetMeshes = [];
    const packetGeom = new THREE.SphereGeometry(0.4, 16, 16);
    const packetMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      blending: THREE.AdditiveBlending
    });

    for (let i = 0; i < 18; i++) {
      const packet = new THREE.Mesh(packetGeom, packetMat);
      packet.userData = {
        linkIndex: i % Math.max(1, this.pipelineLinks.length),
        t: Math.random(),
        speed: 0.003 + Math.random() * 0.004
      };
      this.scene.add(packet);
      this.packetMeshes.push(packet);
    }
  }

  buildDecisionSurfaceScene() {
    this.clearDynamicObjects();

    const group = new THREE.Group();

    // 1. 3D Clusters (Class 0 = Cyan, Class 1 = Coral)
    const count = 280;
    const geom0 = new THREE.SphereGeometry(0.35, 12, 12);
    const mat0 = new THREE.MeshStandardMaterial({ color: 0x00f2fe, emissive: 0x00f2fe, emissiveIntensity: 0.5 });
    
    const mat1 = new THREE.MeshStandardMaterial({ color: 0xff477e, emissive: 0xff477e, emissiveIntensity: 0.5 });

    for (let i = 0; i < count; i++) {
      const isClass0 = Math.random() > 0.4;
      const mesh = new THREE.Mesh(geom0, isClass0 ? mat0 : mat1);
      
      const angle = Math.random() * Math.PI * 2;
      const r = isClass0 ? 3 + Math.random() * 8 : 9 + Math.random() * 8;
      const x = Math.cos(angle) * r + (Math.random() - 0.5) * 3;
      const z = Math.sin(angle) * r + (Math.random() - 0.5) * 3;
      const y = (Math.random() - 0.5) * 6;

      mesh.position.set(x, y, z);
      group.add(mesh);
    }

    // 2. 3D Decision Hyperplane Mesh
    const planeGeom = new THREE.CylinderGeometry(8.5, 8.5, 0.15, 32, 1, true);
    const planeMat = new THREE.MeshStandardMaterial({
      color: 0xa855f7,
      emissive: 0xa855f7,
      emissiveIntensity: 0.4,
      transparent: true,
      opacity: 0.4,
      side: THREE.DoubleSide,
      wireframe: true
    });
    const decisionBoundary = new THREE.Mesh(planeGeom, planeMat);
    decisionBoundary.rotation.x = Math.PI / 2;
    group.add(decisionBoundary);
    this.decisionBoundary = decisionBoundary;

    this.scene.add(group);
    this.dynamicGroup = group;
  }

  buildNeuralSpaceScene() {
    this.clearDynamicObjects();

    const group = new THREE.Group();
    const layers = [4, 7, 7, 3];
    const layerDist = 9;
    const nodeMeshes = [];

    layers.forEach((nodesInLayer, layerIdx) => {
      const x = (layerIdx - (layers.length - 1) / 2) * layerDist;
      const layerNodes = [];

      for (let i = 0; i < nodesInLayer; i++) {
        const y = (i - (nodesInLayer - 1) / 2) * 3.5;
        const geom = new THREE.IcosahedronGeometry(1.0, 2);
        const mat = new THREE.MeshStandardMaterial({
          color: layerIdx === 0 ? 0x00f2fe : (layerIdx === layers.length - 1 ? 0x10b981 : 0xa855f7),
          emissive: 0x241442,
          roughness: 0.3
        });
        const node = new THREE.Mesh(geom, mat);
        node.position.set(x, y, 0);
        group.add(node);
        layerNodes.push(node);
      }
      nodeMeshes.push(layerNodes);
    });

    // Synapses between layers
    for (let l = 0; l < nodeMeshes.length - 1; l++) {
      const srcLayer = nodeMeshes[l];
      const dstLayer = nodeMeshes[l + 1];

      srcLayer.forEach(src => {
        dstLayer.forEach(dst => {
          const lineGeom = new THREE.BufferGeometry().setFromPoints([
            src.position,
            dst.position
          ]);
          const lineMat = new THREE.LineBasicMaterial({
            color: 0x3b82f6,
            transparent: true,
            opacity: 0.25
          });
          const line = new THREE.Line(lineGeom, lineMat);
          group.add(line);
        });
      });
    }

    this.scene.add(group);
    this.dynamicGroup = group;
  }

  clearDynamicObjects() {
    if (this.dynamicGroup) {
      this.scene.remove(this.dynamicGroup);
      this.dynamicGroup = null;
    }
    if (this.dynamicLinksGroup) {
      this.scene.remove(this.dynamicLinksGroup);
      this.dynamicLinksGroup = null;
    }
    if (this.packetMeshes) {
      this.packetMeshes.forEach(p => this.scene.remove(p));
      this.packetMeshes = [];
    }
    this.pipelineNodes = [];
    this.pipelineLinks = [];
  }

  setMode(mode) {
    this.currentMode = mode;
    if (mode === 'pipeline') {
      this.buildPipelineScene();
    } else if (mode === 'decision') {
      this.buildDecisionSurfaceScene();
    } else if (mode === 'neural') {
      this.buildNeuralSpaceScene();
    }
  }

  addEventListeners() {
    window.addEventListener('resize', () => {
      if (!this.container) return;
      const width = this.container.clientWidth;
      const height = this.container.clientHeight;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    });

    this.container.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      this.prevMousePos = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isDragging) {
        const deltaX = e.clientX - this.prevMousePos.x;
        const deltaY = e.clientY - this.prevMousePos.y;
        this.targetRotationY += deltaX * 0.006;
        this.targetRotationX += deltaY * 0.006;
        this.prevMousePos = { x: e.clientX, y: e.clientY };
      }
    });

    // Touch support for mobile devices
    this.container.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    });

    window.addEventListener('touchend', () => {
      this.isDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
      if (this.isDragging && e.touches.length === 1) {
        const deltaX = e.touches[0].clientX - this.prevMousePos.x;
        const deltaY = e.touches[0].clientY - this.prevMousePos.y;
        this.targetRotationY += deltaX * 0.008;
        this.targetRotationX += deltaY * 0.008;
        this.prevMousePos = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const delta = this.clock.getDelta();
    const time = this.clock.getElapsedTime();

    // 1. Rotate Background Particles
    if (this.particles) {
      this.particles.rotation.y = time * 0.02;
    }

    // 2. Smooth scene rotation from drag
    if (this.dynamicGroup) {
      this.dynamicGroup.rotation.y += (this.targetRotationY - this.dynamicGroup.rotation.y) * 0.08;
      this.dynamicGroup.rotation.x += (this.targetRotationX - this.dynamicGroup.rotation.x) * 0.08;
      
      // Auto slight drift
      if (!this.isDragging) {
        this.targetRotationY += 0.002;
      }
    }

    // 3. Pipeline Specific Animations
    if (this.currentMode === 'pipeline') {
      this.pipelineNodes.forEach((node, i) => {
        node.rotation.y += 0.015;
        node.rotation.z += 0.01;
        node.position.y += Math.sin(time * 2 + i) * 0.008;
      });

      // Animate data packets
      if (this.packetMeshes && this.pipelineLinks.length > 0) {
        this.packetMeshes.forEach(pkt => {
          pkt.userData.t += pkt.userData.speed;
          if (pkt.userData.t > 1) {
            pkt.userData.t = 0;
            pkt.userData.linkIndex = (pkt.userData.linkIndex + 1) % this.pipelineLinks.length;
          }
          const link = this.pipelineLinks[pkt.userData.linkIndex];
          if (link && link.curve) {
            const point = link.curve.getPoint(pkt.userData.t);
            pkt.position.copy(point);
          }
        });
      }
    } else if (this.currentMode === 'decision' && this.decisionBoundary) {
      this.decisionBoundary.rotation.z = time * 0.2;
      this.decisionBoundary.position.y = Math.sin(time) * 0.8;
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// Global initialization
window.Pipeline3DScene = Pipeline3DScene;
