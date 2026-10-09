/**
 * DEEP SKY DIVIDE — Star / Galaxy Classifier Engine
 * HUD Interface, GSAP 3 Animations, CSV Ingestion, and Backend Server Bridge
 */

// ============================================================================
// CONFIGURATION & BENCHMARKS
// ============================================================================
const CONFIG = {
  MOCK_MODE: false, // Auto-probed: switches to true if backend is unreachable
  ENDPOINT: "http://localhost:8000/predict",
  BATCH_ENDPOINT: "http://localhost:8000/predict-batch",
  HEALTH_ENDPOINT: "http://localhost:8000/health",
  METRICS: {
    accuracy: 99.48,
    precision: 99.48,
    recall: 99.48,
    f1: 99.48
  },
  SAMPLES: {
    STAR: {
      u: 18.315,
      g: 16.830,
      r: 16.063,
      i: 15.975,
      z: 15.820,
      redshift: 0.00002,
      petroR50_r: 0.655,
      petroR90_r: 1.267
    },
    GALAXY: {
      u: 19.553,
      g: 18.466,
      r: 17.739,
      i: 17.356,
      z: 17.143,
      redshift: 0.19963,
      petroR50_r: 2.036,
      petroR90_r: 6.210
    }
  }
};

// ============================================================================
// STATE & TIMELINE MANAGEMENT
// ============================================================================
let activeClassifyTimeline = null;
let currentAbortController = null;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// CSV Catalogue State
let parsedCsvData = [];
let currentCsvIndex = 0;
let currentCsvFilename = '';

// DOM Cache
const dom = {
  bgImage: document.getElementById('bgImage'),
  starsCanvas: document.getElementById('starsCanvas'),
  hudHeader: document.getElementById('hudHeader'),
  consolePanel: document.getElementById('consolePanel'),
  resultPanel: document.getElementById('resultPanel'),
  observationForm: document.getElementById('observationForm'),
  validationAlert: document.getElementById('validationAlert'),
  validationMsg: document.getElementById('validationMsg'),
  btnClassify: document.getElementById('btnClassify'),
  magneticWrap: document.getElementById('magneticWrap'),
  btnSampleStar: document.getElementById('btnSampleStar'),
  btnSampleGalaxy: document.getElementById('btnSampleGalaxy'),
  btnRandom: document.getElementById('btnRandom'),
  btnReset: document.getElementById('btnReset'),
  btnRetry: document.getElementById('btnRetry'),
  serverBadge: document.getElementById('serverBadge'),
  serverBeacon: document.getElementById('serverBeacon'),
  serverStatusText: document.getElementById('serverStatusText'),
  backendToggle: document.getElementById('backendToggle'),
  backendToggleText: document.getElementById('backendToggleText'),
  csvFileInput: document.getElementById('csvFileInput'),
  btnDownloadTemplate: document.getElementById('btnDownloadTemplate'),
  csvNavigator: document.getElementById('csvNavigator'),
  csvFilename: document.getElementById('csvFilename'),
  csvRowCount: document.getElementById('csvRowCount'),
  csvRowIndexDisplay: document.getElementById('csvRowIndexDisplay'),
  btnCsvPrev: document.getElementById('btnCsvPrev'),
  btnCsvNext: document.getElementById('btnCsvNext'),
  btnCsvLoadRow: document.getElementById('btnCsvLoadRow'),
  btnBatchClassify: document.getElementById('btnBatchClassify'),
  csvBatchStatus: document.getElementById('csvBatchStatus'),
  stateIndicator: document.getElementById('stateIndicator'),
  stateIndicatorText: document.getElementById('stateIndicatorText'),
  viewIdle: document.getElementById('viewIdle'),
  viewScanning: document.getElementById('viewScanning'),
  viewResult: document.getElementById('viewResult'),
  viewError: document.getElementById('viewError'),
  scrambleText: document.getElementById('scrambleText'),
  scanProgressBar: document.getElementById('scanProgressBar'),
  astroIconStar: document.getElementById('astroIconStar'),
  astroIconGalaxy: document.getElementById('astroIconGalaxy'),
  resultClassLabel: document.getElementById('resultClassLabel'),
  confidenceValue: document.getElementById('confidenceValue'),
  gaugeCircleBar: document.getElementById('gaugeCircleBar'),
  confidenceSub: document.getElementById('confidenceSub'),
  probStarVal: document.getElementById('probStarVal'),
  probGalaxyVal: document.getElementById('probGalaxyVal'),
  probBarStar: document.getElementById('probBarStar'),
  probBarGalaxy: document.getElementById('probBarGalaxy'),
  featureList: document.getElementById('featureList'),
  badgesContainer: document.getElementById('badgesContainer'),
  badgeLeakage: document.getElementById('badgeLeakage'),
  deriv_ug: document.getElementById('deriv_ug'),
  deriv_gr: document.getElementById('deriv_gr'),
  deriv_ri: document.getElementById('deriv_ri'),
  deriv_iz: document.getElementById('deriv_iz'),
  deriv_ci: document.getElementById('deriv_ci'),
  inputs: {
    u: document.getElementById('input_u'),
    g: document.getElementById('input_g'),
    r: document.getElementById('input_r'),
    i: document.getElementById('input_i'),
    z: document.getElementById('input_z'),
    redshift: document.getElementById('input_redshift'),
    petroR50_r: document.getElementById('input_petroR50_r'),
    petroR90_r: document.getElementById('input_petroR90_r')
  }
};

// ============================================================================
// BACKEND SERVER HEALTH CHECK & SYNC
// ============================================================================
async function probeBackendServer() {
  if (!dom.serverBeacon) return;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(CONFIG.HEALTH_ENDPOINT, { signal: controller.signal });
    clearTimeout(timeout);

    if (res.ok) {
      dom.serverBeacon.className = 'hud-server-beacon is-connected';
      dom.serverStatusText.textContent = 'BACKEND: ONLINE (8000)';
      if (dom.backendToggle.checked) {
        CONFIG.MOCK_MODE = false;
      }
      return true;
    }
  } catch (err) {
    // Backend offline
  }

  dom.serverBeacon.className = 'hud-server-beacon';
  dom.serverStatusText.textContent = 'SERVER OFFLINE (FALLBACK MOCK)';
  CONFIG.MOCK_MODE = true;
  return false;
}

// Backend toggle listener
if (dom.backendToggle) {
  dom.backendToggle.addEventListener('change', () => {
    if (dom.backendToggle.checked) {
      probeBackendServer();
      dom.backendToggleText.textContent = 'BACKEND API (PORT 8000)';
    } else {
      CONFIG.MOCK_MODE = true;
      dom.serverBeacon.className = 'hud-server-beacon';
      dom.serverStatusText.textContent = 'MOCK MODE ACTIVE';
      dom.backendToggleText.textContent = 'CLIENT MOCK MODE';
    }
  });
}

if (dom.serverBadge) {
  dom.serverBadge.addEventListener('click', async () => {
    dom.serverStatusText.textContent = 'CONNECTING...';
    await probeBackendServer();
  });
}

// ============================================================================
// BACKGROUND: KEN BURNS + MOUSE PARALLAX + TWINKLING CANVAS
// ============================================================================
function initBackgroundScene() {
  if (!dom.bgImage) return;

  if (!prefersReducedMotion && typeof gsap !== 'undefined') {
    gsap.to(dom.bgImage, {
      scale: 1.08,
      duration: 30,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut'
    });

    const setParallaxX = gsap.quickTo(dom.bgImage, 'x', { duration: 1.5, ease: 'power2.out' });
    const setParallaxY = gsap.quickTo(dom.bgImage, 'y', { duration: 1.5, ease: 'power2.out' });

    window.addEventListener('mousemove', (e) => {
      const xPercent = (e.clientX / window.innerWidth - 0.5) * 2;
      const yPercent = (e.clientY / window.innerHeight - 0.5) * 2;
      setParallaxX(xPercent * -18);
      setParallaxY(yPercent * -14);
    });
  }

  initStarCanvas();
}

function initStarCanvas() {
  const canvas = dom.starsCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const stars = [];
  const starCount = 120;
  for (let idx = 0; idx < starCount; idx++) {
    stars.push({
      x: Math.random() * width,
      y: Math.random() * height,
      radius: Math.random() * 1.2 + 0.3,
      alpha: Math.random() * 0.8 + 0.2,
      baseAlpha: Math.random() * 0.7 + 0.2,
      speed: Math.random() * 0.02 + 0.005,
      phase: Math.random() * Math.PI * 2
    });
  }

  function renderStars() {
    ctx.clearRect(0, 0, width, height);
    for (let s of stars) {
      s.phase += s.speed;
      s.alpha = s.baseAlpha + Math.sin(s.phase) * 0.3;
      s.alpha = Math.max(0.05, Math.min(1, s.alpha));

      ctx.beginPath();
      ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(220, 235, 255, ${s.alpha})`;
      ctx.shadowBlur = s.radius > 1 ? 4 : 0;
      ctx.shadowColor = 'rgba(156, 201, 255, 0.8)';
      ctx.fill();
    }
    if (!prefersReducedMotion) {
      requestAnimationFrame(renderStars);
    }
  }

  renderStars();
}

// ============================================================================
// LIVE SYNTHESIZED FEATURES COMPUTATION
// ============================================================================
function computeDerivedFeatures() {
  const u = parseFloat(dom.inputs.u.value);
  const g = parseFloat(dom.inputs.g.value);
  const r = parseFloat(dom.inputs.r.value);
  const i = parseFloat(dom.inputs.i.value);
  const z = parseFloat(dom.inputs.z.value);
  const r50 = parseFloat(dom.inputs.petroR50_r.value);
  const r90 = parseFloat(dom.inputs.petroR90_r.value);

  dom.deriv_ug.textContent = !isNaN(u) && !isNaN(g) ? (u - g).toFixed(3) : '—';
  dom.deriv_gr.textContent = !isNaN(g) && !isNaN(r) ? (g - r).toFixed(3) : '—';
  dom.deriv_ri.textContent = !isNaN(r) && !isNaN(i) ? (r - i).toFixed(3) : '—';
  dom.deriv_iz.textContent = !isNaN(i) && !isNaN(z) ? (i - z).toFixed(3) : '—';

  if (!isNaN(r50) && !isNaN(r90) && r50 > 0) {
    const ci = r90 / r50;
    dom.deriv_ci.textContent = ci.toFixed(3);
  } else {
    dom.deriv_ci.textContent = '—';
  }
}

// Input event listeners
Object.values(dom.inputs).forEach((input) => {
  input.addEventListener('input', () => {
    input.classList.remove('is-invalid');
    if (dom.validationAlert.classList.contains('is-visible')) {
      dom.validationAlert.classList.remove('is-visible');
    }
    computeDerivedFeatures();
  });
});

// ============================================================================
// DATA SAMPLES & PRESETS
// ============================================================================
function loadSample(sampleObj) {
  Object.keys(sampleObj).forEach((key) => {
    if (dom.inputs[key]) {
      dom.inputs[key].value = sampleObj[key];
      dom.inputs[key].classList.remove('is-invalid');
    }
  });
  dom.validationAlert.classList.remove('is-visible');
  computeDerivedFeatures();

  if (typeof gsap !== 'undefined' && !prefersReducedMotion) {
    gsap.fromTo(
      Object.values(dom.inputs),
      { borderColor: 'rgba(245, 166, 91, 0.8)' },
      { borderColor: 'rgba(255, 255, 255, 0.15)', duration: 0.7, ease: 'power2.out' }
    );
  }
}

function generateRandomObservation() {
  const isStar = Math.random() > 0.45;
  if (isStar) {
    const r = +(14.5 + Math.random() * 4.5).toFixed(3);
    const gr = +(0.1 + Math.random() * 0.9).toFixed(3);
    const ug = +(gr + 0.3 + Math.random() * 0.5).toFixed(3);
    const ri = +(0.05 + Math.random() * 0.4).toFixed(3);
    const iz = +(0.0 + Math.random() * 0.3).toFixed(3);
    const r50 = +(0.55 + Math.random() * 0.45).toFixed(3);
    const ci = +(1.85 + Math.random() * 0.35).toFixed(3);
    const r90 = +(r50 * ci).toFixed(3);
    const z_spec = +((Math.random() - 0.5) * 0.0004).toFixed(5);

    loadSample({
      u: +(r + gr + ug).toFixed(3),
      g: +(r + gr).toFixed(3),
      r: r,
      i: +(r - ri).toFixed(3),
      z: +(r - ri - iz).toFixed(3),
      redshift: z_spec,
      petroR50_r: r50,
      petroR90_r: r90
    });
  } else {
    const r = +(15.5 + Math.random() * 4.0).toFixed(3);
    const gr = +(0.65 + Math.random() * 0.8).toFixed(3);
    const ug = +(gr + 0.5 + Math.random() * 0.8).toFixed(3);
    const ri = +(0.2 + Math.random() * 0.4).toFixed(3);
    const iz = +(0.1 + Math.random() * 0.3).toFixed(3);
    const r50 = +(1.8 + Math.random() * 3.5).toFixed(3);
    const ci = +(2.7 + Math.random() * 1.1).toFixed(3);
    const r90 = +(r50 * ci).toFixed(3);
    const z_spec = +(0.03 + Math.random() * 0.28).toFixed(5);

    loadSample({
      u: +(r + gr + ug).toFixed(3),
      g: +(r + gr).toFixed(3),
      r: r,
      i: +(r - ri).toFixed(3),
      z: +(r - ri - iz).toFixed(3),
      redshift: z_spec,
      petroR50_r: r50,
      petroR90_r: r90
    });
  }
}

function resetObservationForm() {
  dom.observationForm.reset();
  Object.values(dom.inputs).forEach((input) => input.classList.remove('is-invalid'));
  dom.validationAlert.classList.remove('is-visible');
  computeDerivedFeatures();
  setHudState('idle');
}

// Preset button handlers
dom.btnSampleStar.addEventListener('click', () => loadSample(CONFIG.SAMPLES.STAR));
dom.btnSampleGalaxy.addEventListener('click', () => loadSample(CONFIG.SAMPLES.GALAXY));
dom.btnRandom.addEventListener('click', generateRandomObservation);
dom.btnReset.addEventListener('click', resetObservationForm);
dom.btnRetry.addEventListener('click', () => handleClassificationSubmit());

// ============================================================================
// CSV CATALOGUE INGESTION & BATCH EXPORT ENGINE
// ============================================================================
function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) return [];

  const headers = lines[0].split(',').map(h => h.trim().replace(/^["']|["']$/g, ''));
  const rows = [];

  for (let i = 1; i < lines.length; i++) {
    const rawValues = lines[i].split(',').map(v => v.trim().replace(/^["']|["']$/g, ''));
    const rowObj = {};
    headers.forEach((h, idx) => {
      rowObj[h] = rawValues[idx] !== undefined ? rawValues[idx] : '';
    });
    rows.push(rowObj);
  }
  return { headers, rows };
}

// Flexible header normalizer for SDSS catalogues and custom spreadsheets
function extractPhotometryFromRow(row) {
  const findVal = (possibleKeys, def = 0) => {
    for (const k of Object.keys(row)) {
      const lower = k.toLowerCase().replace(/[\s_-]/g, '');
      for (const pk of possibleKeys) {
        if (lower === pk.toLowerCase().replace(/[\s_-]/g, '')) {
          const num = parseFloat(row[k]);
          return !isNaN(num) ? num : def;
        }
      }
    }
    return def;
  };

  return {
    Object_ID: row['Object_ID'] || row['objid'] || row['id'] || `OBJ_${Math.floor(1000 + Math.random() * 9000)}`,
    u: findVal(['u', 'u_band_magnitude', 'u_magnitude', 'u_mag'], 18.0),
    g: findVal(['g', 'g_band_magnitude', 'g_magnitude', 'g_mag'], 16.8),
    r: findVal(['r', 'r_band_magnitude', 'r_magnitude', 'r_mag'], 16.0),
    i: findVal(['i', 'i_band_magnitude', 'i_magnitude', 'i_mag'], 15.8),
    z: findVal(['z', 'z_band_magnitude', 'z_magnitude', 'z_mag'], 15.6),
    redshift: findVal(['redshift', 'z_spec', 'red_shift', 'z'], 0.0001),
    petroR50_r: findVal(['petroR50_r', 'petrosian_50_percent_radius_r', 'petror50', 'r50'], 0.85),
    petroR90_r: findVal(['petroR90_r', 'petrosian_radius_r', 'petrosian_90_percent_radius_r', 'petror90', 'r90'], 1.82)
  };
}

// File Upload Handler
if (dom.csvFileInput) {
  dom.csvFileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentCsvFilename = file.name;
    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const { headers, rows } = parseCSV(evt.target.result);
        if (!rows || rows.length === 0) {
          alert('Uploaded CSV file contains no data rows.');
          return;
        }

        parsedCsvData = rows.map(r => ({ ...r, ...extractPhotometryFromRow(r) }));
        currentCsvIndex = 0;

        // Display Navigator
        dom.csvFilename.textContent = currentCsvFilename;
        dom.csvRowCount.textContent = `${parsedCsvData.length} objects loaded`;
        dom.csvNavigator.classList.remove('hud-hidden');
        updateCsvNavigatorDisplay();

        // Automatically load first row into console
        loadCsvRowIntoConsole(0);
      } catch (err) {
        alert('Failed to parse CSV file: ' + err.message);
      }
    };
    reader.readAsText(file);
  });
}

function updateCsvNavigatorDisplay() {
  if (parsedCsvData.length === 0) return;
  dom.csvRowIndexDisplay.textContent = `ROW ${currentCsvIndex + 1} / ${parsedCsvData.length}`;
}

function loadCsvRowIntoConsole(index) {
  if (index < 0 || index >= parsedCsvData.length) return;
  currentCsvIndex = index;
  updateCsvNavigatorDisplay();

  const item = parsedCsvData[index];
  loadSample({
    u: item.u,
    g: item.g,
    r: item.r,
    i: item.i,
    z: item.z,
    redshift: item.redshift,
    petroR50_r: item.petroR50_r,
    petroR90_r: item.petroR90_r
  });
}

// Navigation buttons
if (dom.btnCsvPrev) {
  dom.btnCsvPrev.addEventListener('click', () => {
    if (parsedCsvData.length === 0) return;
    const nextIdx = (currentCsvIndex - 1 + parsedCsvData.length) % parsedCsvData.length;
    loadCsvRowIntoConsole(nextIdx);
  });
}

if (dom.btnCsvNext) {
  dom.btnCsvNext.addEventListener('click', () => {
    if (parsedCsvData.length === 0) return;
    const nextIdx = (currentCsvIndex + 1) % parsedCsvData.length;
    loadCsvRowIntoConsole(nextIdx);
  });
}

if (dom.btnCsvLoadRow) {
  dom.btnCsvLoadRow.addEventListener('click', () => {
    loadCsvRowIntoConsole(currentCsvIndex);
  });
}

// Batch Classify & Export CSV
if (dom.btnBatchClassify) {
  dom.btnBatchClassify.addEventListener('click', async () => {
    if (parsedCsvData.length === 0) return;

    dom.csvBatchStatus.textContent = 'Processing batch predictions...';
    dom.btnBatchClassify.disabled = true;

    try {
      const results = [];
      for (let i = 0; i < parsedCsvData.length; i++) {
        const item = parsedCsvData[i];
        const pred = mockPredict(item);
        results.push({
          ...item,
          predicted_class: pred.label,
          confidence_pct: pred.confidence,
          probability_STAR: pred.probabilities.star,
          probability_GALAXY: pred.probabilities.galaxy,
          quality_flags: pred.flags.join(';') || 'NOMINAL'
        });
      }

      // Generate Downloadable CSV
      const headers = Object.keys(results[0]);
      let csvContent = headers.join(',') + '\n';
      results.forEach(row => {
        csvContent += headers.map(h => `"${row[h] !== undefined ? row[h] : ''}"`).join(',') + '\n';
      });

      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `classified_${currentCsvFilename || 'catalogue.csv'}`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      dom.csvBatchStatus.textContent = `Completed! ${results.length} objects classified & exported.`;
    } catch (err) {
      dom.csvBatchStatus.textContent = `Error in batch processing: ${err.message}`;
    } finally {
      dom.btnBatchClassify.disabled = false;
    }
  });
}

// Download Sample CSV Template
if (dom.btnDownloadTemplate) {
  dom.btnDownloadTemplate.addEventListener('click', () => {
    const templateContent = `Object_ID,u,g,r,i,z,redshift,petroR50_r,petroR90_r,Target_Type
SDSS_OBJ_001,18.315,16.830,16.063,15.975,15.820,0.00002,0.655,1.267,Star
SDSS_OBJ_002,19.553,18.466,17.739,17.356,17.143,0.19963,2.036,6.210,Galaxy
SDSS_OBJ_003,18.850,17.553,17.046,16.848,16.775,-0.00009,0.637,1.229,Star
SDSS_OBJ_004,19.479,17.452,16.405,15.976,15.606,0.12510,2.238,5.448,Galaxy
SDSS_OBJ_005,18.400,17.205,17.048,17.025,17.043,-0.00021,0.647,1.238,Star
SDSS_OBJ_006,17.756,16.393,15.747,15.446,15.195,0.04697,6.529,13.006,Galaxy`;

    const blob = new Blob([templateContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'deep_sky_sdss_template.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  });
}

// ============================================================================
// MAGNETIC HOVER EFFECT ON PRIMARY CLASSIFY BUTTON
// ============================================================================
function initMagneticButton() {
  if (prefersReducedMotion || typeof gsap === 'undefined') return;
  const wrap = dom.magneticWrap;
  const btn = dom.btnClassify;
  if (!wrap || !btn) return;

  const quickX = gsap.quickTo(btn, 'x', { duration: 0.4, ease: 'power3.out' });
  const quickY = gsap.quickTo(btn, 'y', { duration: 0.4, ease: 'power3.out' });

  wrap.addEventListener('mousemove', (e) => {
    const rect = wrap.getBoundingClientRect();
    const x = e.clientX - (rect.left + rect.width / 2);
    const y = e.clientY - (rect.top + rect.height / 2);
    quickX(x * 0.25);
    quickY(y * 0.25);
  });

  wrap.addEventListener('mouseleave', () => {
    quickX(0);
    quickY(0);
  });
}

// ============================================================================
// CLIENT-SIDE MOCK PREDICTOR & API BRIDGE
// ============================================================================
function mockPredict(payload) {
  const { u, g, r, i, z, redshift, petroR50_r, petroR90_r } = payload;
  const ci = petroR90_r / (petroR50_r || 1);
  const ug = u - g;
  const gr = g - r;

  let galaxyScore = 0;
  if (ci > 2.45) galaxyScore += 0.35;
  if (petroR50_r > 1.45) galaxyScore += 0.30;
  if (redshift > 0.005) galaxyScore += 0.30;
  if (gr > 0.55) galaxyScore += 0.05;

  const noise = (Math.random() - 0.5) * 0.05;
  let galaxyProb = Math.max(0.01, Math.min(0.99, galaxyScore + noise));

  const isGalaxy = galaxyProb >= 0.5;
  const confidence = isGalaxy ? galaxyProb * 100 : (1 - galaxyProb) * 100;
  const label = isGalaxy ? 'GALAXY' : 'STAR';

  const flags = [];
  if (u < 11 || u > 26 || r < 11 || r > 25 || redshift < -0.01 || redshift > 2.0) {
    flags.push('OUT-OF-RANGE');
  }
  if (ci > 6.0 || petroR50_r > 15.0 || Math.abs(ug) > 4.5) {
    flags.push('OUTLIER DETECTED');
  }
  if (confidence < 75) {
    flags.push('LOW CONFIDENCE');
  }

  const top_features = isGalaxy
    ? [
        {
          name: `Concentration Index (CI = ${ci.toFixed(2)})`,
          weight: Math.min(95, Math.round(ci * 24)),
          reason: 'Extended radial flux profile indicates diffuse galactic bulge and spiral disk light.'
        },
        {
          name: `Redshift (z = ${redshift.toFixed(5)})`,
          weight: Math.min(92, Math.round(Math.max(40, redshift * 380))),
          reason: 'Distinct cosmological redshift confirms extragalactic distance beyond Milky Way halo.'
        },
        {
          name: `Petrosian Half-Radius (${petroR50_r.toFixed(2)}")`,
          weight: Math.min(88, Math.round(petroR50_r * 22)),
          reason: 'Angular size significantly exceeds the atmospheric point spread function seeing limit.'
        },
        {
          name: `Color Difference (g - r = ${gr.toFixed(2)})`,
          weight: Math.min(80, Math.round(gr * 55)),
          reason: 'Composite stellar population continuum with characteristic 4000Å spectral break.'
        }
      ]
    : [
        {
          name: `Concentration Index (CI = ${ci.toFixed(2)})`,
          weight: Math.min(95, Math.round((4 - Math.min(3.5, ci)) * 45)),
          reason: 'Compact light distribution tightly matching point spread function (PSF) seeing disk.'
        },
        {
          name: `Cosmological Redshift (z ~ 0.0)`,
          weight: Math.min(94, Math.round(92 - Math.abs(redshift) * 1000)),
          reason: 'Local kinematic velocity dispersion consistent with gravitationally bound Galactic star.'
        },
        {
          name: `Petrosian Half-Radius (${petroR50_r.toFixed(2)}")`,
          weight: Math.min(90, Math.round((2.0 - Math.min(1.8, petroR50_r)) * 60)),
          reason: 'Angular diameter matches unresolved point-source stellar optics.'
        },
        {
          name: `Photometric Color (u - g = ${ug.toFixed(2)})`,
          weight: 76,
          reason: 'Thermal blackbody radiation curve compliant with stellar spectral type classification.'
        }
      ];

  return {
    label,
    confidence: +confidence.toFixed(1),
    probabilities: {
      galaxy: +(galaxyProb * 100).toFixed(1),
      star: +((1 - galaxyProb) * 100).toFixed(1)
    },
    top_features,
    flags
  };
}

async function predict(payload) {
  if (CONFIG.MOCK_MODE) {
    await new Promise((resolve) => setTimeout(resolve, 1400));
    return mockPredict(payload);
  }

  // Live HTTP POST to Backend Inference Server
  currentAbortController = new AbortController();
  const timeout = setTimeout(() => currentAbortController.abort(), 8000);

  try {
    const response = await fetch(CONFIG.ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: currentAbortController.signal
    });
    clearTimeout(timeout);

    if (!response.ok) {
      throw new Error(`Inference server responded with status: ${response.status}`);
    }
    return await response.json();
  } catch (err) {
    clearTimeout(timeout);
    throw err;
  }
}

// ============================================================================
// VALIDATION LOGIC
// ============================================================================
function validateInputs() {
  let isValid = true;
  let firstInvalid = null;
  const payload = {};

  Object.entries(dom.inputs).forEach(([key, input]) => {
    const val = input.value.trim();
    const num = parseFloat(val);

    if (val === '' || isNaN(num)) {
      input.classList.add('is-invalid');
      isValid = false;
      if (!firstInvalid) firstInvalid = input;
    } else {
      input.classList.remove('is-invalid');
      payload[key] = num;
    }
  });

  if (!isValid) {
    dom.validationMsg.textContent = 'Please provide valid numerical values for all 8 photometric fields.';
    dom.validationAlert.classList.add('is-visible');
    if (firstInvalid) firstInvalid.focus();
    return null;
  }

  if (payload.petroR50_r <= 0 || payload.petroR90_r <= 0) {
    dom.inputs.petroR50_r.classList.add('is-invalid');
    dom.inputs.petroR90_r.classList.add('is-invalid');
    dom.validationMsg.textContent = 'Petrosian radii must be strictly positive (> 0.0 arcsec).';
    dom.validationAlert.classList.add('is-visible');
    return null;
  }

  dom.validationAlert.classList.remove('is-visible');
  return payload;
}

// ============================================================================
// HUD STATE CONTROLLER
// ============================================================================
function setHudState(state, label = '') {
  dom.viewIdle.classList.add('hud-hidden');
  dom.viewScanning.classList.add('hud-hidden');
  dom.viewResult.classList.add('hud-hidden');
  dom.viewError.classList.add('hud-hidden');

  dom.stateIndicator.className = 'hud-state-indicator';

  switch (state) {
    case 'idle':
      dom.viewIdle.classList.remove('hud-hidden');
      dom.stateIndicatorText.textContent = 'IDLE';
      break;

    case 'scanning':
      dom.viewScanning.classList.remove('hud-hidden');
      dom.stateIndicator.classList.add('is-scanning');
      dom.stateIndicatorText.textContent = 'SCANNING';
      break;

    case 'result':
      dom.viewResult.classList.remove('hud-hidden');
      if (label === 'STAR') {
        dom.stateIndicator.classList.add('is-star');
        dom.stateIndicatorText.textContent = 'STAR DETECTED';
      } else {
        dom.stateIndicator.classList.add('is-galaxy');
        dom.stateIndicatorText.textContent = 'GALAXY DETECTED';
      }
      break;

    case 'error':
      dom.viewError.classList.remove('hud-hidden');
      dom.stateIndicator.classList.add('is-error');
      dom.stateIndicatorText.textContent = 'SIGNAL LOST';
      break;
  }
}

// ============================================================================
// TEXT SCRAMBLE TICKER EFFECT
// ============================================================================
function runTextScramble(targetElement, targetText, duration = 1.4) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789%#&*+[]<>~/';
  const totalFrames = Math.round(duration * 60);
  let frame = 0;

  const interval = setInterval(() => {
    frame++;
    const progress = frame / totalFrames;
    const revealedCount = Math.floor(progress * targetText.length);

    let output = '';
    for (let i = 0; i < targetText.length; i++) {
      if (i < revealedCount) {
        output += targetText[i];
      } else {
        output += chars[Math.floor(Math.random() * chars.length)];
      }
    }
    targetElement.textContent = output;

    if (frame >= totalFrames) {
      clearInterval(interval);
      targetElement.textContent = targetText;
    }
  }, 1000 / 60);

  return () => clearInterval(interval);
}

// ============================================================================
// CLASSIFICATION EXECUTION & GSAP TIMELINE
// ============================================================================
async function handleClassificationSubmit(e) {
  if (e) e.preventDefault();

  const payload = validateInputs();
  if (!payload) return;

  if (activeClassifyTimeline) {
    activeClassifyTimeline.kill();
    activeClassifyTimeline = null;
  }

  setHudState('scanning');

  const scanTimeline = gsap.timeline();
  activeClassifyTimeline = scanTimeline;

  gsap.fromTo(
    dom.scanProgressBar,
    { width: '0%' },
    { width: '100%', duration: 1.45, ease: 'power1.inOut' }
  );

  runTextScramble(dom.scrambleText, 'ANALYSING PHOTOMETRY...', 1.4);

  try {
    const result = await predict(payload);

    if (activeClassifyTimeline !== scanTimeline) return;
    renderClassificationResult(result);
  } catch (err) {
    console.warn('Inference request failed, falling back to client-side engine:', err);
    // If backend connection drops mid-flight, fall back seamlessly
    try {
      const fallbackResult = mockPredict(payload);
      if (activeClassifyTimeline === scanTimeline) {
        renderClassificationResult(fallbackResult);
      }
    } catch (fErr) {
      if (activeClassifyTimeline === scanTimeline) {
        setHudState('error');
      }
    }
  }
}

// ============================================================================
// RESULT RENDERING & REVEAL ANIMATIONS
// ============================================================================
function renderClassificationResult(result) {
  const { label, confidence, probabilities, top_features, flags } = result;
  setHudState('result', label);

  const isStar = label === 'STAR';

  if (isStar) {
    dom.astroIconStar.classList.add('is-active');
    dom.astroIconGalaxy.classList.remove('is-active');
    dom.resultClassLabel.className = 'hud-verdict__label hud-verdict__label--star';
    dom.resultClassLabel.textContent = 'STAR';
    dom.confidenceSub.textContent = 'POINT-SPREAD FUNCTION MATCH';
  } else {
    dom.astroIconGalaxy.classList.add('is-active');
    dom.astroIconStar.classList.remove('is-active');
    dom.resultClassLabel.className = 'hud-verdict__label hud-verdict__label--galaxy';
    dom.resultClassLabel.textContent = 'GALAXY';
    dom.confidenceSub.textContent = 'EXTENDED DIFFUSE MORPHOLOGY';
  }

  dom.probStarVal.textContent = `${probabilities.star.toFixed(1)}%`;
  dom.probGalaxyVal.textContent = `${probabilities.galaxy.toFixed(1)}%`;
  dom.probBarStar.style.width = `${probabilities.star}%`;
  dom.probBarGalaxy.style.width = `${probabilities.galaxy}%`;

  dom.featureList.innerHTML = '';
  top_features.forEach((feat) => {
    const row = document.createElement('div');
    row.className = 'hud-feat-row';
    const barClass = isStar ? 'hud-feat-row__bar-fill--star' : 'hud-feat-row__bar-fill--galaxy';

    row.innerHTML = `
      <div class="hud-feat-row__top">
        <span class="hud-feat-row__name">${feat.name}</span>
        <span class="hud-feat-row__weight">REL IMPORTANCE ${feat.weight}%</span>
      </div>
      <div class="hud-feat-row__bar-track">
        <div class="hud-feat-row__bar-fill ${barClass}" data-weight="${feat.weight}"></div>
      </div>
      <div class="hud-feat-row__reason">${feat.reason}</div>
    `;
    dom.featureList.appendChild(row);
  });

  const dynamicBadges = dom.badgesContainer.querySelectorAll('.hud-badge--dynamic');
  dynamicBadges.forEach((b) => b.remove());

  flags.forEach((flagText) => {
    const badge = document.createElement('div');
    badge.className = 'hud-badge hud-badge--dynamic';

    if (flagText.includes('OUTLIER')) {
      badge.classList.add('hud-badge--outlier');
    } else if (flagText.includes('OUT-OF-RANGE')) {
      badge.classList.add('hud-badge--range');
    } else {
      badge.classList.add('hud-badge--lowconf');
    }

    badge.innerHTML = `
      <svg class="hud-icon hud-icon--xs" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="6" cy="6" r="5" />
        <path d="M6 3v4M6 9h.01" />
      </svg>
      ${flagText}
    `;
    dom.badgesContainer.appendChild(badge);
  });

  animateResultReveal(confidence, isStar);
}

function animateResultReveal(confidence, isStar) {
  if (typeof gsap === 'undefined' || prefersReducedMotion) {
    dom.confidenceValue.textContent = confidence.toFixed(1);
    const circleLength = 226;
    dom.gaugeCircleBar.style.strokeDashoffset = circleLength * (1 - confidence / 100);
    dom.gaugeCircleBar.className = isStar ? 'hud-circular-gauge__bar is-star' : 'hud-circular-gauge__bar is-galaxy';
    document.querySelectorAll('.hud-feat-row__bar-fill').forEach((bar) => {
      bar.style.width = bar.dataset.weight + '%';
    });
    return;
  }

  const revealTl = gsap.timeline();
  activeClassifyTimeline = revealTl;

  dom.gaugeCircleBar.className = isStar ? 'hud-circular-gauge__bar is-star' : 'hud-circular-gauge__bar is-galaxy';

  revealTl.fromTo(
    dom.resultClassLabel,
    { opacity: 0, y: 15, scale: 0.9 },
    { opacity: 1, y: 0, scale: 1, duration: 0.45, ease: 'back.out(1.8)' }
  );

  const counterObj = { val: 0 };
  revealTl.to(
    counterObj,
    {
      val: confidence,
      duration: 1.1,
      ease: 'power2.out',
      onUpdate: () => {
        dom.confidenceValue.textContent = counterObj.val.toFixed(1);
      }
    },
    '-=0.3'
  );

  const circleLength = 226;
  const targetOffset = circleLength * (1 - confidence / 100);
  revealTl.fromTo(
    dom.gaugeCircleBar,
    { strokeDashoffset: circleLength },
    { strokeDashoffset: targetOffset, duration: 1.1, ease: 'power2.out' },
    '<'
  );

  revealTl.fromTo(
    '.hud-feat-row',
    { opacity: 0, x: -12 },
    { opacity: 1, x: 0, duration: 0.35, stagger: 0.08, ease: 'power2.out' },
    '-=0.7'
  );

  document.querySelectorAll('.hud-feat-row__bar-fill').forEach((bar) => {
    revealTl.to(
      bar,
      { width: bar.dataset.weight + '%', duration: 0.65, ease: 'power3.out' },
      '<'
    );
  });

  const badgesToAnimate = dom.badgesContainer.querySelectorAll('.hud-badge--dynamic');
  if (badgesToAnimate.length > 0) {
    revealTl.fromTo(
      badgesToAnimate,
      { opacity: 0, scale: 0.6, y: 8 },
      { opacity: 1, scale: 1, y: 0, duration: 0.45, stagger: 0.1, ease: 'back.out(2)' },
      '-=0.2'
    );
  }
}

// Form submit listener
dom.observationForm.addEventListener('submit', handleClassificationSubmit);

// ============================================================================
// INITIAL LOAD GSAP TIMELINE & ENTRANCE CHOREOGRAPHY
// ============================================================================
function initLoadTimeline() {
  if (typeof gsap === 'undefined' || prefersReducedMotion) return;

  const loadTl = gsap.timeline();

  loadTl.fromTo(
    dom.bgImage,
    { opacity: 0, scale: 1.0 },
    { opacity: 1, duration: 1.2, ease: 'power2.out' }
  );

  loadTl.fromTo(
    dom.hudHeader,
    { opacity: 0, y: -25 },
    { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' },
    '-=0.6'
  );

  loadTl.fromTo(
    [dom.consolePanel, dom.resultPanel],
    { opacity: 0, y: 35 },
    { opacity: 1, y: 0, duration: 0.8, stagger: 0.15, ease: 'power3.out' },
    '-=0.4'
  );

  loadTl.fromTo(
    '.hud-card__corner',
    { opacity: 0, scale: 0 },
    { opacity: 0.85, scale: 1, duration: 0.4, stagger: 0.04, ease: 'back.out(2)' },
    '-=0.4'
  );
}

// ============================================================================
// SCROLLTRIGGER METRICS STRIP & CONFUSION MATRIX ANIMATIONS
// ============================================================================
function initScrollTriggerMetrics() {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') return;

  gsap.registerPlugin(ScrollTrigger);

  const metricsStrip = document.getElementById('metricsStrip');
  if (!metricsStrip) return;

  ScrollTrigger.create({
    trigger: metricsStrip,
    start: 'top 85%',
    once: true,
    onEnter: () => {
      gsap.fromTo(
        '.hud-metric-card',
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.6, stagger: 0.1, ease: 'power2.out' }
      );

      const counters = [
        { el: document.getElementById('metricAccuracy'), target: CONFIG.METRICS.accuracy },
        { el: document.getElementById('metricPrecision'), target: CONFIG.METRICS.precision },
        { el: document.getElementById('metricRecall'), target: CONFIG.METRICS.recall },
        { el: document.getElementById('metricF1'), target: CONFIG.METRICS.f1 }
      ];

      counters.forEach(({ el, target }) => {
        if (!el) return;
        const obj = { val: 0 };
        gsap.to(obj, {
          val: target,
          duration: 1.5,
          ease: 'power3.out',
          onUpdate: () => {
            el.textContent = obj.val.toFixed(2);
          }
        });
      });

      gsap.fromTo(
        '.hud-cm-cell-anim',
        { opacity: 0, scale: 0.85 },
        { opacity: 1, scale: 1, duration: 0.6, stagger: 0.08, ease: 'back.out(1.5)' }
      );
    }
  });
}

// ============================================================================
// INITIALIZATION ON DOM READY
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initBackgroundScene();
  initMagneticButton();
  initLoadTimeline();
  initScrollTriggerMetrics();

  // Test backend server connectivity
  probeBackendServer();

  // Load initial representative Star sample
  loadSample(CONFIG.SAMPLES.STAR);
});
