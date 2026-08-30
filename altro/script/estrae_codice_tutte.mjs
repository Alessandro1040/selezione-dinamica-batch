#!/usr/bin/env deno
// =====================================================================
// Estrae da visualizzazione.html i codici Python ESATTI generati
// dall'app per TUTTE le varianti usate dalle tabelle della Sezione 6:
//
//   base       : gd_wolfe / bb_armijo / newton_cg_tied / newton_l1_tied
//   riuso M=inf: gd_reuse / bb_reuse / newton_cg_reuse / newton_l1_reuse
//   H indipend.: newton_cg_hind / newton_l1_hind   (M=10, H M_H=inf)
//   validation : gd_val / bb_val / newton_cg_val / newton_l1_val
//   descesa    : gd_desc / bb_desc / newton_cg_desc / newton_l1_desc
//
// I file salvati in altro/script/codice_generato/ sono la FONTE delle
// stringhe incorporate in riproduci_tutte_le_tabelle.py (che le esegue
// con exec, come l'helper _batch_run dell'app): le righe eseguite dallo
// script sono quindi IDENTICHE a quelle generate dall'applicazione.
//
// Uso:  deno run --allow-read --allow-write estrae_codice_tutte.mjs
// =====================================================================

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP  = join(HERE, '..', '..', 'visualizzazione.html');
const OUT  = join(HERE, 'codice_generato');

// ---------------------------------------------------------------------
// DOM fittizio (come estrai_codice_descesa.mjs)
// ---------------------------------------------------------------------
function makeElement(id) {
  const el = {
    id, value: '', checked: false, innerHTML: '', textContent: '', className: '',
    style: new Proxy({}, { get: () => '', set: () => true }),
    dataset: {},
    children: [],
    options: [],
    selectedIndex: 0,
    files: [],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {},
    removeEventListener() {},
    appendChild() {},
    insertBefore() {},
    removeChild() {},
    replaceChild() {},
    querySelector() { return null; },
    querySelectorAll() { return []; },
    closest() { return null; },
    focus() {},
    blur() {},
    click() {},
    setAttribute() {},
    getAttribute() { return null; },
    removeAttribute() {},
    scrollIntoView() {},
    getContext() { return null; },
    toDataURL() { return ''; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 0, height: 0 }; },
  };
  return el;
}

const elements = new Map();
const fakeDoc = {
  getElementById(id) {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  },
  createElement() { return makeElement(''); },
  addEventListener() {},
  removeEventListener() {},
  querySelector() { return null; },
  querySelectorAll() { return []; },
  body: makeElement('body'),
  documentElement: makeElement('html'),
  head: makeElement('head'),
};

globalThis.document = fakeDoc;
globalThis.window = globalThis;
globalThis.self = globalThis;
globalThis.navigator = { platform: 'deno', userAgent: 'deno' };
globalThis.Plotly = {
  react() {}, newPlot() {}, plot() {}, update() {}, purge() {},
  relayout() {}, toImage() { return { toDataURL() { return ''; } }; },
};
globalThis.loadPyodide = async () => ({
  runPython() {}, globals: { get() { return null; }, set() {} },
});
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.MathJax = { typesetPromise: async () => {} };
globalThis.URL.createObjectURL = () => '';
globalThis.URL.revokeObjectURL = () => {};
globalThis.confirm = () => true;
globalThis.alert = () => {};
globalThis.fetch = async () => ({ ok: true, text: async () => '' });

// ---------------------------------------------------------------------
// Estrazione dello <script> principale
// ---------------------------------------------------------------------
const html = readFileSync(APP, 'utf-8');
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = blocks.reduce((a, b) => (b.length > a.length ? b : a), '');

const factory = new Function(
  main +
    '\nreturn { generateAlgoCode };'
);
const { generateAlgoCode } = factory();

// ---------------------------------------------------------------------
// Opzioni: default della teoria (Sez. 6 / Appendice B/E dell'app)
// ---------------------------------------------------------------------
function baseOpts(extra) {
  return Object.assign({
    hkSubset: true,
    lineSearch: 'wolfe',
    dynamicBatch: true,
    hessianFreeL1: false,
    reuseBatch: false,
    maxConsec: null,
    reuseHessian: true,
    maxHessianReuse: null,
    validation: false,
    valPct: 0.2,
    valTol: 1e-4,
    valPatience: 3,
    valFreq: 1,
    valMinAbs: 0,
    valStrategy: 'fixed',
    descentLoss: false,
    descentTol: 1e-4,
    descentMinAbs: 0,
    descentPatience: 1,
    descentFreq: 1,
  }, extra || {});
}

const ARM = { lineSearch: 'armijo' };   // BB-CCV e Newton-L1

const CFG = [
  // (file, algo, opts)
  ['gd_wolfe.py',              'gd',        baseOpts({})],
  ['bb_armijo.py',             'bb',        baseOpts(ARM)],
  ['newton_cg_tied.py',        'newton_cg', baseOpts({})],
  ['newton_l1_tied.py',        'newton_l1', baseOpts(ARM)],
  ['gd_reuse.py',              'gd',        baseOpts({ reuseBatch: true, maxConsec: null })],
  ['bb_reuse.py',              'bb',        baseOpts({ ...ARM, reuseBatch: true, maxConsec: null })],
  ['newton_cg_reuse.py',       'newton_cg', baseOpts({ reuseBatch: true, maxConsec: null })],
  ['newton_l1_reuse.py',       'newton_l1', baseOpts({ ...ARM, reuseBatch: true, maxConsec: null })],
  ['newton_cg_hind.py',        'newton_cg', baseOpts({ reuseBatch: true, maxConsec: 10, reuseHessian: false, maxHessianReuse: null })],
  ['newton_l1_hind.py',        'newton_l1', baseOpts({ ...ARM, reuseBatch: true, maxConsec: 10, reuseHessian: false, maxHessianReuse: null })],
  ['gd_val.py',                'gd',        baseOpts({ validation: true })],
  ['bb_val.py',                'bb',        baseOpts({ ...ARM, validation: true })],
  ['newton_cg_val.py',         'newton_cg', baseOpts({ validation: true })],
  ['newton_l1_val.py',         'newton_l1', baseOpts({ ...ARM, validation: true })],
  ['gd_desc.py',               'gd',        baseOpts({ descentLoss: true })],
  ['bb_desc.py',               'bb',        baseOpts({ ...ARM, descentLoss: true })],
  ['newton_cg_desc.py',        'newton_cg', baseOpts({ descentLoss: true })],
  ['newton_l1_desc.py',        'newton_l1', baseOpts({ ...ARM, descentLoss: true })],
];

const ENTRY = {
  gd: 'dynamic_gd', bb: 'bb_dynamic_gd', newton_cg: 'newton_cg', newton_l1: 'newton_l1',
};

mkdirSync(OUT, { recursive: true });
for (const [fname, algo, opts] of CFG) {
  const code = generateAlgoCode(algo, opts);
  if (!code.includes(`def ${ENTRY[algo]}(`)) {
    throw new Error(`codice per ${fname} senza firma ${ENTRY[algo]}: estrazione fallita`);
  }
  writeFileSync(join(OUT, fname), code, 'utf-8');
  console.log(`scritto ${fname} (${code.length} byte)`);
}
console.log(`OK: ${CFG.length} codici estratti da visualizzazione.html`);
// ====CHUNK-END====

