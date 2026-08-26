#!/usr/bin/env deno
// =====================================================================
// Estrae da visualizzazione.html i codici Python ESATTI generati
// dall'app per il criterio "Riuso per discesa della loss sul batch"
// (M/M_H automatici via riduzione relativa di J_batch). Li salva in
// altro/script/descent_codice_generato/:
//
//   Dynamic GD    -> gd_wolfe.py       (line search di Wolfe)
//   BB-CCV        -> bb_armijo.py      (line search di Armijo)
//   Newton-CG     -> newton_cg_tied.py (H_k legato a S_k)
//   Newton-CG L1  -> newton_l1_tied.py (H_k legato a S_k)
//
// Stesso approccio usato per le varianti *Validation: harness Deno con
// DOM fittizio, si valuta lo <script> dell'app e si chiamano i generatori
// generateGDDescent / generateBBDescent / generateNewtonCGDescent /
// generateNewtonL1Descent con le opzioni dei default della teoria
// (batch dinamico CCV attivo, line search come nella Sez. 6 / Appendice E,
// Hessiana in modalita' legata a S_k).
//
// Uso:  deno run --allow-read --allow-write estrai_codice_descesa.mjs
// =====================================================================

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP  = join(HERE, '..', '..', 'visualizzazione.html');
const OUT  = join(HERE, 'descent_codice_generato');

// ---------------------------------------------------------------------
// DOM fittizio (abbastanza robusto per il top-level dell'app)
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
// Estrazione dello <script> principale (l'ultimo blocco inline)
// ---------------------------------------------------------------------
const html = readFileSync(APP, 'utf-8');
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const main = blocks.reduce((a, b) => (b.length > a.length ? b : a), '');

// Valuta il codice dell'app in una funzione (scope proprio) e recupera
// i generatori di codice Python delle varianti *Descent.
const factory = new Function(
  main +
    '\nreturn { generateGDDescent, generateBBDescent, generateNewtonCGDescent, generateNewtonL1Descent };'
);
const { generateGDDescent, generateBBDescent, generateNewtonCGDescent, generateNewtonL1Descent } = factory();

// ---------------------------------------------------------------------
// Generazione dei 4 codici (varianti *Descent, default della teoria)
// ---------------------------------------------------------------------
const GD_OPTS = {
  dynamicBatch: true,
  lineSearch: 'wolfe',
  reuseHessian: true,
  hkSubset: true,
  hessianFreeL1: false,
  descentLoss: true,
  descentTol: 1e-4,
  descentMinAbs: 0.0,
  descentPatience: 1,
  descentFreq: 1,
};

const FILES = [
  ['gd_wolfe.py',        'dynamic_gd',     generateGDDescent,       GD_OPTS],
  ['bb_armijo.py',       'bb_dynamic_gd',  generateBBDescent,       { ...GD_OPTS, lineSearch: 'armijo' }],
  ['newton_cg_tied.py',  'newton_cg',      generateNewtonCGDescent, GD_OPTS],
  ['newton_l1_tied.py',  'newton_l1',      generateNewtonL1Descent, GD_OPTS],
];

mkdirSync(OUT, { recursive: true });
for (const [fname, entry, gen, opts] of FILES) {
  const code = gen(opts);
  if (!code.includes(`def ${entry}(`)) {
    throw new Error(`codice generato per ${fname} senza firma ${entry}: estrazione fallita`);
  }
  writeFileSync(join(OUT, fname), code, 'utf-8');
  console.log(`scritto ${join('descent_codice_generato', fname)} (${code.length} byte)`);
}
console.log('OK: 4 codici *Descent estratti da visualizzazione.html');
