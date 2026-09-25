// Minimal mock of the Photoshop ExtendScript DOM, enough to execute a dimcomp
// builder script end to end in Node and inspect the resulting layer tree.
// It checks control flow, math and API usage - not Photoshop's rendering.
"use strict";
const fs = require("fs");
const vm = require("vm");

const [, , scriptPath, fontsMode] = process.argv;
const alerts = [];
const saved = [];
let actions = [];

function UnitValue(v, u) { this.value = v; this.unit = u || "px"; }
UnitValue.prototype.as = function () { return this.value; };
function uv(n) { return new UnitValue(n, "px"); }
function num(x) { return x instanceof UnitValue ? x.value : Number(x); }

function SolidColor() { const self = this; this.rgb = { red: 0, green: 0, blue: 0,
  set hexValue(h) { self.rgb.red = parseInt(h.slice(0, 2), 16); self.rgb.green = parseInt(h.slice(2, 4), 16); self.rgb.blue = parseInt(h.slice(4, 6), 16); self.rgb._hex = h; },
  get hexValue() { return self.rgb._hex; } }; }

let uid = 0;
class Layer {
  constructor(doc, parent, kind) { this.doc = doc; this.parent = parent; this.id = ++uid; this.name = "Layer " + this.id;
    this.kind = kind || "NORMAL"; this.rect = [0, 0, 0, 0]; this.isBackgroundLayer = false; this.typename = "ArtLayer"; }
  set kind(k) { this._kind = k; if (k === "TEXT") this.textItem = new TextItem(this); }
  get kind() { return this._kind; }
  get bounds() { const r = this._kind === "TEXT" ? this.textItem.box() : this.rect; return r.map(uv); }
  translate(dx, dy) { dx = num(dx); dy = num(dy);
    if (this._kind === "TEXT") { this.textItem.dx += dx; this.textItem.dy += dy; }
    else this.rect = [this.rect[0] + dx, this.rect[1] + dy, this.rect[2] + dx, this.rect[3] + dy]; }
  resize(sx, sy, anchor) { if (anchor !== "TOPLEFT") throw new Error("mock: only TOPLEFT resize");
    const r = this.rect; this.rect = [r[0], r[1], r[0] + (r[2] - r[0]) * sx / 100, r[1] + (r[3] - r[1]) * sy / 100]; }
  remove() { this.parent._remove(this); }
  move(rel, where) { this.parent._remove(this); if (where === "PLACEATEND" && rel instanceof LayerSet) { rel._layers.push(this); this.parent = rel; }
    else throw new Error("mock: unsupported move " + where); }
  duplicate(target, where) { const l = new Layer(target, target, this._kind); l.rect = this.rect.slice(); l.name = this.name;
    if (where === "PLACEATBEGINNING") target._layers.unshift(l); else throw new Error("mock: duplicate needs PLACEATBEGINNING");
    return l; }
}
class TextItem {
  constructor(layer) { this.layer = layer; this.contents = ""; this._size = uv(12); this.position = [0, 0]; this.horizontalScale = 100;
    this.justification = "LEFT"; this.dx = 0; this.dy = 0; this.font = "ArialMT"; }
  set size(v) { if (!(v instanceof UnitValue)) throw new Error("mock: set size with a UnitValue"); this._size = v; }
  get size() { return this._size; }
  box() { const s = num(this._size), w = this.contents.length * s * 0.55 * this.horizontalScale / 100, cap = 0.7 * s;
    const x = num(this.position[0]), y = num(this.position[1]);
    const x0 = this.justification === "CENTER" ? x - w / 2 : x;
    return [x0 + this.dx, y - cap + this.dy, x0 + w + this.dx, y + this.dy]; }
}
class Collection {
  constructor(owner, make) { this.owner = owner; this.make = make; }
  add() { const l = this.make(); this.owner._layers.unshift(l); return l; }
}
class LayerSet {
  constructor(doc, parent) { this.doc = doc; this.parent = parent; this._layers = []; this.name = "Group"; this.typename = "LayerSet";
    this.artLayers = new Collection(this, () => new Layer(doc, this));
    this.layerSets = new Collection(this, () => new LayerSet(doc, this)); }
  get layers() { return this._layers; }
  _remove(l) { this._layers = this._layers.filter((x) => x !== l); }
}
class Doc extends LayerSet {
  constructor(w, h, name) { super(null, null); this.doc = this; this.width = uv(w); this.height = uv(h); this.name = name;
    this.artLayers = new Collection(this, () => new Layer(this, this));
    this.layerSets = new Collection(this, () => new LayerSet(this, this));
    this.activeLayer = null; this.pathItems = new Paths(this); this.selection = { fill() {}, deselect() {}, selectAll() {} }; }
  mergeVisibleLayers() {} close() { app.documents.splice(app.documents.indexOf(this), 1); }
  saveAs(f) { saved.push(f.fsName); }
}
class Paths { constructor(doc) { this.doc = doc; this.items = []; }
  add(name, subs) { const pts = [].concat(...subs.map((s) => s.entireSubPath.map((p) => p.anchor)));
    if (!pts.length) throw new Error("mock: empty path");
    pts.forEach((p) => { if (!isFinite(p[0]) || !isFinite(p[1])) throw new Error("mock: NaN path point in " + name); });
    const it = { name, pts, subs: subs.length, select: () => { this.doc._selPath = it; }, remove() {},
      makeSelection() {} }; this.items.push(it); return it; } }

function PathPointInfo() {} function SubPathInfo() {}
function ActionDescriptor() { this.putReference = this.putObject = this.putDouble = () => {}; }
function ActionReference() { this.putClass = () => {}; }

const fonts = fontsMode === "nogibson"
  ? [{ family: "Arial", style: "Regular", postScriptName: "ArialMT" }]
  : [{ family: "Gibson", style: "Regular", postScriptName: "Gibson-Regular" },
     { family: "Gibson", style: "SemiBold", postScriptName: "Gibson-SemiBold" },
     { family: "Arial", style: "Regular", postScriptName: "ArialMT" }];

const product = { w: 1400, h: 1000 };
const app = {
  documents: [], fonts, preferences: { rulerUnits: "IN", typeUnits: "PT" }, backgroundColor: new SolidColor(),
  get activeDocument() { return this._active; }, set activeDocument(d) { this._active = d; },
  documents_add(w, h) {}, open(f) { const d = new Doc(product.w, product.h, f.name); const l = new Layer(d, d); l.rect = [40, 30, 40 + product.w, 30 + product.h];
    d._layers.push(l); d.activeLayer = l; this.documents.push(d); this._active = d; return d; },
};
app.documents.add = function (w, h, res, name) { const d = new Doc(w, h, name); const bg = new Layer(d, d); bg.isBackgroundLayer = true; bg.name = "Background";
  d._layers.push(bg); d.activeLayer = bg; app.documents.push(d); app._active = d; return d; };

function executeAction(id) {
  actions.push(id);
  const d = app.activeDocument;
  if (id === "newPlacedLayer") { d.activeLayer.name = "smart"; return; }
  if (id === "Mk  ") { const p = d._selPath; if (!p) throw new Error("mock: no selected path");
    const parent = d.activeLayer.parent || d; const l = new Layer(d, parent, "SOLIDFILL");
    const xs = p.pts.map((q) => q[0]), ys = p.pts.map((q) => q[1]);
    l.rect = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; l.subpaths = p.subs;
    const i = parent._layers.indexOf(d.activeLayer); parent._layers.splice(Math.max(i, 0), 0, l); d.activeLayer = l; return; }
  throw new Error("mock: unknown action " + id);
}
// ExtendScript's File is callable with or without `new`
function File(p) { if (!(this instanceof File)) return new File(p); this.fsName = String(p); this.name = this.fsName.split("/").pop(); }
Object.defineProperty(File.prototype, "exists", { get() { return /\.(tif|tiff|png)$/.test(this.name); } });
Object.defineProperty(File.prototype, "parent", { get() { return "/work"; } });
File.openDialog = () => null;

const ctx = {
  app, UnitValue, SolidColor, PathPointInfo, SubPathInfo, ActionDescriptor, ActionReference, File,
  executeAction, charIDToTypeID: (s) => s, stringIDToTypeID: (s) => s,
  alert: (m) => alerts.push(String(m)), $: { fileName: "/work/script.jsx" },
  Units: { PIXELS: "PX" }, TypeUnits: { PIXELS: "PX" }, LayerKind: { TEXT: "TEXT", NORMAL: "NORMAL" },
  Justification: { CENTER: "CENTER", LEFT: "LEFT" }, PointKind: { CORNERPOINT: "CORNER" },
  ShapeOperation: { SHAPEADD: "ADD" }, DialogModes: { NO: "NO" }, NewDocumentMode: { RGB: "RGB" },
  DocumentFill: { BACKGROUNDCOLOR: "BG" }, ElementPlacement: { PLACEATEND: "PLACEATEND", PLACEATBEGINNING: "PLACEATBEGINNING" },
  AnchorPosition: { TOPLEFT: "TOPLEFT" }, SaveOptions: { DONOTSAVECHANGES: "NO" }, SelectionType: { REPLACE: "R" },
  PhotoshopSaveOptions: function () {}, Extension: { LOWERCASE: "lc" },
};
Object.defineProperty(Layer.prototype, "parentDoc", { get() { return this.doc; } });

let src = fs.readFileSync(scriptPath, "utf8").replace(/^#target.*$/m, "");
vm.createContext(ctx);
vm.runInContext(src, ctx, { filename: scriptPath });

function tree(node) {
  return node._layers.map((l) => l instanceof LayerSet
    ? { group: l.name, children: tree(l) }
    : { name: l.name, kind: l._kind, bounds: l.bounds.map((v) => Math.round(v.value)),
        font: l.textItem ? l.textItem.font : undefined, text: l.textItem ? l.textItem.contents : undefined,
        subpaths: l.subpaths });
}
const doc = app.documents[app.documents.length - 1];
process.stdout.write(JSON.stringify({ alerts, saved, tree: tree(doc), prefs: app.preferences }, null, 1));
