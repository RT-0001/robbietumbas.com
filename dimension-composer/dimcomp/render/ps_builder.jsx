// ---------------------------------------------------------------------------
// dimcomp Photoshop builder (ExtendScript / ES3). DATA is prepended by dimcomp.
// Builds a layered PSD: product smart object, live Gibson text, vector shapes.
// Line breaks around labels are computed here from Photoshop's own text bounds,
// so spacing is exact for the real font.
// ---------------------------------------------------------------------------

var STEP = "start";
var WARN = [];

var px = function (v) { return Number(v.as("px")); };
var bnds = function (layer) { var b = layer.bounds; return [px(b[0]), px(b[1]), px(b[2]), px(b[3])]; };
var color = function (hex) { var c = new SolidColor(); c.rgb.hexValue = hex.replace("#", ""); return c; };
var norm = function (s) { return String(s).toLowerCase().replace(/[\s_-]/g, ""); };

// ---- fonts ----------------------------------------------------------------
var FONT_CACHE = {};
var STYLE_NAMES = { 100: "thin", 200: "extralight", 300: "light", 400: "regular", 500: "medium",
                    600: "semibold", 700: "bold", 800: "extrabold", 900: "black" };

var findFont = function (family, weight, psHint) {
    var key = family + "|" + weight;
    if (FONT_CACHE[key]) return FONT_CACHE[key];
    var want = STYLE_NAMES[weight] || "regular";
    var fam = norm(family), hit = null, sameFamily = [];
    for (var i = 0; i < app.fonts.length; i++) {
        var f = app.fonts[i];
        if (psHint && f.postScriptName === psHint) { hit = f.postScriptName; break; }
        if (norm(f.family) === fam) {
            sameFamily.push(f.style);
            if (norm(f.style) === want || (want === "regular" && norm(f.style) === "book")) hit = f.postScriptName;
        }
    }
    if (!hit) {
        WARN.push(family + " " + want + " is not active in Photoshop" +
                  (sameFamily.length ? " (found styles: " + sameFamily.join(", ") + ")" : "") +
                  ". Activate it in Adobe Fonts and re-run; using Arial for now.");
        hit = "ArialMT";
    }
    FONT_CACHE[key] = hit;
    return hit;
};

var CAP_CACHE = {};
var capRatio = function (doc, ps) {
    // cap height / font size, measured on a throwaway "H"
    if (CAP_CACHE[ps]) return CAP_CACHE[ps];
    var l = doc.artLayers.add();
    l.kind = LayerKind.TEXT;
    var t = l.textItem;
    t.contents = "H"; t.font = ps; t.size = new UnitValue(200, "px"); t.position = [100, 400];
    var b = bnds(l);
    l.remove();
    CAP_CACHE[ps] = (b[3] - b[1]) / 200;
    return CAP_CACHE[ps];
};

// Text centered on its cap height at (cx, cy). Returns the layer.
var addText = function (doc, parent, spec, cx, cy, capPx, opts) {
    opts = opts || {};
    var f = spec.font;
    var ps = findFont(f.family, f.weight, null);
    var size = capPx / capRatio(doc, ps);
    var l = parent.artLayers.add();
    l.kind = LayerKind.TEXT;
    l.name = opts.name || spec.id;
    var t = l.textItem;
    t.contents = opts.text !== undefined ? opts.text : spec.text;
    t.font = ps;
    t.size = new UnitValue(size, "px");
    if (f.h_scale && f.h_scale !== 1) t.horizontalScale = f.h_scale * 100;
    t.color = color(spec.fill);
    t.justification = Justification.CENTER;
    t.position = [cx, cy + capPx / 2];            // point text: position is on the baseline
    var b = bnds(l);
    if (opts.hangX !== undefined) {
        // center only the numerals on hangX; the trailing inch mark hangs outside
        var digits = stripInchMarks(String(t.contents));
        var tmp = parent.artLayers.add(); tmp.kind = LayerKind.TEXT;
        tmp.textItem.contents = digits; tmp.textItem.font = ps; tmp.textItem.size = new UnitValue(size, "px");
        tmp.textItem.position = [0, size * 2];
        var tb = bnds(tmp); tmp.remove();
        l.translate(opts.hangX - (tb[2] - tb[0]) / 2 - b[0], 0);
    } else {
        l.translate((cx - (b[0] + b[2]) / 2), 0);    // exact horizontal centering on real glyphs
    }
    return l;
};

// trailing inch marks off the end (no regex: old parsers misread quotes inside regex literals)
var stripInchMarks = function (str) {
    var end = str.length;
    while (end > 0) {
        var code = str.charCodeAt(end - 1);
        if (code === 0x201D || code === 0x2033 || code === 34 || code === 39) end--; else break;
    }
    return str.substring(0, end);
};

var cxcy = function (box) { return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[3] - box[1]]; };

// Title: registered / trademark signs become their own small raised layers (DOM text can't style a sub-range)
var addTitle = function (doc, spec) {
    var c = cxcy(spec.box), cap = c[2];
    var runs = spec.runs;
    if (runs.length === 1) return addText(doc, doc, spec, c[0], c[1], cap, { name: "title" });
    var grp = doc.layerSets.add(); grp.name = "title";
    var ps = findFont(spec.font.family, spec.font.weight, null);
    var ratio = capRatio(doc, ps);
    var layers = [], widths = [], lead = [];
    var space = measureSpace(doc, ps, cap / ratio);
    for (var i = 0; i < runs.length; i++) {
        var r = runs[i];
        var txt = r.text.replace(/^\s+|\s+$/g, "");
        lead.push(/^\s/.test(r.text) ? space : 0);
        var l = grp.artLayers.add();
        l.kind = LayerKind.TEXT; l.name = "title_" + (i + 1);
        var t = l.textItem;
        t.contents = txt; t.font = ps; t.size = new UnitValue(cap / ratio * r.scale, "px");
        t.color = color(spec.fill); t.justification = Justification.LEFT;
        t.position = [0, c[1] + cap / 2 - r.rise * cap];
        layers.push(l);
        var b = bnds(l); widths.push(b[2] - b[0]);
    }
    var gap = 0.06 * cap, total = 0;
    for (i = 0; i < runs.length; i++) total += widths[i] + lead[i] + (i ? gap : 0);
    var x = c[0] - total / 2;
    for (i = 0; i < runs.length; i++) {
        x += lead[i] + (i ? gap : 0);
        var bb = bnds(layers[i]);
        layers[i].translate(x - bb[0], 0);
        x += widths[i];
    }
    return grp;
};

var measureSpace = function (doc, ps, size) {
    var w = function (s) {
        var l = doc.artLayers.add(); l.kind = LayerKind.TEXT;
        l.textItem.contents = s; l.textItem.font = ps; l.textItem.size = new UnitValue(size, "px"); l.textItem.position = [0, size * 2];
        var b = bnds(l); l.remove(); return b[2] - b[0];
    };
    return w("H H") - w("HH");
};

// ---- shapes ---------------------------------------------------------------
var makePath = function (doc, polys, name) {
    var subs = [];
    for (var i = 0; i < polys.length; i++) {
        var pts = [];
        for (var j = 0; j < polys[i].length; j++) {
            var p = new PathPointInfo();
            p.kind = PointKind.CORNERPOINT;
            p.anchor = polys[i][j]; p.leftDirection = polys[i][j]; p.rightDirection = polys[i][j];
            pts.push(p);
        }
        var sp = new SubPathInfo();
        sp.closed = true; sp.operation = ShapeOperation.SHAPEADD; sp.entireSubPath = pts;
        subs.push(sp);
    }
    return doc.pathItems.add(name, subs);
};

// Vector shape layer (solid fill + vector mask) from polygons; raster fallback if the action fails.
var addShape = function (doc, polys, hex, name) {
    STEP = "shape " + name;
    var path = makePath(doc, polys, name + "_path");
    var c = color(hex);
    try {
        path.select();
        var d = new ActionDescriptor(), ref = new ActionReference();
        ref.putClass(stringIDToTypeID("contentLayer"));
        d.putReference(charIDToTypeID("null"), ref);
        var lay = new ActionDescriptor(), fill = new ActionDescriptor(), rgb = new ActionDescriptor();
        rgb.putDouble(charIDToTypeID("Rd  "), c.rgb.red);
        rgb.putDouble(charIDToTypeID("Grn "), c.rgb.green);
        rgb.putDouble(charIDToTypeID("Bl  "), c.rgb.blue);
        fill.putObject(charIDToTypeID("Clr "), charIDToTypeID("RGBC"), rgb);
        lay.putObject(charIDToTypeID("Type"), stringIDToTypeID("solidColorLayer"), fill);
        d.putObject(charIDToTypeID("Usng"), stringIDToTypeID("contentLayer"), lay);
        executeAction(charIDToTypeID("Mk  "), d, DialogModes.NO);
        doc.activeLayer.name = name;
    } catch (e) {
        var l = doc.artLayers.add(); l.name = name;
        path.makeSelection(0, true, SelectionType.REPLACE);
        doc.selection.fill(c); doc.selection.deselect();
        WARN.push("drew " + name + " as pixels (shape layer action failed: " + e + ")");
    }
    try { path.remove(); } catch (e2) {}
    return doc.activeLayer;
};

// Thin quad per segment = stroked line as a filled vector shape
var lineQuads = function (segs, w) {
    var out = [];
    for (var i = 0; i < segs.length; i++) {
        var a = segs[i][0], b = segs[i][1];
        var dx = b[0] - a[0], dy = b[1] - a[1], L = Math.sqrt(dx * dx + dy * dy);
        if (L < 1) continue;
        var nx = -dy / L * w / 2, ny = dx / L * w / 2;
        out.push([[a[0] + nx, a[1] + ny], [b[0] + nx, b[1] + ny], [b[0] - nx, b[1] - ny], [a[0] - nx, a[1] - ny]]);
    }
    return out;
};

// Line minus padded label box (Liang-Barsky clip) -> 0..2 segments
var splitLine = function (p0, p1, box, gap) {
    var x0 = box[0] - gap, y0 = box[1] - gap, x1 = box[2] + gap, y1 = box[3] + gap;
    var dx = p1[0] - p0[0], dy = p1[1] - p0[1];
    var P = [-dx, dx, -dy, dy], Q = [p0[0] - x0, x1 - p0[0], p0[1] - y0, y1 - p0[1]];
    var tin = 0, tout = 1;
    for (var i = 0; i < 4; i++) {
        if (P[i] === 0) { if (Q[i] < 0) return [[p0, p1]]; continue; }
        var r = Q[i] / P[i];
        if (P[i] < 0) { if (r > tin) tin = r; } else { if (r < tout) tout = r; }
    }
    if (tin >= tout) return [[p0, p1]];
    var at = function (t) { return [p0[0] + dx * t, p0[1] + dy * t]; };
    var out = [];
    if (tin > 0.001) out.push([p0, at(tin)]);
    if (tout < 0.999) out.push([at(tout), p1]);
    return out;
};

var inGroup = function (layer, grp) { try { layer.move(grp, ElementPlacement.PLACEATEND); } catch (e) {} };

// ---- images ---------------------------------------------------------------
var b64decode = function (str) {
    var chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    var out = [], buf = 0, bits = 0;
    for (var i = 0; i < str.length; i++) {
        var c = str.charAt(i);
        if (c === "=") break;
        var v = chars.indexOf(c);
        if (v < 0) continue;
        buf = ((buf << 6) | v) & 0xFFFFFF; bits += 6;
        if (bits >= 8) { bits -= 8; out.push(String.fromCharCode((buf >> bits) & 255)); }
    }
    return out.join("");
};

var assetFile = function (name) {
    var f = new File(Folder.temp + "/dimcomp_" + name);
    f.encoding = "BINARY";
    f.open("w"); f.write(b64decode(DATA.assets[name])); f.close();
    return f;
};

// Open an image, copy its (merged) layer into doc as a smart object named `name`,
// and fit its visible pixels to target [x0, y0, x1, y1].
var placeImage = function (doc, f, target, name, checkAspect) {
    var src = app.open(f);
    if (src.layers.length > 1) src.mergeVisibleLayers();
    var srcLayer = src.activeLayer;
    if (srcLayer.isBackgroundLayer) WARN.push(name + ": image has no transparency; it will cover the background");
    var lay = srcLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
    src.close(SaveOptions.DONOTSAVECHANGES);
    app.activeDocument = doc;
    doc.activeLayer = lay;
    try { executeAction(stringIDToTypeID("newPlacedLayer"), undefined, DialogModes.NO); lay = doc.activeLayer; }
    catch (e) { WARN.push(name + " kept as pixels (smart object conversion failed)"); }
    lay.name = name;
    var b = bnds(lay);
    var sx = (target[2] - target[0]) / (b[2] - b[0]), sy = (target[3] - target[1]) / (b[3] - b[1]);
    if (checkAspect && Math.abs(sx / sy - 1) > 0.02)
        WARN.push(name + " proportions differ from the layout photo by " + Math.round(Math.abs(sx / sy - 1) * 100) +
                  "% - is it the same cutout?");
    lay.resize(sx * 100, sx * 100, AnchorPosition.TOPLEFT);
    b = bnds(lay);
    lay.translate(target[0] - b[0], target[1] - b[1]);
    return lay;
};

var placeProduct = function (doc, spec) {
    STEP = "open product photo";
    var f = new File(File($.fileName).parent + "/" + spec.src_name);
    if (!f.exists) {
        f = File.openDialog("Select the product photo (" + spec.src_name + ")");
        if (!f) throw new Error("no product photo selected");
    }
    STEP = "place product";
    return placeImage(doc, f, spec.alpha_bbox, "product", true);
};

// ---- main -----------------------------------------------------------------
var dimcompBuild = function () {
    var C = DATA.scene.canvas;
    STEP = "new document";
    var oldBg = app.backgroundColor;
    app.backgroundColor = color(C.bg);
    var doc = app.documents.add(C.w, C.h, 72, DATA.out_name, NewDocumentMode.RGB, DocumentFill.BACKGROUNDCOLOR);
    app.backgroundColor = oldBg;

    var L = DATA.scene.layers, i, j;
    var byId = {};
    for (i = 0; i < L.length; i++) byId[L[i].id] = L[i];

    placeProduct(doc, byId.product);

    // callouts
    for (i = 0; i < L.length; i++) {
        if (L[i].type !== "group") continue;
        STEP = "callout " + L[i].id;
        var g = doc.layerSets.add(); g.name = L[i].id;
        var kids = L[i].children, shapes = [];
        for (j = 0; j < kids.length; j++) {
            var k = kids[j];
            if (k.type === "text") { var c = cxcy(k.box); addText(doc, g, k, c[0], c[1], c[2]); }
            else if (k.type === "polygon" || k.type === "image") shapes.push(k);
        }
        for (j = shapes.length - 1; j >= 0; j--) {      // shapes under the text, first listed lowest
            doc.activeLayer = g.layers[0];
            var s;
            if (shapes[j].type === "image") {
                STEP = "place " + shapes[j].id;
                s = placeImage(doc, assetFile(shapes[j].src_name), shapes[j].alpha_bbox, shapes[j].id, false);
            } else {
                s = addShape(doc, [shapes[j].points], shapes[j].fill, shapes[j].id);
            }
            inGroup(s, g);
        }
    }

    // dimensions
    var dims = doc.layerSets.add(); dims.name = "dimensions";
    for (i = 0; i < L.length; i++) {
        var d = L[i];
        if (d.type !== "dimension") continue;
        STEP = "dimension " + d.axis;
        var dg = dims.layerSets.add(); dg.name = d.id;
        var c2 = cxcy(d.label.box);
        var lab = addText(doc, dg, d.label, c2[0], c2[1], c2[2],
                          { name: d.id + "_label", hangX: d.label.hang_x });
        var segs = splitLine(d.line[0], d.line[1], bnds(lab), d.gap);
        doc.activeLayer = lab;
        var sh = addShape(doc, lineQuads(segs, d.stroke), d.color, d.id + "_line");
        inGroup(sh, dg);
    }

    STEP = "title";
    addTitle(doc, byId.title);

    STEP = "save";
    var out = new File(File($.fileName).parent + "/" + DATA.out_name + ".psd");
    var opts = new PhotoshopSaveOptions(); opts.layers = true; opts.embedColorProfile = true;
    doc.saveAs(out, opts, true, Extension.LOWERCASE);
    return out;
};

(function () {
    var ru = app.preferences.rulerUnits, tu = app.preferences.typeUnits;
    app.preferences.rulerUnits = Units.PIXELS;
    app.preferences.typeUnits = TypeUnits.PIXELS;
    try {
        var out = dimcompBuild();
        alert("Built " + DATA.out_name + ".psd\n" + (WARN.length ? "\nCheck:\n- " + WARN.join("\n- ") : "No warnings."));
    } catch (e) {
        alert("dimcomp stopped at step: " + STEP + "\n\n" + e + (e.line ? "\n(script line " + e.line + ")" : "") +
              (WARN.length ? "\n\nEarlier warnings:\n- " + WARN.join("\n- ") : ""));
    } finally {
        app.preferences.rulerUnits = ru;
        app.preferences.typeUnits = tu;
    }
})();
