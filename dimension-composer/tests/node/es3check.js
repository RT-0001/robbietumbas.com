// Parse a .jsx as ECMAScript 3 (ExtendScript's dialect). Exit 1 with the location on failure.
const fs = require("fs");
const acorn = require("acorn");
const src = fs.readFileSync(process.argv[2], "utf8").replace(/^#target.*$/m, "");
try {
  acorn.parse(src, { ecmaVersion: 3, allowReserved: false });
  console.log("ES3 OK");
} catch (e) {
  console.log("ES3 FAIL:", e.message);
  process.exit(1);
}
