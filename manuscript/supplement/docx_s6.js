// docx_s6.js: write Supplementary Material v6 (.docx) from s6_model.json (produced by build_s6.py).
// Usage: node docx_s6.js s6_model.json out.docx
// Layout: A4, 2.5 cm margins, Times New Roman 12 pt text, tables in 7 to 7.5 pt with repeated header rows,
// wide tables on landscape pages, page numbers in the footer. Table cells are lists of lines, each a list of runs.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, PageOrientation, Footer, PageNumber, TableLayoutType, VerticalAlign, PageBreak,
} = require("docx");

const [, , modelPath, outPath] = process.argv;
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));

const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1418;
const TEXT_W = PAGE_W - 2 * MARGIN;
const TEXT_W_LAND = PAGE_H - 2 * MARGIN;

function textRuns(runs, extra = {}) {
  return runs.map((r) => new TextRun({
    text: r.text, bold: !!r.bold || !!extra.bold, italics: !!r.italic, font: FONT,
    highlight: r.highlight ? "yellow" : undefined, size: extra.size,
  }));
}
function para(runs, opts = {}) {
  return new Paragraph({ children: textRuns(runs, opts.run || {}), spacing: { after: 160, line: 276 }, ...opts.para });
}
function heading(text, level) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    spacing: { before: level === 1 ? 360 : 240, after: 120 }, keepNext: true,
    children: [new TextRun({ text, bold: true, font: FONT, size: level === 1 ? 28 : 24, color: "000000" })],
  });
}
function pngSize(path) {
  const b = fs.readFileSync(path);
  if (b.toString("ascii", 1, 4) !== "PNG") throw new Error("not a PNG: " + path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
const cellText = (lines) => lines.map((l) => l.map((r) => r.text).join("")).join(" ");
const maxWord = (s) => Math.max(0, ...String(s).split(/\s+/).map((w) => w.length));

// Column widths: each column gets room for its longest word, the rest is shared in proportion to how much
// more room each column's typical cell would need to sit on one line. Widths sum exactly to the table width.
function columnWidths(header, rows, total, fontHalfPts) {
  // Each column needs room for its longest word (minW) and would like room for its longest line (need,
  // capped at 48 characters). If every column's need fits, the needs are scaled up to the table width;
  // otherwise every column gets its minimum and the remainder is shared in proportion to unmet need.
  const CH = Math.round(fontHalfPts * 5.2), CH_B = Math.round(fontHalfPts * 5.6), PAD = 140;
  const n = header.length;
  const body = rows.filter((r) => Array.isArray(r));
  const sub = rows.filter((r) => r && r.subheader).map((r) => r.subheader);
  const minW = [], need = [];
  for (let j = 0; j < n; j++) {
    const cells = body.map((r) => r[j]);
    const heads = [header[j], ...sub.map((s) => s[j])];
    const hw = Math.max(...heads.map((h) => maxWord(cellText(h)))) * CH_B;
    const bw = Math.max(0, ...cells.map((c) => Math.max(...c.map((l) => maxWord(l.map((r) => r.text).join(""))))));
    minW.push(Math.max(hw, bw * CH) + PAD);
    const longest = Math.max(0, ...cells.map((c) => Math.max(...c.map((l) => l.map((r) => r.text).join("").length))));
    const hlen = Math.max(...heads.map((h) => cellText(h).length));
    need.push(Math.max(minW[j], Math.min(Math.max(longest, Math.ceil(hlen / 2)), 48) * CH + PAD));
  }
  const sumNeed = need.reduce((a, b) => a + b, 0);
  const sumMin = minW.reduce((a, b) => a + b, 0);
  let w;
  if (sumNeed <= total) {
    w = need.map((x) => Math.floor((x / sumNeed) * total));
  } else if (sumMin >= total) {
    w = minW.map((m) => Math.floor((m / sumMin) * total));
  } else {
    const extra = need.map((x, j) => x - minW[j]);
    const sumExtra = extra.reduce((a, b) => a + b, 0) || 1;
    w = minW.map((m, j) => Math.floor(m + (extra[j] / sumExtra) * (total - sumMin)));
  }
  w[n - 1] += total - w.reduce((a, b) => a + b, 0);
  return w;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function makeCell(lines, width, opts) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, columnSpan: opts.span,
    shading: opts.fill ? { type: ShadingType.CLEAR, color: "auto", fill: opts.fill } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 }, verticalAlign: VerticalAlign.TOP,
    children: lines.map((l) => new Paragraph({ spacing: { after: 0, line: 228 },
      children: textRuns(l, { size: opts.size, bold: opts.bold }) })),
  });
}

function makeTable(t, width) {
  const size = t.font || 15;
  const widths = columnWidths(t.header, t.rows, width, size);
  const n = t.header.length;
  const rows = [new TableRow({ tableHeader: true,
    children: t.header.map((h, j) => makeCell(h, widths[j], { size, bold: true, fill: "EDEDED" })) })];
  for (const r of t.rows) {
    if (r.section) {
      rows.push(new TableRow({ cantSplit: true, children: [makeCell([r.section], width, { size, bold: true, fill: "F6F6F6", span: n })] }));
    } else if (r.subheader) {
      rows.push(new TableRow({ cantSplit: true,
        children: r.subheader.map((h, j) => makeCell(h, widths[j], { size, bold: true, fill: "EDEDED" })) }));
    } else {
      rows.push(new TableRow({ cantSplit: true, children: r.map((c, j) => makeCell(c, widths[j], { size })) }));
    }
  }
  return new Table({ width: { size: width, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows });
}

function tableBlock(b, width) {
  return [
    new Paragraph({ keepNext: true, spacing: { before: 120, after: 120 },
      children: [new TextRun({ text: `Supplementary Table ${b.number}. `, bold: true, font: FONT }),
                 new TextRun({ text: b.title, bold: true, font: FONT })] }),
    makeTable(b, width),
    new Paragraph({ spacing: { before: 80, after: 240, line: 240 }, children: textRuns(b.note, { size: 18 }) }),
  ];
}

function figureBlock(b) {
  const img = pngSize(b.path);
  const maxW = 600, maxH = 760;                       // pixels at 96 dpi, inside a portrait text area
  let wPx = maxW, hPx = Math.round((img.h / img.w) * maxW);
  if (hPx > maxH) { hPx = maxH; wPx = Math.round((img.w / img.h) * maxH); }
  return [
    new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 },
      children: [new ImageRun({ type: "png", data: img.data, transformation: { width: wPx, height: hPx },
        altText: { title: `Supplementary Figure ${b.number}`, name: `FigureS${b.number}`,
                   description: b.caption.map((r) => r.text).join("").slice(0, 250) } })] }),
    new Paragraph({ spacing: { after: 240, line: 240 }, children: textRuns(b.caption, { size: 20 }) }),
  ];
}

// ------------------------------------------------------------------ sections
const sections = [];
let current = [];
let currentLand = false;
const footer = { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) };
function props(land) {
  return { page: { size: land ? { width: PAGE_W, height: PAGE_H, orientation: PageOrientation.LANDSCAPE }
                                : { width: PAGE_W, height: PAGE_H },
                   margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } };
}
function close() {
  if (current.length) sections.push({ properties: props(currentLand), footers: footer, children: current });
  current = [];
}
function ensure(land) {
  if (land !== currentLand) { close(); currentLand = land; }
}

const f = model.front;
current.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: f.title, bold: true, size: 32, font: FONT })] }));
current.push(new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: f.subtitle, italics: true, size: 26, font: FONT })] }));
current.push(new Paragraph({ spacing: { after: 360 }, children: [new TextRun({ text: f.author, size: 24, font: FONT })] }));

let figuresStarted = false;
for (const b of model.blocks) {
  if (b.type === "heading") {
    ensure(false);
    if (b.level === 1 && (b.text === "Supplementary Tables" || b.text === "Supplementary Figures" || b.text === "Supplementary Methods" ||
                          b.text === "Supplementary Results" || b.text === "References") && current.length) {
      current.push(new Paragraph({ children: [new PageBreak()] }));
    }
    current.push(heading(b.text, b.level));
  } else if (b.type === "para") {
    ensure(false);
    current.push(para(b.runs));
  } else if (b.type === "contents") {
    ensure(false);
    current.push(heading("Contents", 2));
    for (const it of b.items) current.push(new Paragraph({ spacing: { after: 30 }, indent: { left: 360, hanging: 360 },
      children: [new TextRun({ text: it, font: FONT, size: 20 })] }));
  } else if (b.type === "table") {
    ensure(!!b.landscape);
    current.push(...tableBlock(b, b.landscape ? TEXT_W_LAND : TEXT_W));
  } else if (b.type === "figure") {
    ensure(false);
    if (figuresStarted) current.push(new Paragraph({ children: [new PageBreak()] }));
    figuresStarted = true;
    current.push(...figureBlock(b));
  } else if (b.type === "ref") {
    ensure(false);
    current.push(new Paragraph({ spacing: { after: 100, line: 252 }, indent: { left: 360, hanging: 360 },
      children: textRuns(b.runs, { size: 22 }) }));
  } else {
    throw new Error("unknown block " + b.type);
  }
}
close();

const doc = new Document({
  creator: "Corbin Smith", title: "Supplementary Material",
  styles: { default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
    ] },
  sections,
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(outPath, buf); console.log(`docx written: ${outPath} (${sections.length} sections)`); });
