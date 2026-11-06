// docx_build.js: write the v12 manuscript .docx from v12_model.json (produced by build_v12.py).
// Usage: node docx_build.js v12_model.json out.docx
// Layout follows the Frontiers manuscript conventions: A4, 2.5 cm margins, Times New Roman 12 pt,
// continuous line numbers, page numbers, word and exhibit counts on page 1, figures and tables in the text.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, LevelFormat, PageOrientation, LineNumberRestartFormat,
  Footer, PageNumber, TableLayoutType, VerticalAlign,
} = require("docx");

const [, , modelPath, outPath] = process.argv;
const model = JSON.parse(fs.readFileSync(modelPath, "utf8"));

const FONT = "Times New Roman";
const PAGE_W = 11906, PAGE_H = 16838, MARGIN = 1418;          // A4, 2.5 cm
const TEXT_W = PAGE_W - 2 * MARGIN;                            // portrait text width (DXA)
const TEXT_W_LAND = PAGE_H - 2 * MARGIN;                       // landscape text width (DXA)

function textRuns(runs, extra = {}) {
  return runs.map((r) => new TextRun({
    text: r.text, bold: !!r.bold, italics: !!r.italic, font: FONT,
    highlight: r.highlight ? "yellow" : undefined, ...extra,
  }));
}

function para(runs, opts = {}) {
  return new Paragraph({ children: textRuns(runs, opts.run || {}), spacing: { after: 160, line: 276 },
                         alignment: opts.align || AlignmentType.LEFT, ...opts.para });
}

function heading(text, level) {
  const size = level === 1 ? 28 : level === 2 ? 24 : 24;
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: level === 1 ? 360 : 240, after: 120 },
    keepNext: true,
    children: [new TextRun({ text, bold: true, font: FONT, size, italics: level === 3, color: "000000" })],
  });
}

function pngSize(path) {
  const b = fs.readFileSync(path);
  if (b.toString("ascii", 1, 4) !== "PNG") throw new Error("not a PNG: " + path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}

// Column widths: every column first gets room for its longest word (so no word is broken across
// lines), then the remaining width goes to the columns whose cells are long, in proportion to how much
// more room they would need to sit on one line. Widths sum exactly to the table width.
function columnWidths(header, rows, total) {
  const CH = 80, CH_B = 88, PAD = 150;             // 8 pt Times: about 80 DXA per character
  const n = header.length;
  const minW = [], want = [];
  for (let j = 0; j < n; j++) {
    const body = rows.map((r) => String(r[j]));
    const hw = Math.max(...String(header[j]).split(/\s+/).map((w) => w.length)) * CH_B;
    const bw = Math.max(0, ...body.map((c) => Math.max(...c.split(/\s+/).map((w) => w.length)))) * CH;
    minW.push(Math.max(hw, bw) + PAD);
    const mean = body.reduce((a, c) => a + c.length, 0) / Math.max(1, body.length);
    want.push(Math.max(minW[j], Math.min(Math.max(mean, String(header[j]).length / 2), 90) * CH + PAD));
  }
  let w;
  const sumMin = minW.reduce((a, b) => a + b, 0);
  if (sumMin >= total) {
    w = minW.map((m) => Math.floor((m / sumMin) * total));
  } else {
    const extra = want.map((x, j) => Math.max(0, x - minW[j]));
    const sumExtra = extra.reduce((a, b) => a + b, 0) || 1;
    const spare = total - sumMin;
    w = minW.map((m, j) => Math.floor(m + (extra[j] / sumExtra) * spare));
  }
  w[n - 1] += total - w.reduce((a, b) => a + b, 0);
  return w;
}

function makeTable(t, width) {
  const widths = columnWidths(t.header, t.rows, width);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const cell = (text, j, isHeader) => new TableCell({
    width: { size: widths[j], type: WidthType.DXA },
    borders,
    shading: isHeader ? { type: ShadingType.CLEAR, color: "auto", fill: "EDEDED" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    verticalAlign: VerticalAlign.TOP,
    children: [new Paragraph({ spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: String(text), font: FONT, size: 16, bold: isHeader })] })],
  });
  const rows = [new TableRow({ tableHeader: true, children: t.header.map((h, j) => cell(h, j, true)) })];
  for (const r of t.rows) rows.push(new TableRow({ cantSplit: true, children: r.map((c, j) => cell(c, j, false)) }));
  return new Table({ width: { size: width, type: WidthType.DXA }, columnWidths: widths,
                     layout: TableLayoutType.FIXED, rows });
}

function tableBlock(b, width) {
  const out = [];
  const m = b.title.match(/^(Table \d\.)\s*(.*)$/);
  out.push(new Paragraph({ keepNext: true, spacing: { before: 240, after: 120 },
    children: [new TextRun({ text: m[1] + " ", bold: true, font: FONT }),
               new TextRun({ text: m[2], bold: true, font: FONT })] }));
  out.push(makeTable(b, width));
  out.push(new Paragraph({ spacing: { before: 80, after: 240, line: 240 },
    children: textRuns(b.note, { size: 18 }) }));
  return out;
}

function figureBlock(b) {
  const img = pngSize(b.path);
  const wPx = 600;                                   // about 15.9 cm at 96 dpi, inside the 16 cm text width
  const hPx = Math.round((img.h / img.w) * wPx);
  return [
    new Paragraph({ keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 },
      children: [new ImageRun({ type: "png", data: img.data, transformation: { width: wPx, height: hPx },
                                altText: { title: `Figure ${b.n}`, description: b.caption.map((r) => r.text).join("").slice(0, 250), name: `Figure${b.n}` } })] }),
    new Paragraph({ spacing: { after: 240, line: 240 }, children: textRuns(b.caption, { size: 20 }) }),
  ];
}

// --------------------------------------------------------------------------- front matter
const f = model.front;
const frontChildren = [
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: f.title, bold: true, size: 32, font: FONT })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: f.author, size: 24, font: FONT, bold: true })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: f.affil, size: 24, font: FONT })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "ORCID: " + f.orcid, size: 24, font: FONT })] }),
  new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: "Correspondence: ", size: 24, font: FONT, bold: true }), ...textRuns(f.corr)] }),
  new Paragraph({ spacing: { after: 240 }, children: [new TextRun({ text: f.counts, size: 22, font: FONT, italics: true })] }),
];

// --------------------------------------------------------------------------- body, split into sections
const sections = [];
let current = [...frontChildren];
const numbering = { config: [{ reference: "recs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
  alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] };

function sectionProps(landscape) {
  return {
    page: {
      size: landscape ? { width: PAGE_W, height: PAGE_H, orientation: PageOrientation.LANDSCAPE } : { width: PAGE_W, height: PAGE_H },
      margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN },
    },
    lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS, distance: 300 },
  };
}
const footer = { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, suppressLineNumbers: true,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) };

function closeSection(landscape) {
  if (current.length) sections.push({ properties: sectionProps(landscape), footers: footer, children: current });
  current = [];
}

let afterAbstract = false;
for (const b of model.blocks) {
  if (b.type === "heading") {
    if (afterAbstract) {           // keywords follow the abstract paragraph
      current.push(new Paragraph({ spacing: { after: 240 }, children: [
        new TextRun({ text: "Keywords: ", bold: true, font: FONT }), new TextRun({ text: f.keywords, font: FONT })] }));
      afterAbstract = false;
    }
    current.push(heading(b.text, b.level));
  } else if (b.type === "para") {
    current.push(para(b.runs));
    if (b.keywords_after) afterAbstract = true;
  } else if (b.type === "olist") {
    for (const it of b.items) current.push(new Paragraph({ numbering: { reference: "recs", level: 0 },
      spacing: { after: 80, line: 276 }, children: textRuns(it) }));
  } else if (b.type === "table") {
    if (b.landscape) {
      closeSection(false);
      current.push(...tableBlock(b, TEXT_W_LAND));
      closeSection(true);
    } else {
      current.push(...tableBlock(b, TEXT_W));
    }
  } else if (b.type === "figure") {
    current.push(...figureBlock(b));
  } else if (b.type === "ref") {
    current.push(new Paragraph({ spacing: { after: 100, line: 252 }, indent: { left: 360, hanging: 360 },
      children: textRuns(b.runs, { size: 22 }) }));
  } else {
    throw new Error("unknown block " + b.type);
  }
}
closeSection(false);

const doc = new Document({
  creator: "Corbin Smith",
  title: f.title,
  styles: { default: { document: { run: { font: FONT, size: 24 } } },
            paragraphStyles: [
              { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { outlineLevel: 0 } },
              { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
                run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { outlineLevel: 1 } },
            ] },
  numbering,
  sections,
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(outPath, buf);
  console.log(`docx written: ${outPath} (${sections.length} sections)`);
});
