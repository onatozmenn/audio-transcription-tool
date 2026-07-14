import { jsPDF } from "jspdf";
import type { AppLocale } from "./app-copy";

type PdfExportOptions = {
  fileName: string;
  locale: AppLocale;
  text: string;
  title: string;
};

const PAGE_WIDTH_MM = 210;
const PAGE_HEIGHT_MM = 297;
const MARGIN_MM = 20;

function exportSelectablePdf(options: PdfExportOptions): void {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const maxWidth = PAGE_WIDTH_MM - MARGIN_MM * 2;
  let y = 25;

  if (options.title.trim()) {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(16);
    doc.setTextColor(255, 0, 0);
    const titleLines = doc.splitTextToSize(options.title.trim(), maxWidth) as string[];
    doc.text(titleLines, MARGIN_MM, y);
    y += titleLines.length * 8 + 6;
  }

  doc.setFont("helvetica", "normal");
  doc.setFontSize(12);
  doc.setTextColor(0, 0, 0);
  const textLines = doc.splitTextToSize(options.text, maxWidth) as string[];
  const lineHeight = 6;

  for (const line of textLines) {
    if (y + lineHeight > PAGE_HEIGHT_MM - MARGIN_MM) {
      doc.addPage();
      y = 25;
    }
    doc.text(line, MARGIN_MM, y, { maxWidth, align: "justify" });
    y += lineHeight;
  }

  doc.save(options.fileName);
}

function wrapCanvasText(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  locale: AppLocale,
): string[] {
  if (!text) return [""];

  const segmenter = new Intl.Segmenter(locale, { granularity: "word" });
  const segments = Array.from(segmenter.segment(text), (segment) => segment.segment);
  const lines: string[] = [];
  let currentLine = "";

  const pushLongSegment = (segment: string) => {
    for (const character of Array.from(segment)) {
      const candidate = currentLine + character;
      if (currentLine && context.measureText(candidate).width > maxWidth) {
        lines.push(currentLine.trimEnd());
        currentLine = character;
      } else {
        currentLine = candidate;
      }
    }
  };

  for (const segment of segments) {
    const candidate = currentLine + segment;
    if (!currentLine || context.measureText(candidate).width <= maxWidth) {
      currentLine = candidate;
      continue;
    }

    lines.push(currentLine.trimEnd());
    currentLine = "";
    const normalizedSegment = segment.trimStart();
    if (context.measureText(normalizedSegment).width <= maxWidth) {
      currentLine = normalizedSegment;
    } else {
      pushLongSegment(normalizedSegment);
    }
  }

  if (currentLine || lines.length === 0) lines.push(currentLine.trimEnd());
  return lines;
}

function exportUnicodeCanvasPdf(options: PdfExportOptions): void {
  const doc = new jsPDF({ unit: "mm", format: "a4" });
  const canvas = document.createElement("canvas");
  canvas.width = 1240;
  canvas.height = 1754;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas rendering is unavailable.");

  const pixelsPerMm = canvas.width / PAGE_WIDTH_MM;
  const margin = MARGIN_MM * pixelsPerMm;
  const maxWidth = canvas.width - margin * 2;
  const bottom = canvas.height - margin;
  const isRtl = /[\u0590-\u08FF]/u.test(`${options.title}\n${options.text}`);
  const textX = isRtl ? canvas.width - margin : margin;
  let y = margin;
  let pageHasContent = false;

  const resetPage = () => {
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.textBaseline = "alphabetic";
    context.direction = isRtl ? "rtl" : "ltr";
    context.textAlign = isRtl ? "right" : "left";
    y = margin;
    pageHasContent = false;
  };

  let pageIndex = 0;
  const commitPage = () => {
    if (!pageHasContent) return;
    if (pageIndex > 0) doc.addPage();
    doc.addImage(
      canvas.toDataURL("image/jpeg", 0.92),
      "JPEG",
      0,
      0,
      PAGE_WIDTH_MM,
      PAGE_HEIGHT_MM,
      undefined,
      "FAST",
    );
    pageIndex += 1;
    resetPage();
  };

  const drawWrappedText = (
    value: string,
    font: string,
    color: string,
    lineHeight: number,
    paragraphSpacing: number,
  ) => {
    context.font = font;
    context.fillStyle = color;

    for (const paragraph of value.split(/\r?\n/u)) {
      const lines = wrapCanvasText(context, paragraph, maxWidth, options.locale);
      for (const line of lines) {
        if (y + lineHeight > bottom) {
          commitPage();
          context.font = font;
          context.fillStyle = color;
        }
        context.fillText(line, textX, y, maxWidth);
        y += lineHeight;
        pageHasContent = true;
      }
      y += paragraphSpacing;
    }
  };

  resetPage();
  if (options.title.trim()) {
    drawWrappedText(
      options.title.trim(),
      '700 34px "Noto Sans", "Segoe UI", sans-serif',
      "#ff0000",
      45,
      18,
    );
  }
  drawWrappedText(
    options.text,
    '400 24px "Noto Sans", "Segoe UI", sans-serif',
    "#000000",
    36,
    12,
  );
  commitPage();
  doc.save(options.fileName);
}

export function exportTranscriptAsPdf(options: PdfExportOptions): void {
  const hasUnicode = /[^\x09\x0A\x0D\x20-\x7E]/u.test(`${options.title}\n${options.text}`);
  if (hasUnicode) {
    exportUnicodeCanvasPdf(options);
    return;
  }

  exportSelectablePdf(options);
}
