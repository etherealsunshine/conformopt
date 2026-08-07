import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT_DIR = "/Users/utkarsh/qfitonsteroids/results/slides_movable_point_v1_build";
const FINAL = "/Users/utkarsh/qfitonsteroids/results/slides/movable_point_synthetic_tail_v1.pptx";
const SOURCE_BRIEF = "/Users/utkarsh/.codex/attachments/07c7ce2a-9d78-4b26-a742-bc4b9973ba5e/pasted-text.txt";
const FROZEN_DIR = "/Users/utkarsh/qfitonsteroids/results/frozen_v3_metric_data_0p5";

const W = 1280;
const H = 720;
const M = 72;
const INK = "#000000";
const MUTED = "#59616d";
const RULE = "#B8BCC4";
const PANEL = "#EDEDED";
const LIGHT = "#F7F7F7";
const ACCENT = "#3D8DFF";
const ACCENT_LIGHT = "#6DCBF4";
const WARN = "#C9473A";
const GOOD = "#1A7F64";

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

function shape(slide, name, left, top, width, height, fill = "none", line = "none") {
  return slide.shapes.add({
    geometry: "rect",
    name,
    position: { left, top, width, height },
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
  });
}

function text(slide, name, value, pos, style = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    name,
    position: pos,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    fontSize: style.fontSize ?? 22,
    color: style.color ?? INK,
    bold: style.bold ?? false,
    alignment: style.alignment ?? "left",
  };
  return box;
}

function title(slide, value, kicker = "Synthetic 20-site audit") {
  text(slide, "kicker", kicker.toUpperCase(), { left: M, top: 42, width: 520, height: 28 }, {
    fontSize: 14,
    bold: true,
    color: MUTED,
  });
  text(slide, "title", value, { left: M, top: 78, width: 950, height: 92 }, {
    fontSize: 38,
    bold: true,
    color: INK,
  });
  shape(slide, "rule", M, 170, W - 2 * M, 1, RULE, RULE);
}

function bulletList(slide, items, left, top, width, gap = 42, fontSize = 24) {
  items.forEach((item, i) => {
    text(slide, `bullet-${i}`, `- ${item}`, { left, top: top + i * gap, width, height: gap - 4 }, {
      fontSize,
      color: INK,
    });
  });
}

function smallLabel(slide, value, left, top, width = 240) {
  text(slide, `label-${left}-${top}`, value, { left, top, width, height: 26 }, {
    fontSize: 14,
    bold: true,
    color: MUTED,
  });
}

function notes(slide, body, sources = []) {
  const sourceBlock = [
    "",
    "[Sources]",
    `- User-provided slide brief: ${SOURCE_BRIEF}`,
    `- Frozen metric data copied locally: ${FROZEN_DIR}`,
    ...sources,
  ].join("\n");
  slide.speakerNotes.textFrame.setText(`${body}${sourceBlock}`);
  slide.speakerNotes.setVisible(true);
}

function addMetric(slide, value, label, left, top, color = INK) {
  text(slide, `metric-${label}`, value, { left, top, width: 240, height: 62 }, {
    fontSize: 48,
    bold: true,
    color,
  });
  text(slide, `metric-label-${label}`, label, { left, top: top + 62, width: 265, height: 54 }, {
    fontSize: 18,
    color: MUTED,
  });
}

function addFooter(slide, num) {
  text(slide, "footer", `${num}`, { left: W - 104, top: H - 56, width: 32, height: 24 }, {
    fontSize: 13,
    color: MUTED,
    alignment: "right",
  });
}

const slides = [
  {
    kind: "cover",
    title: "Misplaced, not missing",
    subtitle: "Why the synthetic tail is no longer an insertion problem, and why escape remains the live question.",
    notes: "Open with the distinction. We used to phrase the problem as 'we do not find the minor conformer.' The newer evidence says the minor density is often already in the model, but in the wrong configuration. That changes the mechanism we should test.",
  },
  {
    kind: "cascade",
    title: "The frozen metric is strong, but the tail is real",
    notes: "Counts are per start across 20 sites x 50 starts. The middle of the cascade matters: only four starts drop at the rotamer/direct/symmetry gates after occupancy. The large downstream drop is tmol, not a geometric mismatch between soft objective and hard audit.",
  },
  {
    kind: "minor",
    title: "Single-recovery failures mostly lose the minor state",
    notes: "This is conditional on exactly one conformer being recovered. The classification is by occupancy rank rather than A/B label, because the A/B label is not consistently the major/minor state across the panel.",
  },
  {
    kind: "nulls",
    title: "The nulls point away from search coverage",
    notes: "Do not narrate this as a graveyard of attempts. The pattern is the argument. Initialization is the sharpest null: even slots initialized on the minor moved away, so the issue was not simply lack of initial coverage.",
  },
  {
    kind: "ceiling",
    title: "The residual does not contain a conformer-shaped hole",
    notes: "This is the ceiling slide. If deposited ground truth only fits the residual at r=0.340 and fitted occupancy 0.070, no insertion rule can do much better. That result closes insertion, not escape.",
  },
  {
    kind: "limits",
    title: "Some of the tail is information-limited",
    notes: "This caps upside. 5Z8H and 2VFP together account for 70 of the 142 missed-minor failures. The right interpretation is not that the optimizer is innocent everywhere, but that a large chunk of the tail is near or beyond the support-resolution limit.",
  },
  {
    kind: "barrier",
    title: "Misplaced should be movable, but only by crossing a barrier",
    notes: "This is the movable-point argument. The wrong configuration explains real density at about 66 percent efficiency, so the objective rewards it locally. Gradient descent will not first break that partial fit in order to find a better one.",
  },
  {
    kind: "decision",
    title: "The decision is whether to test escape cleanly",
    notes: "Frame this as the question for Fraser. S2 at 30 degrees per step was too much: it damaged convergence and worsened both missed-minor and missed-major categories. That does not refute escape. S1 at 10 degrees is the clean version of the question.",
  },
  {
    kind: "backup-metric",
    title: "Backup: frozen-v3 metric definition",
    notes: "Use only if someone asks about what changed between metric versions. The important answer is that v3 fixed assignment accounting and is frozen for model experiments.",
  },
  {
    kind: "backup-sites",
    title: "Backup: the tail is concentrated by site",
    notes: "Use this if someone wants to know whether the missed-minor result is panel-wide. It is not uniform; 2VFP, 5Z8H, 1ZV8, 7UO8, and 4C16 carry much of the tail.",
  },
];

async function main() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir("/Users/utkarsh/qfitonsteroids/results/slides", { recursive: true });
  const p = Presentation.create({ slideSize: { width: W, height: H } });

  slides.forEach((spec, index) => {
    const slide = p.slides.add();
    slide.background.fill = "#FFFFFF";

    if (spec.kind === "cover") {
      shape(slide, "left-band", 0, 0, 38, H, ACCENT, ACCENT);
      text(slide, "small-kicker", "SYNTHETIC 20-SITE AUDIT", { left: M, top: 72, width: 520, height: 30 }, {
        fontSize: 16, bold: true, color: MUTED,
      });
      text(slide, "cover-title", spec.title, { left: M, top: 170, width: 940, height: 110 }, {
        fontSize: 64, bold: true, color: INK,
      });
      text(slide, "subtitle", spec.subtitle, { left: M, top: 315, width: 870, height: 92 }, {
        fontSize: 26, color: MUTED,
      });
      addMetric(slide, "742", "both conformers found", M, 500, GOOD);
      addMetric(slide, "626", "strict successes", M + 285, 500, INK);
      addMetric(slide, "S1", "escape test still open", M + 570, 500, ACCENT);
      notes(slide, spec.notes);
      addFooter(slide, index + 1);
      return;
    }

    title(slide, spec.title);

    if (spec.kind === "cascade") {
      const categories = ["Found", "Occupancy", "Geom.", "Strict"];
      const values = [742, 714, 710, 626];
      slide.charts.add("bar", {
        position: { left: 110, top: 230, width: 720, height: 330 },
        categories,
        series: [{ name: "starts", values, fill: ACCENT }],
        hasLegend: false,
        dataLabels: { showValue: true, position: "outEnd" },
        yAxis: { minimumScale: 0, maximumScale: 1000, majorUnit: 250 },
      });
      text(slide, "cascade-callout", "Rotamer, clash, and symmetry remove only 4 starts after occupancy; tmol is the dominant remaining filter.", { left: 880, top: 252, width: 300, height: 205 }, {
        fontSize: 24, bold: true, color: INK,
      });
      text(slide, "cascade-sub", "Frozen metric: qfit-synth20-merge050-one-to-one-tmol044-v3", { left: 880, top: 470, width: 300, height: 70 }, {
        fontSize: 17, color: MUTED,
      });
      notes(slide, spec.notes);
    }

    if (spec.kind === "minor") {
      addMetric(slide, "187", "single-recovery starts", 112, 235, INK);
      addMetric(slide, "142", "missed the minor conformer", 395, 235, WARN);
      addMetric(slide, "45", "missed the major conformer", 705, 235, MUTED);
      slide.charts.add("bar", {
        position: { left: 140, top: 440, width: 700, height: 150 },
        categories: ["missed minor", "missed major"],
        series: [{ name: "starts", values: [142, 45], fill: ACCENT }],
        hasLegend: false,
        dataLabels: { showValue: true, position: "outEnd" },
        yAxis: { minimumScale: 0, maximumScale: 160, majorUnit: 40 },
      });
      text(slide, "minor-note", "The asymmetry is occupancy-rank based, not an A/B label artifact.", { left: 890, top: 345, width: 285, height: 120 }, {
        fontSize: 24, bold: true,
      });
      notes(slide, spec.notes);
    }

    if (spec.kind === "nulls") {
      bulletList(slide, [
        "Occupancy freeze timing did not rescue starved slots",
        "Initialization stratification did not keep minor-seeded slots there",
        "Separation, torsion reachability, and mask geometry did not explain the tail",
        "Signal weighting changed the score and hurt recovery",
      ], 110, 230, 780, 58, 25);
      shape(slide, "nulls-panel", 930, 228, 245, 258, LIGHT, RULE);
      text(slide, "nulls-panel-title", "Pattern", { left: 955, top: 255, width: 195, height: 36 }, {
        fontSize: 24, bold: true,
      });
      text(slide, "nulls-panel-copy", "The failed arms mostly changed scoring or local search. None cleanly tested escape from a rewarded partial fit.", { left: 955, top: 312, width: 190, height: 132 }, {
        fontSize: 21, color: MUTED,
      });
      notes(slide, spec.notes);
    }

    if (spec.kind === "ceiling") {
      addMetric(slide, "r = 0.340", "deposited missed-conformer ceiling", 110, 230, WARN);
      addMetric(slide, "0.070", "least-squares fitted occupancy", 515, 230, WARN);
      addMetric(slide, "66%", "median density support at misplaced slots", 900, 230, ACCENT);
      bulletList(slide, [
        "If the conformer were absent, the residual should call for its deposited occupancy",
        "Instead, the residual asks for only a small leftover component",
        "Conclusion: insertion is closed; the mass is already partly absorbed",
      ], 130, 445, 930, 48, 24);
      notes(slide, spec.notes);
    }

    if (spec.kind === "limits") {
      const sites = ["5Z8H", "2VFP", "1ZV8", "7UO8", "4C16"];
      const vals = [26, 44, 20, 15, 7];
      slide.charts.add("bar", {
        position: { left: 105, top: 235, width: 640, height: 300 },
        categories: sites,
        series: [{ name: "missed-minor failures", values: vals, fill: WARN }],
        hasLegend: false,
        dataLabels: { showValue: true, position: "outEnd" },
        yAxis: { minimumScale: 0, maximumScale: 50, majorUnit: 10 },
      });
      text(slide, "limits-callout", "5Z8H + 2VFP account for 70 of 142 missed-minor failures, where two-state support is barely resolvable.", { left: 815, top: 260, width: 330, height: 170 }, {
        fontSize: 28, bold: true,
      });
      text(slide, "limits-sub", "That caps the upside of any optimizer-only fix.", { left: 815, top: 455, width: 330, height: 60 }, {
        fontSize: 21, color: MUTED,
      });
      notes(slide, spec.notes);
    }

    if (spec.kind === "barrier") {
      shape(slide, "energy-axis", 140, 560, 880, 2, INK, INK);
      shape(slide, "local-well", 225, 407, 230, 110, "#D0EDFA", "#D0EDFA");
      shape(slide, "barrier", 520, 260, 185, 255, PANEL, PANEL);
      shape(slide, "true-well", 765, 365, 230, 150, "#D6EFE5", "#D6EFE5");
      text(slide, "local", "rewarded partial fit\n66% density support", { left: 238, top: 430, width: 205, height: 70 }, {
        fontSize: 21, bold: true, alignment: "center",
      });
      text(slide, "bar", "barrier", { left: 548, top: 345, width: 130, height: 40 }, {
        fontSize: 24, bold: true, alignment: "center", color: MUTED,
      });
      text(slide, "true", "correct conformer\nbetter global fit", { left: 782, top: 412, width: 195, height: 70 }, {
        fontSize: 21, bold: true, alignment: "center",
      });
      text(slide, "gd", "Gradient descent will not break a fit it is currently rewarded for making.", { left: 130, top: 222, width: 475, height: 80 }, {
        fontSize: 28, bold: true,
      });
      text(slide, "sde", "Noise or restart can test escape; insertion cannot.", { left: 710, top: 242, width: 390, height: 68 }, {
        fontSize: 26, bold: true, color: ACCENT,
      });
      notes(slide, spec.notes);
    }

    if (spec.kind === "decision") {
      addMetric(slide, "S2", "30 degrees/step", 110, 230, WARN);
      addMetric(slide, "742 -> 575", "both-found recovery collapsed", 400, 230, WARN);
      addMetric(slide, "S1", "10 degrees/step remains untested", 760, 230, ACCENT);
      bulletList(slide, [
        "S2 looks like non-convergence, not a clean failed-escape result",
        "The honest test is smaller perturbation around local minima",
        "Decision: spend another quarter on S1 or move to experimental density",
      ], 130, 445, 930, 48, 24);
      notes(slide, spec.notes);
    }

    if (spec.kind === "backup-metric") {
      bulletList(slide, [
        "v1 undercounted recovery through independent greedy labels",
        "v2 fixed one-to-one assignment but omitted genuine near-duplicate occupancy",
        "v3 uses 0.5 A protected merge, one-to-one assignment, and tmol <= +0.44",
        "Frozen control cascade: 742 -> 714 -> 710 -> 710 -> 710 -> 626",
      ], 110, 230, 920, 58, 24);
      notes(slide, spec.notes);
    }

    if (spec.kind === "backup-sites") {
      const rows = [
        ["1ZV8", 20, 1],
        ["2VFP", 44, 0],
        ["5Z8H", 26, 2],
        ["7UO8", 15, 6],
        ["4C16", 7, 17],
      ];
      smallLabel(slide, "Site", 130, 230);
      smallLabel(slide, "Missed minor", 390, 230);
      smallLabel(slide, "Missed major", 670, 230);
      rows.forEach((r, i) => {
        const y = 275 + i * 64;
        text(slide, `site-${i}`, r[0], { left: 130, top: y, width: 160, height: 38 }, { fontSize: 26, bold: true });
        shape(slide, `minorbar-${i}`, 390, y + 4, r[1] * 7, 28, WARN, WARN);
        text(slide, `minorval-${i}`, `${r[1]}`, { left: 710, top: y, width: 60, height: 32 }, { fontSize: 24, bold: true });
        shape(slide, `majorbar-${i}`, 810, y + 4, r[2] * 10, 28, MUTED, MUTED);
        text(slide, `majorval-${i}`, `${r[2]}`, { left: 1055, top: y, width: 60, height: 32 }, { fontSize: 24, bold: true });
      });
      notes(slide, spec.notes);
    }

    addFooter(slide, index + 1);
  });

  for (const [i, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(`${OUT_DIR}/${stem}.png`, await p.export({ slide, format: "png", scale: 1 }));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(`${OUT_DIR}/${stem}.layout.json`, await layout.text());
  }
  await writeBlob(`${OUT_DIR}/montage.webp`, await p.export({ format: "webp", montage: true, scale: 1 }));
  await fs.writeFile(`${OUT_DIR}/source-notes.txt`, [
    "Sources used:",
    SOURCE_BRIEF,
    FROZEN_DIR,
    "",
    "No external sources used. Claims and figures come from the user brief and local frozen-v3 artifacts.",
  ].join("\n"));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(FINAL);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
