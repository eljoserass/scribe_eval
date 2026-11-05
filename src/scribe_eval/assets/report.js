const data = JSON.parse(document.getElementById("data").textContent);
const root = document.getElementById("report");
const tip = document.getElementById("tooltip");
const SVG = "http://www.w3.org/2000/svg";
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else node[key] = value;
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

const gb = (bytes) => (bytes / 1e9).toFixed(2) + " GB";
const secs = (value) => value.toFixed(1) + " s";
const number = (value) =>
  value === null || value === undefined ? "—" : Number(value).toLocaleString("en", { maximumFractionDigits: 3 });

function metricValue(name, key, value) {
  if (name.includes("rss")) return gb(value);
  if (name === "wer" || name === "cer" || key === "rate") return (value * 100).toFixed(1) + "%";
  return number(value);
}

/* Colors follow the entity, so a process keeps its color across charts. */
function colorMap(names) {
  return new Map(names.map((name, index) => [name, COLORS[index % COLORS.length]]));
}

function legend(entries) {
  return el("div", { class: "legend" }, ...entries.map(([name, color]) => {
    const key = el("i");
    key.style.background = color;
    return el("span", {}, key, name);
  }));
}

function showTip(event, rows) {
  tip.replaceChildren(...rows.map(([label, value, color]) => {
    const line = el("div");
    if (color) {
      const key = el("i");
      key.style.background = color;
      line.append(key);
    }
    line.append(el("strong", {}, value), " " + label);
    return line;
  }));
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  tip.style.left = Math.min(event.clientX + 14, innerWidth - box.width - 8) + "px";
  tip.style.top = Math.max(event.clientY - box.height - 10, 8) + "px";
}

const hideTip = () => { tip.hidden = true; };

function niceMax(value) {
  if (!value) return 1;
  const step = 10 ** Math.floor(Math.log10(value));
  for (const factor of [1, 1.2, 1.6, 2, 2.4, 3.2, 4, 6, 8, 10]) {
    if (value <= factor * step) return factor * step;
  }
  return 10 * step;
}

function frame(width, height, xMax, yMax, xLabel, yFormat) {
  const margin = { top: 12, right: 24, bottom: 34, left: 56 };
  const plot = { w: width - margin.left - margin.right, h: height - margin.top - margin.bottom };
  const chart = svg("svg", { viewBox: `0 0 ${width} ${height}`, class: "chart" });
  const x = (value) => margin.left + (value / xMax) * plot.w;
  const y = (value) => margin.top + plot.h - (value / yMax) * plot.h;
  for (let tick = 0; tick <= 4; tick++) {
    const value = (yMax / 4) * tick;
    chart.append(svg("line", { class: "grid", x1: margin.left, x2: width - margin.right, y1: y(value), y2: y(value) }));
    chart.append(svg("text", { x: margin.left - 8, y: y(value) + 4, "text-anchor": "end" }, yFormat(value)));
  }
  for (let tick = 0; tick <= 5; tick++) {
    const value = (xMax / 5) * tick;
    chart.append(svg("text", { x: x(value), y: height - 16, "text-anchor": "middle" }, number(Math.round(value * 10) / 10)));
  }
  chart.append(svg("text", { x: margin.left + plot.w / 2, y: height - 2, "text-anchor": "middle" }, xLabel));
  return { chart, x, y, margin, plot };
}

/* Last value of a series at a time, by binary search. */
function valueAt(entry, time) {
  let low = 0;
  let high = entry.seconds.length - 1;
  let found = -1;
  while (low <= high) {
    const middle = (low + high) >> 1;
    if (entry.seconds[middle] <= time) { found = middle; low = middle + 1; } else high = middle - 1;
  }
  return found < 0 ? 0 : entry.bytes[found];
}

/* RAM over time: one line per process, plus the total, with labelled bands. */
function memoryChart(series, windows, colors, xLabel, bandLabels = true) {
  if (!series.length) return el("p", { class: "muted" }, "No memory series.");
  const times = [...new Set(series.flatMap((entry) => entry.seconds))].sort((a, b) => a - b);
  const stride = Math.max(1, Math.floor(times.length / 500));
  const shown = times.filter((_, index) => index % stride === 0);
  const total = (time) => series.reduce((sum, entry) => sum + valueAt(entry, time), 0);
  const totals = shown.map(total);
  const xMax = niceMax(times[times.length - 1]);
  const yMax = niceMax(Math.max(...totals));
  const f = frame(960, 300, xMax, yMax, xLabel, (v) => (v / 1e9).toFixed(1) + " GB");
  windows.forEach(([name, start, end], index) => {
    if (index % 2 && !bandLabels) return;
    f.chart.append(svg("rect", { class: "band", x: f.x(start), y: f.margin.top, width: Math.max(f.x(end) - f.x(start) - 2, 0), height: f.plot.h }));
    if (bandLabels) f.chart.append(svg("text", { x: f.x(start) + 6, y: f.margin.top + 14 }, name));
  });
  const path = (xs, ys) => xs.map((time, i) => `${i ? "L" : "M"}${f.x(time).toFixed(1)} ${f.y(ys[i]).toFixed(1)}`).join("");
  f.chart.append(svg("path", { d: path(shown, totals), fill: "none", stroke: "var(--text)", "stroke-width": 2, "stroke-dasharray": "5 4" }));
  for (const entry of series) {
    f.chart.append(svg("path", { d: path(shown, shown.map((time) => valueAt(entry, time))), fill: "none", stroke: colors.get(entry.process), "stroke-width": 2, "stroke-linejoin": "round" }));
  }
  const hair = svg("line", { class: "hair", y1: f.margin.top, y2: f.margin.top + f.plot.h, visibility: "hidden" });
  f.chart.append(hair);
  const hit = svg("rect", { x: f.margin.left, y: f.margin.top, width: f.plot.w, height: f.plot.h, fill: "transparent" });
  hit.addEventListener("pointermove", (event) => {
    const box = f.chart.getBoundingClientRect();
    const pointer = ((event.clientX - box.left) * (960 / box.width) - f.margin.left) / f.plot.w * xMax;
    const time = shown.reduce((best, value) => Math.abs(value - pointer) < Math.abs(best - pointer) ? value : best, shown[0]);
    const window = windows.find(([, start, end]) => time >= start && time <= end);
    hair.setAttribute("x1", f.x(time));
    hair.setAttribute("x2", f.x(time));
    hair.setAttribute("visibility", "visible");
    showTip(event, [
      [window ? window[0] : "", secs(time)],
      ...series.map((entry) => [entry.process, gb(valueAt(entry, time)), colors.get(entry.process)]),
      ["total", gb(total(time)), "var(--text)"],
    ]);
  });
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTip(); });
  f.chart.append(hit);
  return el("div", {}, el("div", { class: "scroll" }, f.chart), legend([...series.map((entry) => [entry.process, colors.get(entry.process)]), ["total (dashed)", "var(--text)"]]));
}

/* Join every sample on one clock, to see memory across the whole run. */
function wholeRun(samples) {
  const timed = samples.filter((sample) => sample.started_at && sample.memory.length).sort((a, b) => a.started_at - b.started_at);
  if (!timed.length) return { series: [], windows: [] };
  const origin = timed[0].started_at;
  const byProcess = new Map();
  const windows = [];
  for (const sample of timed) {
    const offset = sample.started_at - origin;
    windows.push([sample.id, offset, offset + sample.wall_seconds]);
    for (const entry of sample.memory) {
      if (!byProcess.has(entry.process)) byProcess.set(entry.process, { process: entry.process, seconds: [], bytes: [] });
      const target = byProcess.get(entry.process);
      target.seconds.push(...entry.seconds.map((value) => value + offset));
      target.bytes.push(...entry.bytes);
    }
  }
  return { series: [...byProcess.values()], windows };
}

/* Peak RAM against a length measure, one dot per sample and process. */
function scatter(samples, field, label, colors) {
  const usable = samples.filter((sample) => sample[field] && sample.processes.length);
  if (!usable.length) return el("p", { class: "muted" }, `No ${label.toLowerCase()} data.`);
  const peaks = usable.flatMap((sample) => sample.processes.map((entry) => entry.peak));
  const f = frame(470, 260, niceMax(Math.max(...usable.map((sample) => sample[field]))), niceMax(Math.max(...peaks)), label, (v) => (v / 1e9).toFixed(1) + " GB");
  for (const sample of usable) {
    for (const entry of sample.processes) {
      const dot = svg("circle", { cx: f.x(sample[field]), cy: f.y(entry.peak), r: 5, fill: colors.get(entry.name), stroke: "var(--surface)", "stroke-width": 2 });
      dot.addEventListener("pointermove", (event) => showTip(event, [
        [entry.name, gb(entry.peak), colors.get(entry.name)],
        [label.toLowerCase(), number(sample[field])],
        ["sample", sample.id],
      ]));
      dot.addEventListener("pointerleave", hideTip);
      f.chart.append(dot);
    }
  }
  return f.chart;
}

/* Where time goes in each sample. */
function timeBars(samples) {
  const usable = samples.filter((sample) => sample.phases.length);
  if (!usable.length) return el("p", { class: "muted" }, "No phase timings.");
  const parts = (sample) => {
    const parents = new Set(sample.phases.filter((phase) => phase.parent).map((phase) => phase.parent));
    const output = [];
    for (const phase of sample.phases) {
      if (!phase.parent && !parents.has(phase.name)) output.push(phase);
      if (phase.parent) output.push(phase);
    }
    return output;
  };
  const names = [...new Set(usable.flatMap((sample) => parts(sample).map((phase) => phase.name)))];
  const colors = colorMap(names);
  const total = (sample) => parts(sample).reduce((sum, phase) => sum + phase.seconds, 0);
  const xMax = niceMax(Math.max(...usable.map(total)));
  const row = 30;
  const left = 200;
  const width = 960;
  const chart = svg("svg", { viewBox: `0 0 ${width} ${usable.length * row + 24}`, class: "chart" });
  const x = (value) => (value / xMax) * (width - left - 24);
  usable.forEach((sample, index) => {
    const top = index * row + 4;
    chart.append(svg("text", { x: left - 10, y: top + 15, "text-anchor": "end" }, sample.id.length > 26 ? "…" + sample.id.slice(-25) : sample.id));
    let offset = 0;
    for (const phase of parts(sample)) {
      const bar = svg("rect", { x: left + x(offset), y: top, width: Math.max(x(phase.seconds) - 2, 1), height: 20, rx: 3, fill: colors.get(phase.name) });
      bar.addEventListener("pointermove", (event) => showTip(event, [[phase.name, secs(phase.seconds), colors.get(phase.name)], ["sample", sample.id]]));
      bar.addEventListener("pointerleave", hideTip);
      chart.append(bar);
      offset += phase.seconds;
    }
  });
  for (let tick = 0; tick <= 4; tick++) {
    const value = (xMax / 4) * tick;
    chart.append(svg("text", { x: left + x(value), y: usable.length * row + 18, "text-anchor": "middle" }, value.toFixed(0) + " s"));
  }
  return el("div", {}, el("div", { class: "scroll" }, chart), legend([...colors]));
}

function cards(entries) {
  return el("div", { class: "cards" }, ...entries.map(([value, label]) => el("div", { class: "card" }, el("b", {}, value), el("span", {}, label))));
}

function table(headers, rows, numeric = []) {
  const head = el("tr", {}, ...headers.map((header, index) => el("th", { class: numeric.includes(index) ? "num" : "" }, header)));
  const body = rows.map((row) => el("tr", {}, ...row.map((cell, index) => el("td", { class: numeric.includes(index) ? "num" : "" }, cell))));
  return el("div", { class: "scroll" }, el("table", {}, el("thead", {}, head), el("tbody", {}, ...body)));
}

function metricsTable(dataset) {
  const rows = Object.entries(dataset.metrics).map(([name, values]) => {
    const shown = Object.entries(values).filter(([key]) => key !== "n").map(([key, value]) => `${key} ${metricValue(name, key, value)}`).join(" · ");
    return [name, String(values.n), shown];
  });
  for (const [name, reason] of Object.entries(dataset.not_applicable)) rows.push([name, "n/a", reason]);
  return table(["Metric", "n", "Values"], rows, [1]);
}

function samplesTable(samples) {
  const rows = samples.map((sample) => [
    sample.id,
    sample.status === "success" ? "ok" : "failed",
    number(sample.audio_minutes),
    sample.scores.wer === undefined ? "—" : (sample.scores.wer * 100).toFixed(1) + "%",
    secs(sample.wall_seconds),
    sample.memory_edges.start ? gb(sample.memory_edges.start) : "—",
    sample.memory_edges.end ? gb(sample.memory_edges.end) : "—",
    sample.peak_rss_bytes ? gb(sample.peak_rss_bytes) : "—",
    number(sample.prompt_tokens),
  ]);
  return table(["Sample", "Status", "Audio (min)", "WER", "Time", "RAM start", "RAM end", "Peak RAM", "Prompt tokens"], rows, [2, 3, 4, 5, 6, 7, 8]);
}

function noteDetails(sample) {
  const panels = [];
  if (sample.note) panels.push(el("div", {}, el("b", {}, "Generated"), el("pre", {}, sample.note)));
  if (sample.reference_note) panels.push(el("div", {}, el("b", {}, "Reference"), el("pre", {}, sample.reference_note)));
  const status = sample.status === "success" ? null : el("p", { class: "failed" }, "Failed: " + sample.error);
  const link = el("a", { href: sample.link }, "Full run");
  return el("details", {}, el("summary", {}, sample.id), status, el("div", { class: "notes" }, ...panels), el("p", { class: "small" }, link));
}

function datasetSection(name, dataset) {
  const section = el("section");
  const samples = dataset.samples;
  section.append(el("h2", {}, name));
  section.append(el("p", { class: "muted" }, `${dataset.language} · input: ${dataset.input}`));
  section.append(cards([
    [`${dataset.n_run} / ${dataset.n_selected}`, "Samples run"],
    [String(dataset.n_failed), "Failed"],
    ...(dataset.metrics.wer ? [[metricValue("wer", "global", dataset.metrics.wer.global), "WER"]] : []),
    ...(dataset.metrics.real_time_factor ? [[dataset.metrics.real_time_factor.global.toFixed(2), "Real-time factor"]] : []),
    ...(dataset.metrics.peak_rss_bytes ? [[gb(dataset.metrics.peak_rss_bytes.max), "Peak RAM, pipeline"]] : []),
  ]));
  for (const [id, error] of Object.entries(dataset.failures)) {
    section.append(el("p", { class: "failed" }, `Failed · ${id}: ${error}`));
  }
  section.append(el("h3", {}, "Metrics"), metricsTable(dataset));
  section.append(el("h3", {}, "Time by phase"), timeBars(samples));

  const processes = [...new Set(samples.flatMap((sample) => sample.processes.map((entry) => entry.name)))];
  const colors = colorMap(processes);
  const run = wholeRun(samples);
  if (run.series.length) {
    const note = data.isolate_samples ? "Each sample ran in fresh processes." : "Models stayed loaded. Memory that keeps climbing across samples is being retained.";
    section.append(el("h3", {}, "RAM across the run"), el("p", { class: "muted small" }, note), memoryChart(run.series, run.windows, colors, "Seconds since the first sample", false));
  }
  const withMemory = samples.filter((sample) => sample.memory.length);
  if (withMemory.length) {
    const holder = el("div");
    const picker = el("select", { "aria-label": "Sample" }, ...withMemory.map((sample, index) => el("option", { value: index }, sample.id)));
    const draw = () => {
      const sample = withMemory[picker.value];
      holder.replaceChildren(memoryChart(sample.memory, sample.windows, colors, "Seconds since the sample started"));
    };
    picker.addEventListener("change", draw);
    draw();
    section.append(el("h3", {}, "RAM during a sample"), el("p", {}, picker), holder);
  }
  section.append(el("h3", {}, "Peak RAM against length"));
  section.append(el("p", { class: "muted small" }, data.isolate_samples ? "One dot per sample and process. A flat line means memory doesn't grow with the consultation." : "One dot per sample and process. Samples share processes, so order affects these peaks as well as length."));
  section.append(el("div", { class: "pair" },
    el("div", {}, scatter(samples, "audio_minutes", "Audio (min)", colors)),
    el("div", {}, scatter(samples, "prompt_tokens", "Prompt tokens", colors))));
  section.append(legend([...colors]));
  section.append(el("h3", {}, "Samples"), samplesTable(samples));
  section.append(el("h3", {}, "Notes"), ...samples.map(noteDetails));
  return section;
}

function environment() {
  const hardware = data.hardware;
  const accelerators = hardware.accelerators.map((entry) => `${entry.name} (${entry.type})`).join(", ") || "None detected";
  const compute = Object.entries(data.compute).map(([phase, devices]) => `${phase}: ${devices.join(", ")}`).join(" · ") || "Unknown";
  const memory = data.telemetry.machine_memory_bytes;
  const rows = [
    ["CPU", `${hardware.cpu}, ${hardware.logical_cpus} threads`],
    ["Memory", gb(hardware.memory_bytes)],
    ["Accelerators present", accelerators],
    ["Actually used", compute],
  ];
  if (memory.peak) rows.push(["Machine memory in use", `${gb(memory.baseline)} at rest, ${gb(memory.peak)} at peak, including everything else running`]);
  rows.push(["Versions", Object.entries(data.versions).filter(([key]) => key !== "platform").map(([key, value]) => `${key} ${value}`).join(" · ")]);
  return table(["", ""], rows);
}

function spansTable() {
  const rows = Object.entries(data.telemetry.spans).map(([name, span]) => [name, String(span.count), secs(span.mean_seconds), secs(span.total_seconds)]);
  return rows.length ? table(["Span", "Count", "Mean", "Total"], rows, [1, 2, 3]) : el("p", { class: "muted" }, "No traces exported.");
}

document.title = `${data.name} · evaluation report`;
root.append(
  el("h1", {}, data.name),
  el("p", { class: "muted" }, `${data.pipeline} · ${data.state} · ${new Date(data.created_at).toLocaleString()}`),
  ...Object.entries(data.datasets).map(([name, dataset]) => datasetSection(name, dataset)),
  el("h2", {}, "Environment"), environment(),
  el("h2", {}, "Inside the pipeline"),
  el("p", { class: "muted small" }, "Raw spans the pipeline exported. Metrics above use the harness's outside time and shared phase names."),
  spansTable(),
);
