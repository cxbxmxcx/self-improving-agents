// Hand-built d3 chart primitives: line with markers, bar, and a verdict heatmap.
// Three charts total in the whole app, so these stay deliberately small.

import { creatorHex, escapeHtml, fmtScore } from "../format.js";

const d3 = window.d3;

const GRID = "#E6DDC4";
const INK = "#6B5F49";

function frame(el, height) {
  const width = el.clientWidth || 800;
  const margin = { top: 16, right: 24, bottom: 34, left: 48 };
  const svg = d3
    .select(el)
    .append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("display", "block");
  return { svg, width, height, margin };
}

// series: [{name, color, points: [{x, y, tooltip}]}]
export function lineChart(el, series, { height = 340, yDomain = [0, 1.05], refLine, xLabel = "", stars = [] } = {}) {
  el.innerHTML = "";
  const { svg, width, margin } = frame(el, height);
  const allX = series.flatMap((s) => s.points.map((p) => p.x));
  if (!allX.length) return;
  const x = d3
    .scaleLinear()
    .domain(d3.extent(allX))
    .range([margin.left, width - margin.right]);
  const y = d3
    .scaleLinear()
    .domain(yDomain)
    .range([height - margin.bottom, margin.top]);

  svg
    .append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x).ticks(Math.min(10, allX.length)).tickFormat(d3.format("d")))
    .call((g) => g.selectAll("text").attr("fill", INK).attr("font-size", 10))
    .call((g) => g.selectAll("line,path").attr("stroke", GRID));
  svg
    .append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6))
    .call((g) => g.selectAll("text").attr("fill", INK).attr("font-size", 10))
    .call((g) => g.selectAll("line,path").attr("stroke", GRID));
  if (xLabel)
    svg
      .append("text")
      .attr("x", width / 2)
      .attr("y", height - 4)
      .attr("text-anchor", "middle")
      .attr("font-size", 10)
      .attr("fill", INK)
      .text(xLabel);

  if (refLine !== undefined) {
    svg
      .append("line")
      .attr("x1", margin.left)
      .attr("x2", width - margin.right)
      .attr("y1", y(refLine))
      .attr("y2", y(refLine))
      .attr("stroke", "#97876A")
      .attr("stroke-dasharray", "3 4");
  }

  const lineGen = d3
    .line()
    .x((p) => x(p.x))
    .y((p) => y(p.y));

  const tip = d3
    .select(el)
    .append("div")
    .style("position", "fixed")
    .style("display", "none")
    .style("background", "rgba(44,44,42,0.95)")
    .style("color", "#fff")
    .style("padding", "6px 9px")
    .style("border-radius", "5px")
    .style("font-size", "11px")
    .style("pointer-events", "none")
    .style("white-space", "pre")
    .style("z-index", 40);

  for (const s of series) {
    svg
      .append("path")
      .datum(s.points)
      .attr("fill", "none")
      .attr("stroke", s.color)
      .attr("stroke-width", 2)
      .attr("d", lineGen);
    svg
      .append("g")
      .selectAll("circle")
      .data(s.points)
      .join("circle")
      .attr("cx", (p) => x(p.x))
      .attr("cy", (p) => y(p.y))
      .attr("r", 4.5)
      .attr("fill", s.color)
      .on("mousemove", (event, p) => {
        tip
          .style("display", "block")
          .style("left", `${event.clientX + 12}px`)
          .style("top", `${event.clientY + 10}px`)
          .text(p.tooltip ?? `${s.name}: ${fmtScore(p.y)}`);
      })
      .on("mouseleave", () => tip.style("display", "none"));
  }

  // Promoted-round stars, drawn on top.
  const star = d3.symbol().type(d3.symbolStar).size(180);
  svg
    .append("g")
    .selectAll("path.star")
    .data(stars)
    .join("path")
    .attr("class", "star")
    .attr("d", star)
    .attr("transform", (p) => `translate(${x(p.x)},${y(p.y)})`)
    .attr("fill", "#00A087")
    .attr("stroke", "#fff")
    .attr("stroke-width", 1);

  // Legend, top-right.
  const legend = svg
    .append("g")
    .attr("transform", `translate(${width - margin.right},${margin.top})`);
  series.forEach((s, i) => {
    const g = legend.append("g").attr("transform", `translate(0,${i * 16})`);
    g.append("circle").attr("r", 4).attr("cx", -60).attr("fill", s.color);
    g.append("text")
      .attr("x", -50)
      .attr("y", 3.5)
      .attr("font-size", 10.5)
      .attr("fill", INK)
      .text(s.name);
  });
}

// rows: [{x, y, colorBy, tooltip}] — one bar per row.
export function barChart(el, rows, { height = 340, yDomain = [0, 1.05], xLabel = "version" } = {}) {
  el.innerHTML = "";
  const { svg, width, margin } = frame(el, height);
  if (!rows.length) return;
  const x = d3
    .scaleBand()
    .domain(rows.map((r) => r.x))
    .range([margin.left, width - margin.right])
    .padding(0.25);
  const y = d3
    .scaleLinear()
    .domain(yDomain)
    .range([height - margin.bottom, margin.top]);

  svg
    .append("g")
    .attr("transform", `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(x))
    .call((g) => g.selectAll("text").attr("fill", INK).attr("font-size", 10))
    .call((g) => g.selectAll("line,path").attr("stroke", GRID));
  svg
    .append("g")
    .attr("transform", `translate(${margin.left},0)`)
    .call(d3.axisLeft(y).ticks(6))
    .call((g) => g.selectAll("text").attr("fill", INK).attr("font-size", 10))
    .call((g) => g.selectAll("line,path").attr("stroke", GRID));
  svg
    .append("text")
    .attr("x", width / 2)
    .attr("y", height - 4)
    .attr("text-anchor", "middle")
    .attr("font-size", 10)
    .attr("fill", INK)
    .text(xLabel);

  svg
    .append("g")
    .selectAll("rect")
    .data(rows.filter((r) => r.y !== null && r.y !== undefined))
    .join("rect")
    .attr("x", (r) => x(r.x))
    .attr("y", (r) => y(r.y))
    .attr("width", x.bandwidth())
    .attr("height", (r) => y(yDomain[0]) - y(r.y))
    .attr("rx", 2)
    .attr("fill", (r) => creatorHex(r.colorBy))
    .append("title")
    .text((r) => r.tooltip ?? `${r.x}: ${fmtScore(r.y)}`);
}

// Verdict heatmap: questions down, versions across, preference per cell.
export function heatmap(el, { questions, versions, cells }) {
  el.innerHTML = "";
  const colors = { left: "#00A087", right: "#3C5488", tie: "#B7A67F" };
  const rows = questions
    .map((q) => {
      const tds = versions
        .map((v) => {
          const pref = cells[`${q}|${v}`];
          const bg = colors[pref] ?? "transparent";
          const label = pref ?? "—";
          const fg = pref ? "#fff" : "#97876A";
          return `<td style="background:${bg};color:${fg};text-align:center;font-weight:600">${escapeHtml(label)}</td>`;
        })
        .join("");
      return `<tr><td>${escapeHtml(q)}</td>${tds}</tr>`;
    })
    .join("");
  el.innerHTML = `
    <table class="data">
      <thead><tr><th>question</th>${versions
        .map((v) => `<th style="text-align:center">v${v}</th>`)
        .join("")}</tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="panel-caption" style="margin-top:var(--sp-2)">
      LEFT (teal) = candidate won · RIGHT (slate) = reference won · TIE (gray) = neither.</p>`;
}
