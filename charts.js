/* Tiny dependency-free SVG line chart with hover tooltips (works offline, no CDN needed). */
(function () {
  function niceStep(raw) {
    const pow = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) if (m * pow >= raw) return m * pow;
    return 10 * pow;
  }

  /**
   * lineChart(element, {labels:[...], series:[{name,color,values:[...],width}], points:true, xTickEvery:1})
   */
  window.lineChart = function (el, opts) {
    const { labels, series, points = true, xTickEvery = 1, yTitle = "", xTitle = "" } = opts;
    const W = 640, H = 270, L = 56, R = 14, T = 14, B = 38;
    const all = series.flatMap(s => s.values);
    if (!all.length) { el.innerHTML = '<div class="empty">No data</div>'; return; }
    let min = Math.min(...all), max = Math.max(...all);
    if (min === max) { min -= 1; max += 1; }
    const step = niceStep((max - min) / 4);
    const lo = Math.floor(min / step) * step, hi = Math.ceil(max / step) * step;
    const n = labels.length;
    const x = i => L + (n === 1 ? (W - L - R) / 2 : i * (W - L - R) / (n - 1));
    const y = v => T + (hi - v) / (hi - lo) * (H - T - B);

    let svg = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">`;
    for (let v = lo; v <= hi + step / 2; v += step) {
      const yy = y(v);
      svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/>`;
      svg += `<text x="${L - 8}" y="${yy + 4}" text-anchor="end">${Math.round(v * 100) / 100}</text>`;
    }
    labels.forEach((lab, i) => {
      if (i % xTickEvery === 0 || i === n - 1) {
        svg += `<text x="${x(i)}" y="${H - B + 16}" text-anchor="middle">${lab}</text>`;
      }
    });
    if (xTitle) svg += `<text class="axis-title" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${xTitle}</text>`;
    if (yTitle) svg += `<text class="axis-title" transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${yTitle}</text>`;
    series.forEach(s => {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.2}" stroke-linejoin="round" stroke-linecap="round"/>`;
      if (points) s.values.forEach((v, i) => { svg += `<circle cx="${x(i)}" cy="${y(v)}" r="3.2" fill="#fff" stroke="${s.color}" stroke-width="2"/>`; });
    });
    svg += `<line id="hover-line" class="grid" y1="${T}" y2="${H - B}" x1="0" x2="0" style="display:none;stroke:#9db4ee"/></svg>`;
    el.innerHTML = svg;

    // hover tooltip: find the nearest x position and list every series value there
    const svgEl = el.querySelector("svg"), line = svgEl.querySelector("#hover-line"), tip = document.getElementById("tooltip");
    svgEl.addEventListener("mousemove", ev => {
      const rect = svgEl.getBoundingClientRect();
      const px = (ev.clientX - rect.left) * (W / rect.width);
      let idx = Math.round((px - L) / ((W - L - R) / Math.max(n - 1, 1)));
      idx = Math.max(0, Math.min(n - 1, idx));
      line.setAttribute("x1", x(idx)); line.setAttribute("x2", x(idx)); line.style.display = "";
      tip.innerHTML = `<b>${opts.tooltipTitle ? opts.tooltipTitle(idx) : labels[idx]}</b>` +
        series.map(s => `<div><span style="color:${s.color}">●</span> ${s.name}: ${s.values[idx]}</div>`).join("");
      tip.classList.remove("hidden");
      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 230) + "px";
      tip.style.top = (ev.clientY + 14) + "px";
    });
    svgEl.addEventListener("mouseleave", () => { line.style.display = "none"; tip.classList.add("hidden"); });
  };
})();
