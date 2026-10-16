// The sequencing model, in the browser. Same cost, gradient, and liability
// rules as code20261015.py. Investments are found by solving each agent's
// first-order condition in turn.

function hillDetection(a, K) {
  const Ka = K ** a;
  function p(x) {
    const xa = x ** a;
    return xa / (Ka + xa);
  }
  p.derivative = (x) => {
    if (x <= 0) {
      if (a > 1) return 0;
      if (a === 1) return 1 / K;
      return Infinity;
    }
    const xa = x ** a;
    return a * Ka * x ** (a - 1) / (Ka + xa) ** 2;
  };
  return p;
}

function evalNumber(text) {
  const value = Function(`"use strict"; return (${text.replaceAll("^", "**")});`)();
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`could not read the number '${text}'`);
  }
  return value;
}

function slopeFromTheRight(p) {
  // One-sided slopes at shrinking steps. A slope that keeps exploding is the
  // infinite Inada derivative; a slope that settles is the finite value at 0.
  const widths = [1e-3, 1e-5, 1e-7];
  let previous = null;
  let latest = NaN;
  for (const width of widths) {
    const right = p(width);
    const origin = p(0);
    if (!Number.isFinite(right) || !Number.isFinite(origin)) return Infinity;
    latest = (right - origin) / width;
    if (!Number.isFinite(latest)) return Infinity;
    if (previous !== null && latest > 1000 && latest > previous * 5) return Infinity;
    previous = latest;
  }
  return latest;
}

function numericalSlope(p, x) {
  if (x < 0) return Infinity;
  if (!(x > 0)) return slopeFromTheRight(p);
  // Central step of order machine-epsilon^(1/3), kept inside the positive domain,
  // then one Richardson step. That is accurate for a smooth detection function
  // and stays defined when the slope blows up at zero.
  let step = (Number.EPSILON ** (1 / 3)) * Math.max(x, 1e-3);
  if (!(x - step > 0)) step = x / 2;
  const slope = (width) => {
    const right = p(x + width);
    const left = p(x - width);
    if (!Number.isFinite(right) || !Number.isFinite(left)) return NaN;
    return (right - left) / (2 * width);
  };
  const coarse = slope(step);
  const fineWidth = step / 2;
  const fine = x - fineWidth > 0 ? slope(fineWidth) : NaN;
  if (Number.isFinite(coarse) && Number.isFinite(fine)) return (4 * fine - coarse) / 3;
  if (Number.isFinite(fine)) return fine;
  if (Number.isFinite(coarse)) return coarse;
  const forward = Math.min(step, x);
  const rise = p(x + forward) - p(x);
  return Number.isFinite(rise) ? rise / forward : Infinity;
}

function compileDetection(source, agent) {
  const text = source.trim().replaceAll("^", "**");
  if (!text) throw new Error(`agent ${agent} has an empty detection function`);
  const hill = text.match(/^hill\(\s*([^,]+)\s*,\s*([^)]+?)\s*\)$/i);
  if (hill) {
    const exponent = evalNumber(hill[1]);
    const half = evalNumber(hill[2]);
    if (exponent <= 0 || half <= 0) throw new Error(`agent ${agent}: Hill(a, K) needs a > 0 and K > 0`);
    return hillDetection(exponent, half);
  }
  let raw;
  try {
    raw = Function("x", "exp", "log", "sqrt", "sin", "cos", "abs", "min", "pi", "E", `"use strict"; return (${text});`);
  } catch (error) {
    throw new Error(`agent ${agent}: ${error.message}`);
  }
  const call = (x) => raw(x, Math.exp, Math.log, Math.sqrt, Math.sin, Math.cos, Math.abs, Math.min, Math.PI, Math.E);
  function p(x) {
    let value = call(x);
    if (typeof value === "function") value = value(x);
    if (!Number.isFinite(value)) return value;
    return Math.min(1, Math.max(0, value));
  }
  p.derivative = (x) => numericalSlope(p, x);
  for (const point of [0, 0.5, 1]) {
    let value;
    try {
      value = p(point);
    } catch (error) {
      throw new Error(`agent ${agent}: failed at x = ${point} (${error.message})`);
    }
    if (!Number.isFinite(value)) {
      throw new Error(`agent ${agent}: at x = ${point} the value is ${value}`);
    }
  }
  if (!Number.isFinite(p.derivative(0.5))) {
    throw new Error(`agent ${agent}: the derivative is not finite at x = 0.5`);
  }
  return p;
}

// Short caution when the expression, before clipping, misses the maintained
// assumptions: p(0) = 0, values in [0, 1], strictly increasing, strictly
// concave, and infinite slope at zero. Empty when it complies or cannot be read.
function detectionCaution(source) {
  let raw;
  try {
    raw = compileDetection(source, 1);
  } catch (error) {
    return "";
  }
  // compileDetection clips. Recover the shape from a dense sample of the
  // clipped function, and also re-evaluate the source without the clip.
  const unclipped = unclippedDetection(source);
  if (!unclipped) return "";
  const notes = [];
  const origin = unclipped(0);
  if (!Number.isFinite(origin) || Math.abs(origin) > 1e-6) notes.push("p(0) ≠ 0");
  let outside = false;
  let dropped = false;
  let rose = false;
  let plateau = false;
  let previous = origin;
  for (let k = 1; k <= 24; k += 1) {
    const value = unclipped(k / 24);
    if (!Number.isFinite(value) || value < -1e-6 || value > 1 + 1e-6) outside = true;
    if (Number.isFinite(value) && Number.isFinite(previous)) {
      if (value < previous - 1e-4) dropped = true;
      else if (value > previous + 1e-4) rose = true;
      else if (rose) plateau = true;
    }
    previous = value;
  }
  if (outside) notes.push("leaves [0, 1]");
  if (dropped || plateau || !rose) notes.push("not strictly increasing");
  const slopes = [];
  for (let k = 1; k <= 12; k += 1) {
    const slope = numericalSlope(unclipped, k / 12);
    if (Number.isFinite(slope)) slopes.push(slope);
  }
  let slopeRose = false;
  let slopeFell = false;
  let slopeFlat = false;
  for (let i = 1; i < slopes.length; i += 1) {
    const tolerance = 1e-3 * Math.max(1, Math.abs(slopes[i - 1]));
    if (slopes[i] > slopes[i - 1] + tolerance) slopeRose = true;
    else if (slopes[i] < slopes[i - 1] - tolerance) slopeFell = true;
    else if (Math.abs(slopes[i]) <= tolerance) slopeFlat = true;
  }
  if (slopeRose || slopeFlat || !slopeFell) notes.push("not strictly concave");
  if (Number.isFinite(slopeNearZero(unclipped))) notes.push("finite slope at 0");
  return notes.join("; ");
}

function unclippedDetection(source) {
  const text = source.trim().replaceAll("^", "**");
  if (!text) return null;
  const hill = text.match(/^hill\(\s*([^,]+)\s*,\s*([^)]+?)\s*\)$/i);
  if (hill) {
    const exponent = evalNumber(hill[1]);
    const half = evalNumber(hill[2]);
    if (exponent <= 0 || half <= 0) return null;
    return hillDetection(exponent, half);
  }
  let raw;
  try {
    raw = Function("x", "exp", "log", "sqrt", "sin", "cos", "abs", "min", "pi", "E", `"use strict"; return (${text});`);
  } catch (error) {
    return null;
  }
  const call = (x) => raw(x, Math.exp, Math.log, Math.sqrt, Math.sin, Math.cos, Math.abs, Math.min, Math.PI, Math.E);
  return (x) => {
    let value = call(x);
    if (typeof value === "function") value = value(x);
    return value;
  };
}

function slopeNearZero(p) {
  const widths = [1e-2, 1e-4, 1e-6];
  const slopes = widths.map((width) => {
    const rise = p(width) - p(0);
    return Number.isFinite(rise) ? rise / width : Infinity;
  });
  if (slopes.some((slope) => !Number.isFinite(slope))) return Infinity;
  if (slopes[2] > slopes[1] * 1.05 && slopes[1] > slopes[0] * 1.05) return Infinity;
  return slopes[2];
}

// True when the detection slope is decreasing, so the partial cost has one root.
function slopeFalls(detection) {
  let previous = Infinity;
  for (let k = 0; k <= 16; k += 1) {
    const point = k === 0 ? 1e-6 : k / 16;
    const slope = detection.derivative(point);
    if (!Number.isFinite(slope)) continue;
    if (slope > previous + 1e-4 * Math.max(1, Math.abs(previous))) return false;
    previous = slope;
  }
  return true;
}

function rootCoordinate(x, i, gradient) {
  const saved = x[i];
  const partial = (value) => {
    x[i] = value;
    return gradient(x)[i];
  };
  const high = partial(1);
  let result = saved;
  if (Number.isFinite(high) && high <= 0) result = 1;
  else {
    const low = partial(0);
    if (Number.isFinite(low) && low >= 0) result = 0;
    else {
      let lo = 0;
      let hi = 1;
      for (let step = 0; step < 56; step += 1) {
        const mid = (lo + hi) / 2;
        const value = partial(mid);
        if (Number.isFinite(value) && value > 0) hi = mid;
        else lo = mid;
      }
      result = (lo + hi) / 2;
      // A root this close to zero is an artefact of treating a finite slope as infinite.
      if (result < 1e-9) {
        const nearby = partial(1e-8);
        if (Number.isFinite(nearby) && nearby >= 0) result = 0;
      }
    }
  }
  x[i] = saved;
  return result;
}

function bracketCoordinate(x, i, cost) {
  const saved = x[i];
  const at = (value) => {
    x[i] = value;
    return cost(x);
  };
  const steps = 48;
  let bestValue = 0;
  let bestCost = Infinity;
  for (let k = 0; k <= steps; k += 1) {
    const value = k / steps;
    const current = at(value);
    if (current < bestCost) {
      bestCost = current;
      bestValue = value;
    }
  }
  const width = 1 / steps;
  let lo = Math.max(0, bestValue - width);
  let hi = Math.min(1, bestValue + width);
  const ratio = (Math.sqrt(5) + 1) / 2;
  for (let step = 0; step < 80 && hi - lo > 1e-14; step += 1) {
    const left = hi - (hi - lo) / ratio;
    const right = lo + (hi - lo) / ratio;
    if (at(left) < at(right)) hi = right;
    else lo = left;
  }
  x[i] = saved;
  return (lo + hi) / 2;
}

// Each pass solves every agent's first-order condition with the others held
// fixed, starting from the last agent. A falling slope makes that condition
// monotone, so bisection finds it; otherwise the coordinate is minimized directly.
function minimize(cost, gradient, dimension, start, decreasing) {
  const x = new Array(dimension).fill(0);
  if (start && start.length === dimension) {
    for (let i = 1; i < dimension; i += 1) x[i] = Math.min(1, Math.max(0, start[i] || 0));
  }
  let best = x.slice();
  let bestCost = cost(best);
  for (let pass = 0; pass < 80; pass += 1) {
    let shift = 0;
    for (let i = dimension - 1; i >= 1; i -= 1) {
      const next = decreasing && decreasing[i - 1] === false
        ? bracketCoordinate(x, i, cost)
        : rootCoordinate(x, i, gradient);
      shift = Math.max(shift, Math.abs(next - x[i]));
      x[i] = next;
    }
    const value = cost(x);
    if (value < bestCost) {
      bestCost = value;
      best = x.slice();
    }
    if (shift < 1e-13) break;
  }
  return best;
}

function firstOrderSatisfied(x, gradient) {
  const g = gradient(x);
  for (let i = 1; i < x.length; i += 1) {
    const component = g[i];
    if (x[i] <= 1e-12 && !(component < 0)) continue;
    if (x[i] >= 1 - 1e-12 && !(component > 0)) continue;
    if (Math.abs(component) <= 1e-6) continue;
    const trial = x.slice();
    trial[i] = Math.max(0, x[i] - 1e-6);
    const left = gradient(trial)[i];
    trial[i] = Math.min(1, x[i] + 1e-6);
    const right = gradient(trial)[i];
    if (!(left <= 0 && right >= 0)) return false;
  }
  return true;
}

function modelOf(pi, detections) {
  const n = detections.length;
  function failurePrefix(x) {
    const fails = [];
    for (let i = 1; i <= n; i += 1) fails.push(1 - detections[i - 1](x[i]));
    const prefix = [1];
    for (const fail of fails) prefix.push(prefix[prefix.length - 1] * fail);
    return [fails, prefix];
  }
  function cost(x) {
    const [, prefix] = failurePrefix(x);
    let investment = 0;
    for (let i = 1; i <= n; i += 1) {
      investment += ((1 - pi) + pi * prefix[i - 1]) * x[i];
    }
    return pi * prefix[n] + investment;
  }
  function costGradient(x) {
    const [fails, prefix] = failurePrefix(x);
    const continuation = new Array(n + 1);
    continuation[n] = 1;
    for (let i = n - 1; i >= 1; i -= 1) {
      continuation[i] = x[i + 1] + fails[i] * continuation[i + 1];
    }
    const jac = new Array(n + 1).fill(0);
    for (let i = 1; i <= n; i += 1) {
      const rho = pi * prefix[i - 1];
      const slope = detections[i - 1].derivative(x[i]);
      const reach = 1 - pi + rho;
      jac[i] = Number.isFinite(slope) ? reach - rho * slope * continuation[i] : (rho * continuation[i] > 0 ? -Infinity : reach);
    }
    return jac;
  }
  function losses(x) {
    const L = { 0: 0, [n]: 1 };
    for (let i = n; i > 1; i -= 1) L[i - 1] = x[i] + (1 - detections[i - 1](x[i])) * L[i];
    return L;
  }
  function phiStar(i, j, x) {
    const L = losses(x);
    if (j === "U" || i < j) return L[i] - L[i - 1];
    if (i === j) return -L[i - 1];
    return 0;
  }
  return { n, cost, costGradient, losses, phiStar, detections };
}

function solvePrepared(pi, detections, start) {
  if (!(pi > 0 && pi <= 1)) throw new Error("the prior has to lie in (0, 1]");
  const model = modelOf(pi, detections);
  const decreasing = detections.map(slopeFalls);
  let x = minimize(model.cost, model.costGradient, model.n + 1, start, decreasing);
  if (!firstOrderSatisfied(x, model.costGradient)) {
    const retry = minimize(model.cost, model.costGradient, model.n + 1, null, decreasing);
    if (model.cost(retry) < model.cost(x)) x = retry;
  }
  const total = model.cost(x);
  const L = model.losses(x);
  const star = [];
  for (let i = 1; i <= model.n; i += 1) {
    const row = [];
    for (let j = 1; j <= model.n; j += 1) row.push(model.phiStar(i, j, x));
    row.push(model.phiStar(i, "U", x));
    star.push(row);
  }
  const delta = {};
  for (let k = 2; k <= model.n; k += 1) {
    delta[k] = (1 / pi) * (total / model.n - (1 - pi) * x[k]);
  }
  delta[1] = -Object.values(delta).reduce((sum, value) => sum + value, 0);
  const fair = star.map((row, index) => row.map((value) => value + delta[index + 1]));
  const beliefs = [pi];
  for (let i = 1; i < model.n; i += 1) {
    const detection = detections[i - 1](x[i]);
    const denominator = 1 - beliefs[i - 1] * detection;
    beliefs.push(denominator === 0 ? null : beliefs[i - 1] * (1 - detection) / denominator);
  }
  const gradient = model.costGradient(x);
  const unsettled = [];
  if (!firstOrderSatisfied(x, model.costGradient)) {
    for (let i = 1; i <= model.n; i += 1) {
      if (x[i] > 1e-8 && x[i] < 1 - 1e-8 && Math.abs(gradient[i]) > 1e-4) unsettled.push(i);
    }
  }
  let exAnteSum = 0;
  for (let i = 0; i < x.length; i += 1) exAnteSum += x[i];
  return {
    pi,
    x: x.slice(1),
    detection: detections.map((p, index) => p(x[index + 1])),
    L: Array.from({ length: model.n }, (_, index) => L[index + 1]),
    belief: beliefs,
    ex_ante: x.slice(1).map((investment, index) => (
      index === 0 ? total - (1 - pi) * (exAnteSum - x[1]) : (1 - pi) * investment
    )),
    cost: total,
    fair_share: total / model.n,
    phi_star: star,
    phi_F: fair,
    warning: unsettled.length ? `The gradient is still large for agent ${unsettled.join(", ")}.` : null,
  };
}

// A first-best rule is determined by the undetected liabilities once independence
// and inactive-agent independence are imposed. The incentive gap is L_i, an agent
// who has failed pays the same whoever detects later, and an agent who has not yet
// acted pays the same whoever detected.
function phiDesigned(losses, undetected) {
  const n = losses.length;
  const cumulative = [];
  let covered = 0;
  undetected.forEach((liability) => {
    covered += liability;
    cumulative.push(covered);
  });
  const remainder = losses.map((loss, agent) => loss - cumulative[agent]);
  const inactive = new Array(n).fill(0);
  for (let agent = 1; agent < n; agent += 1) {
    const after = agent === n - 1 ? 0 : remainder[agent];
    inactive[agent] = remainder[agent - 1] - after;
  }
  return undetected.map((liability, agent) => {
    const row = [];
    for (let event = 0; event < n; event += 1) {
      if (event > agent) row.push(liability);
      else if (event === agent) row.push(liability - losses[agent]);
      else row.push(inactive[agent]);
    }
    row.push(liability);
    return row;
  });
}

function linspace(start, end, count) {
  return Array.from({ length: count }, (_, index) => start + (end - start) * index / (count - 1));
}

function solvePoint(pi, sources) {
  if (!sources.length) throw new Error("add at least one detection function");
  const detections = sources.map((source, index) => compileDetection(source, index + 1));
  return solvePrepared(pi, detections);
}

function solveSweep(sources) {
  if (!sources.length) throw new Error("add at least one detection function");
  const detections = sources.map((source, index) => compileDetection(source, index + 1));
  const priors = linspace(0.02, 1, 80);
  const investments = [];
  const cost = [];
  const exAnte = [];
  let start = null;
  priors.forEach((prior) => {
    const point = solvePrepared(prior, detections, start);
    investments.push(point.x);
    cost.push(point.cost);
    exAnte.push(point.ex_ante);
    start = [0, ...point.x];
  });
  const grid = linspace(0, 1, 200);
  return {
    priors,
    investments,
    cost,
    ex_ante: exAnte,
    detection_x: grid,
    detection_y: detections.map((p) => grid.map((x) => p(x))),
  };
}

const api = { solvePoint, solveSweep, phiDesigned, detectionCaution };
if (typeof window !== "undefined") window.Sequencing = api;
if (typeof module !== "undefined" && module.exports) module.exports = api;
