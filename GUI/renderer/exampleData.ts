// Seeded PRNG (mulberry32): the example file is identical every time it is saved.
function seeded(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Gaussian peaks at familiar metabolite shifts; "treated" raises two of them.
const PEAKS = [
  { ppm: 0.0, height: 2.0, width: 0.03 }, // reference (TSP / DSS)
  { ppm: 1.33, height: 1.4, width: 0.04, treated: 2.1 }, // lactate
  { ppm: 3.03, height: 1.0, width: 0.04 }, // creatine
  { ppm: 3.25, height: 0.8, width: 0.05 }, // choline
  { ppm: 4.05, height: 0.6, width: 0.04 },
  { ppm: 7.35, height: 0.3, width: 0.06, treated: 0.55 }, // aromatic
];

export function makeExampleCsv(samples = 12, points = 400): string {
  const random = seeded(42);
  const ppm = Array.from({ length: points }, (_, i) => 10 - (10.5 * i) / (points - 1));
  const header = ["Sample", "Group", "Age", ...ppm.map((p) => p.toFixed(4))];
  const rows = Array.from({ length: samples }, (_, s) => {
    const treated = s >= samples / 2;
    const values = ppm.map((p) => {
      let y = 0.002 + random() * 0.001;
      for (const peak of PEAKS) {
        const height = treated && peak.treated ? peak.treated : peak.height;
        y += height * Math.exp(-(((p - peak.ppm) / peak.width) ** 2) / 2) * (0.9 + random() * 0.2);
      }
      return y.toFixed(6);
    });
    const id = `s${String(s + 1).padStart(2, "0")}`;
    return [id, treated ? "Treated" : "Control", String(25 + Math.round(random() * 40)), ...values];
  });
  return [header, ...rows].map((r) => r.join(",")).join("\n") + "\n";
}
