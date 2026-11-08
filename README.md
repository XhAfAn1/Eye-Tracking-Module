# Eye Tracking Module — 4-Algorithm Comprehensive Benchmark Suite

An empirical, interactive benchmark suite comparing **4 state-of-the-art pupil detection algorithms** on a live camera stream with eyeball boundary caging, multi-algorithm bivariate polynomial calibration, interactive level-2 pointer testing, dynamic smooth pursuit tracking, and automated clinical reporting.

---

## The 4 Algorithms Evaluated

1. 🟡 **Orlosky Pupil Detector (Gold):** Darkest-point seed + multi-threshold contour extraction.
2. 🟢 **Pupil Labs Detector2D (Neon Green):** Contrast-adaptive Canny edge filtering + Ramanujan ellipse fitting.
3. 🔵 **Hybrid Fusion Detector (Cyan):** Pupil Labs primary with spatially-bounded adaptive fallback.
4. 🟣 **PuReST Tracker (Electric Pink):** World Champion on LPW Benchmark (Santini et al., CVIU/ETRA).

---

## The 5-Phase Benchmark Workflow

```
[Phase 0: Interactive Alignment]
   │  Adjust Eyeball Boundary Circle on live eye (W/A/S/D, +/-)
   │  Verify all 4 algorithms are locked -> Press [SPACE]
   ▼
[Phase 1: 100% Replicated Precision Calibration]
   │  Lissajous Warmup (4.0s) with glowing animated aura
   │  9-Point Screen Grid with Approach Ring, 360° sweeping cyan arc & quality flash
   │  8 Smooth Pursuit continuous tracking paths (Top, Mid, Bot, Diagonals)
   │  Simultaneously fits 4 independent Bivariate Degree-2 Polynomials on hundreds of points
   │  4-Point Validation pass measuring real-world accuracy
   │  Summary screen showing point quality and validation error
   ▼
[Phase 2: Level 2 Testing (Static Targets)]
   │  Validation targets appear across screen
   │  Inspect 4 live colored pointers (Gold, Green, Cyan, Pink)
   │  Fixate on target dot -> Press [SPACE] to Confirm
   ▼
[Phase 3: Dynamic Pursuit Tracking Test]
   │  Smooth pursuit moving dot traverses horizontal, vertical & diagonal paths
   │  Records real-time trajectory fidelity, tracking lag, and jitter
   ▼
[Phase 4 & 5: Analytics & Reporting]
   │  Generates 6-panel Matplotlib graphic: benchmark_results_charts.png
   │  Generates detailed clinical report: BENCHMARK_REPORT.md
   │  Exports complete telemetry data: benchmark_data_TIMESTAMP.json
   ▼
[In-App Results Podium & Telemetry Dashboard]
```

---

## Quick Start

Open terminal in this directory:
```powershell
cd "E:\Files and Other Things\Projects\Augmentative Communication System for Plegia\Eye Tracking Module - Benchmark Suite"
python benchmark.py
```

*Optional command line flags:*
```powershell
# Specify a camera index manually
python benchmark.py -c 1

# Run in a 1280x800 window instead of full-screen
python benchmark.py --windowed
```

---

## Controls Summary

| State | Keys | Action |
| :--- | :--- | :--- |
| **Phase 0: Alignment** | **`[W] / [A] / [S] / [D]`** or **Arrows** | Move circle cage (Up / Left / Down / Right) |
| | **`[+]` / `[-]`** | Resize circle radius |
| | **`[C]`** | Switch active camera device |
| | **`[Space]`** or **`[Enter]`** | **Start Benchmark** (proceeds to Phase 1) |
| **Phase 1: Calibration** | Look at shrinking circle & sweeping arc | Hands-free automated 9-point collection |
| **Phase 2: Static Testing**| **Look at Target Dot** | Inspect the 4 colored pointers in real time |
| | **`[Space]`** or **`[Enter]`** | **Confirm Fixation** & advance to next test point |
| **Phase 3: Dynamic Tracking**| **Follow moving dot** | Smooth pursuit continuous recording |
| **Phase 4: Results** | **`[R]`** | Re-run benchmark |
| | **`[Q]`** or **`ESC`** | Quit application |

---

## Generated Benchmark Deliverables

1. **`BENCHMARK_REPORT.md`**: Complete clinical comparison with rankings, accuracy tables, jitter metrics, and AAC communication suitability scores.
2. **`benchmark_results_charts.png`**: High-resolution 300 DPI 6-panel infographic:
   * *Panel 1:* Mean & 95th Percentile Gaze Error Bar Chart (px and visual degrees °).
   * *Panel 2:* 2D Target Landing Scatter Plot showing prediction clouds and error confidence ellipses.
   * *Panel 3:* Fixation Jitter / Stability Boxplot (showing dispersion & tremor).
   * *Panel 4:* Dynamic Smooth Pursuit Trajectory Overlay (target path vs 4 tracked paths).
   * *Panel 5:* Latency Distribution KDE with 30 FPS / 60 FPS frame budgets.
   * *Panel 6:* 5-Axis AAC Usability Radar / Spider Chart.
3. **`benchmark_data_TIMESTAMP.json`**: Complete raw frame-by-frame telemetry data for archival and research analysis.
