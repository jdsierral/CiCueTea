#!/usr/bin/env python3
"""Python-ecosystem CQT comparison for CiCueTea.

Measures round-trip reconstruction error and forward/inverse wall time for the
CQT implementations available on PyPI, against the CiCueTea Python reference.
Every run prints a full environment report first, so any pasted output carries
its own provenance (machine, OS, library versions).

Reading the numbers:
  * The reconstruction error is machine-independent: it measures whether the
    library's inverse actually reconstructs the signal (frame theory), not how
    fast the machine is. This is the durable column.
  * Wall times are machine-specific. The relative ordering is fairly stable
    across machines; the absolute milliseconds are not. Compare against the
    `results/` files, which archive runs on specific machines.

Missing libraries are skipped and reported, so a partial environment still
produces a valid (smaller) table: `pip install -r requirements.txt` for the
full set. The Gaborator (C++) is deliberately absent: it is AGPLv3/commercial,
so we neither vendor nor depend on it — download it from gaborator.com to
compare against it yourself.

Every run exercises both probes — white noise (strict exactness) and the
in-range Kaiser-windowed log sweep (realistic streaming use case) — prints
one table per probe, and auto-saves the report to
results/<date>-<machine>.md (latest run wins for the same day + machine).

Usage: python compare.py [--repeats N] [--samples M]
"""

import argparse
import datetime
import importlib
import importlib.metadata
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Python"))

FS = 48000
PPO = 12          # bands per octave
F_MIN = 100.0
F_MAX = 10000.0   # ~6.64 octaves; 80-81 bands at 12 bpo depending on convention
N_BINS = int(np.ceil(PPO * np.log2(F_MAX / F_MIN)))


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2.0)))


def make_signal(kind, n):
    """Test signals. All transforms here are linear, so the round-trip error
    is the input spectrum weighted by the per-frequency reconstruction
    failure:
      * noise — flat weighting, the complete/adversarial probe (default:
        nothing in the band can hide, incl. librosa's HF undersampling);
      * sweep — log sweep confined to the transform range (F_MIN-F_MAX):
        the less strict but audio-realistic use case. Confined deliberately:
        a CQT can never reach DC, and very low frequencies expose the
        interaction between block size and frequency range — a real limit,
        but one only longer blocks can address, identically for every
        strategy; we are comparing algorithms, not block-length gotchas.
        Windowed with a single full-length Kaiser (beta=9, ~60 dB edge
        attenuation, smooth derivatives) so onset/tail transients don't
        register: warmup is not the steady-state streaming condition.
    """
    if kind == "noise":
        return np.random.default_rng(42).standard_normal(n)
    from scipy.signal import chirp
    from scipy.signal.windows import kaiser

    t = np.arange(n) / FS
    x = chirp(t, f0=F_MIN, f1=F_MAX, t1=t[-1], method="logarithmic")
    return x * kaiser(n, beta=9.0)


def count_coefs(X):
    """Total stored coefficient values in a container (array, tensor, or
    nested lists thereof). Complex values count as one coefficient."""
    if hasattr(X, "numel"):        # torch tensor
        return int(X.numel())
    if isinstance(X, (list, tuple)):
        return sum(count_coefs(c) for c in X)
    return int(np.asarray(X).size)


def timed_round_trip(forward, inverse, x, repeats):
    """Best-of-N forward/inverse timing; error from the fastest run."""
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        X = forward(x)
        t1 = time.perf_counter()
        y = inverse(X) if inverse is not None else None
        t2 = time.perf_counter()
        fwd_ms, inv_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        if best is None or fwd_ms + inv_ms < best[0] + best[1]:
            best = (fwd_ms, inv_ms, y)
    fwd_ms, inv_ms, y = best
    n_coefs = count_coefs(X)
    if inverse is None:
        return None, fwd_ms, None, n_coefs
    return rms(x, y), fwd_ms, inv_ms, n_coefs


# --- benchmark entries -------------------------------------------------------
# Each returns (config_note, error, fwd_ms, inv_ms, n_coefs).
# Raises ImportError to skip.

def bench_cicuetea(mode, x, n, repeats):
    from NsgfCQT import NsgfCQT

    T = NsgfCQT(mode, sample_rate=FS, n_samples=n,
                frac=1 / PPO, f_min=F_MIN, f_max=F_MAX)
    note = f"{mode}, {T.n_bands} bands, {F_MIN:.0f}-{F_MAX:.0f} Hz"
    return (note, *timed_round_trip(T.forward, T.inverse, x, repeats))


def bench_librosa(x, n, repeats):
    import librosa

    n_bins = N_BINS # ~80
    kw = dict(sr=FS, fmin=F_MIN, bins_per_octave=PPO)
    fwd = lambda s: librosa.cqt(y=s, n_bins=n_bins, **kw)
    inv = lambda C: librosa.icqt(C=C, length=n, **kw)
    note = f"{n_bins} bins, {F_MIN:.0f}-{F_MAX:.0f} Hz, default hop"
    return (note, *timed_round_trip(fwd, inv, x, repeats))


def bench_nsgt(matrixform, x, n, repeats):
    from nsgt import NSGT, LogScale

    scale = LogScale(F_MIN, F_MAX, 79)  # +DC/Nyquist channels -> 81
    T = NSGT(scale=scale, fs=FS, Ls=n, real=True, matrixform=matrixform)
    note = (f"{'matrixform' if matrixform else 'ragged'}, 81 channels, "
            f"{F_MIN:.0f}-{F_MAX:.0f} Hz")
    return (note, *timed_round_trip(T.forward, T.backward, x, repeats))


def bench_cqt_pytorch(x, n, repeats):
    import cqt_pytorch
    import torch

    torch.set_default_device("cpu")
    n_octaves = 8
    T = cqt_pytorch.CQT(num_octaves=n_octaves, num_bins_per_octave=PPO,
                        sample_rate=FS, block_length=n)
    xt = torch.from_numpy(x.astype(np.float32)).reshape(1, 1, -1)
    err, fwd_ms, inv_ms, n_coefs = timed_round_trip(
        T.encode, T.decode, xt, repeats=repeats)
    # rms() on tensors: redo against the float32 input explicitly
    y = T.decode(T.encode(xt))
    err = rms(xt.numpy().squeeze(), y.detach().numpy().squeeze())
    note = f"{n_octaves} octaves, float32, torch {torch.get_num_threads()} threads"
    return note, err, fwd_ms, inv_ms, n_coefs


def bench_nnaudio(x, n, repeats):
    import torch
    from nnAudio import features

    # Deliberately CPU, like every other row: the comparison is algorithmic
    # and the target use case (real-time audio) cannot afford GPU round trips.
    # nnAudio's GPU/batch throughput is a different benchmark. See README.
    torch.set_default_device("cpu")
    T = features.CQT(sr=FS, fmin=F_MIN, n_bins=N_BINS, bins_per_octave=PPO, verbose=False)
    xt = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    _, fwd_ms, inv_ms, n_coefs = timed_round_trip(T.forward, None, xt, repeats)
    note = "forward only (no exact inverse), float32, magnitude bins"
    return note, None, fwd_ms, inv_ms, n_coefs


# --- environment report ------------------------------------------------------

def cpu_name():
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


def env_report(n, repeats):
    lines = [
        f"date:     {datetime.date.today().isoformat()}",
        f"machine:  {cpu_name()} ({platform.machine()})",
        f"os:       {platform.platform()}",
        f"python:   {platform.python_version()}",
        f"signal:   N = 2^{int(np.log2(n))} = {n} samples "
        f"@ {FS} Hz ({n / FS:.1f} s), best of {repeats} runs, "
        "both probes (noise + in-range sweep)",
    ]
    for pkg, dist in [("numpy", None), ("scipy", None), ("librosa", None),
                      ("nsgt", None), ("torch", None), ("nnAudio", None),
                      ("cqt_pytorch", "cqt-pytorch")]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                try:
                    ver = importlib.metadata.version(dist or pkg)
                except Exception:
                    ver = "?"
            lines.append(f"{pkg + ':':<10}{ver}")
        except ImportError:
            lines.append(f"{pkg + ':':<10}not installed")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parent).strip()
        lines.append(f"cicuetea: {sha}")
    except Exception:
        pass
    return "\n".join(lines)


# --- main --------------------------------------------------------------------

def run_suite(x, n, repeats):
    """Run every benchmark on signal x; returns (table_lines, skipped)."""
    benches = [
        ("CiCueTea (dense)", lambda: bench_cicuetea("dense", x, n, repeats)),
        ("CiCueTea (sparse)", lambda: bench_cicuetea("sparse", x, n, repeats)),
        ("librosa", lambda: bench_librosa(x, n, repeats)),
        ("nsgt (matrixform)", lambda: bench_nsgt(True, x, n, repeats)),
        ("nsgt (ragged)", lambda: bench_nsgt(False, x, n, repeats)),
        ("cqt-pytorch", lambda: bench_cqt_pytorch(x, n, repeats)),
        ("nnAudio", lambda: bench_nnaudio(x, n, repeats)),
    ]

    rows, skipped = [], []
    for name, fn in benches:
        try:
            note, err, fwd, inv, n_coefs = fn()
        except ImportError as e:
            skipped.append(f"{name}: not installed ({e.name})")
            continue
        except Exception as e:
            skipped.append(f"{name}: failed ({type(e).__name__}: {e})")
            continue
        fwd_only = inv is None
        xrt = (n / FS) * 1e3 / (fwd + (inv or 0.0))
        xrt_s = (f"{xrt:.1f}x" if xrt < 10 else f"{xrt:.0f}x") + \
                (" (fwd only)" if fwd_only else "")
        err_s = f"{err:.2e}" if err is not None else "n/a"
        inv_s = "n/a" if fwd_only else f"{inv:.1f}"
        coef_s = f"{n_coefs:.2e} ({n_coefs / n:.2f}x)"
        rows.append((name, note, err_s, coef_s, f"{fwd:.1f}", inv_s, xrt_s))
        print(f"  done: {name:<20s} err={err_s:<9s} coefs={coef_s:<18s} "
              f"fwd={fwd:8.1f} ms  inv={inv_s:>8s} ms", file=sys.stderr)

    header = ("Implementation", "Configuration", "Round-trip RMS error",
              "Coefficients (xN)", "Forward (ms)", "Inverse (ms)", "x realtime")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(7)]
    fmt = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    lines = [fmt(header),
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt(r) for r in rows]
    if skipped:
        lines += ["", "Skipped:"] + [f"  - {s}" for s in skipped]
    return lines


SIGNAL_TITLES = {
    "noise": "White noise — strict exactness probe (flat weighting, "
             "nothing in the band can hide)",
    "sweep": f"Log sweep {F_MIN:.0f}-{F_MAX:.0f} Hz, Kaiser(beta=9) — "
             "realistic in-range streaming use case",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2 ** 20)
    args = ap.parse_args()
    n = args.samples

    report = ["## CQT library comparison", "", "```",
              env_report(n, args.repeats), "```", ""]

    for kind in ("noise", "sweep"):
        print(f"-- {kind} --", file=sys.stderr)
        x = make_signal(kind, n)
        report += [f"### {SIGNAL_TITLES[kind]}", ""]
        report += run_suite(x, n, args.repeats)
        report += [""]

    text = "\n".join(report)
    print(text)

    slug = re.sub(r"[^a-z0-9]+", "-", cpu_name().lower()).strip("-")
    out = Path(__file__).resolve().parent / "results" / \
        f"{datetime.date.today().isoformat()}-{slug}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print(f"\nsaved: {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
