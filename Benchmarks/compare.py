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

Usage: python compare.py [--repeats 3] [--samples 1048576]
"""

import argparse
import datetime
import importlib
import importlib.metadata
import platform
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


def rms(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2.0)))


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
    if inverse is None:
        return None, fwd_ms, None
    return rms(x, y), fwd_ms, inv_ms


# --- benchmark entries -------------------------------------------------------
# Each returns (config_note, error, fwd_ms, inv_ms). Raises ImportError to skip.

def bench_cicuetea(mode, x, n, repeats):
    from NsgfCQT import NsgfCQT

    T = NsgfCQT(mode, sample_rate=FS, n_samples=n,
                frac=1 / PPO, f_min=F_MIN, f_max=F_MAX)
    note = f"{mode}, {T.n_bands} bands, {F_MIN:.0f}-{F_MAX:.0f} Hz"
    return (note, *timed_round_trip(T.forward, T.inverse, x, repeats))


def bench_librosa(x, n, repeats):
    import librosa

    n_bins = int(np.ceil(PPO * np.log2(F_MAX / F_MIN)))  # 80
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
    err, fwd_ms, inv_ms = timed_round_trip(
        T.encode, T.decode, xt, repeats=repeats)
    # rms() on tensors: redo against the float32 input explicitly
    y = T.decode(T.encode(xt))
    err = rms(xt.numpy().squeeze(), y.detach().numpy().squeeze())
    note = f"{n_octaves} octaves, float32, torch {torch.get_num_threads()} threads"
    return note, err, fwd_ms, inv_ms


def bench_nnaudio(x, n, repeats):
    import torch
    from nnAudio import features

    T = features.CQT(sr=FS, fmin=F_MIN,
                     n_bins=int(np.ceil(PPO * np.log2(F_MAX / F_MIN))),
                     bins_per_octave=PPO, verbose=False)
    xt = torch.from_numpy(x.astype(np.float32)).reshape(1, -1)
    _, fwd_ms, inv_ms = timed_round_trip(T.forward, None, xt, repeats)
    note = "forward only (no exact inverse), float32"
    return note, None, fwd_ms, inv_ms


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
        f"signal:   white noise, N = 2^{int(np.log2(n))} = {n} samples "
        f"@ {FS} Hz ({n / FS:.1f} s), best of {repeats} runs",
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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2 ** 20)
    args = ap.parse_args()
    n = args.samples

    print("## CQT library comparison\n")
    print("```")
    print(env_report(n, args.repeats))
    print("```\n")

    x = np.random.default_rng(42).standard_normal(n)

    benches = [
        ("CiCueTea (dense)", lambda: bench_cicuetea("dense", x, n, args.repeats)),
        ("CiCueTea (sparse)", lambda: bench_cicuetea("sparse", x, n, args.repeats)),
        ("librosa", lambda: bench_librosa(x, n, args.repeats)),
        ("nsgt (matrixform)", lambda: bench_nsgt(True, x, n, args.repeats)),
        ("nsgt (ragged)", lambda: bench_nsgt(False, x, n, args.repeats)),
        ("cqt-pytorch", lambda: bench_cqt_pytorch(x, n, args.repeats)),
        ("nnAudio", lambda: bench_nnaudio(x, n, args.repeats)),
    ]

    rows, skipped = [], []
    for name, fn in benches:
        try:
            note, err, fwd, inv = fn()
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
        rows.append((name, note, err_s, f"{fwd:.1f}", inv_s, xrt_s))
        print(f"  done: {name:<20s} err={err_s:<9s} "
              f"fwd={fwd:8.1f} ms  inv={inv_s:>8s} ms", file=sys.stderr)

    header = ("Implementation", "Configuration", "Round-trip RMS error",
              "Forward (ms)", "Inverse (ms)", "x realtime")
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(6)]
    fmt = lambda r: "| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
    print(fmt(header))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in rows:
        print(fmt(r))

    if skipped:
        print("\nSkipped:")
        for s in skipped:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
