"""Metrics on original valid A frame timestamps; FFT is supplemental only."""
import numpy as np

BANDS = ((0.5, 5), (5, 15), (15, 30), (30, 45), (45, 60))


def band_power(t, x):
    dt = 1 / 132
    tt = np.arange(t[0], t[-1], dt)
    if len(tt) < 20:
        return {}
    xx = np.interp(tt, t, x)
    w = np.hanning(len(tt))
    bins = np.fft.rfftfreq(len(tt), dt)
    power = 2 * abs(np.fft.rfft(w * (xx - xx.mean())))**2 / (len(xx) * np.dot(w,w))
    return {f"{lo:g}-{hi:g}": float(sum(power[(bins >= lo) & (bins < hi)])) for lo, hi in BANDS}


def metrics(t, target, model, resolution):
    if len(t) < 12:
        return dict(samples=len(t), fit=None, reason="fewer than 12 valid frames")
    reference_rms = float(np.std(target))
    error = model - target
    rmse = float(np.sqrt(np.mean(error**2)))
    return dict(samples=len(t), fit=float(100*(1-rmse/reference_rms)) if reference_rms > resolution else None,
                target_centered_rms_um=reference_rms, rmse_um=rmse, model_rms_um=float(np.sqrt(np.mean(model**2))),
                target_rms_um=float(np.sqrt(np.mean(target**2))), target_pp_um=float(np.ptp(target)),
                model_pp_um=float(np.ptp(model)), bias_um=float(np.mean(error)),
                correlation=float(np.corrcoef(target, model)[0, 1]) if min(reference_rms, np.std(model)) > 0 else None,
                dominant_peak_time_delta_s=float(t[np.argmax(abs(model - model.mean()))] -
                                                t[np.argmax(abs(target - target.mean()))]),
                band_target_um2=band_power(t, target), band_model_um2=band_power(t, model))


def evaluate(t, trace, src, record, trend, flat_um):
    ft = src["frame_host_time_s"]
    fit = {}
    for label, (lo, hi) in record["windows_s"].items():
        if not len(trace["t"]):
            fit[label] = dict(complete=False, raw={}, oscillation={}, physical_oscillation={})
            continue
        mask = (ft >= lo) & (ft <= hi) & src["frame_valid"] & (ft <= trace["t"][-1])
        complete = not np.any((ft >= lo) & (ft <= hi) & src["frame_valid"] & (ft > trace["t"][-1]))
        stamps = ft[mask]
        y = trace["physical_e_um"]
        physical = np.interp(stamps, trace["t"], y)
        observed = np.interp(stamps, trace["t"], y + trace["remaining"] * 1e6)
        drift = np.interp(stamps, t, trend)
        target = src["A_centered_um"][mask]
        zero = record["raw_visual_zero_um"]
        fit[label] = dict(complete=bool(complete),
                          raw=metrics(stamps, target + zero, observed + zero, flat_um),
                          oscillation=metrics(stamps, target-drift, observed-drift, flat_um),
                          physical_oscillation=metrics(stamps, target-drift, physical-drift, flat_um))
    return fit
