"""Audio combat cue analysis using librosa (RMS loudness & onset bursts)."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from the_eye.schemas import FrameMetrics


def analyze_audio_cues(
    audio_path: Path,
    timeline: list[FrameMetrics],
    *,
    fps: float = 2.0,
    spike_zscore: float = 1.4,
) -> list[FrameMetrics]:
    """Align librosa RMS energy and onset strength onto video frame timestamps."""
    if not timeline or not audio_path.exists():
        return timeline

    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if y.size == 0:
        return timeline

    hop_length = 512
    rms_series = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    onset_series = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(rms_series)), sr=sr, hop_length=hop_length)

    window_half = max(0.25, 0.5 / max(fps, 0.5))
    frame_rms: list[float] = []
    frame_onset: list[float] = []

    for m in timeline:
        mask = (times >= m.timestamp_sec - window_half) & (times <= m.timestamp_sec + window_half)
        if np.any(mask):
            frame_rms.append(float(np.max(rms_series[mask])))
            onset_len = min(len(onset_series), len(mask))
            frame_onset.append(float(np.max(onset_series[:onset_len][mask[:onset_len]])))
        else:
            frame_rms.append(0.0)
            frame_onset.append(0.0)

    rms_arr = np.array(frame_rms, dtype=np.float64)
    onset_arr = np.array(frame_onset, dtype=np.float64)

    rms_mean, rms_std = float(np.mean(rms_arr)), float(np.std(rms_arr))
    onset_mean, onset_std = float(np.mean(onset_arr)), float(np.std(onset_arr))

    enriched: list[FrameMetrics] = []
    for idx, m in enumerate(timeline):
        r_val = frame_rms[idx]
        o_val = frame_onset[idx]
        r_z = (r_val - rms_mean) / rms_std if rms_std > 1e-6 else 0.0
        o_z = (o_val - onset_mean) / onset_std if onset_std > 1e-6 else 0.0
        is_spike = (r_val > 0.02) and (r_z >= spike_zscore or o_z >= spike_zscore)
        enriched.append(
            m.model_copy(
                update={
                    "audio_rms": round(r_val, 4),
                    "audio_onset": round(o_val, 3),
                    "audio_spike": bool(is_spike),
                }
            )
        )
    return enriched
