#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrazione delle feature chroma (12 classi di altezza) per il compito
di riconoscimento della nota su NSynth. Per ogni clip: chromagramma
(chroma_stft, 12 bin) -> media e deviazione standard nel tempo -> 24 features.

Etichette: pitch class = MIDI pitch mod 12 (0=C, 1=C#, ..., 11=B).

Uso:
  python3 features_nota.py <dir_data> <prefix_output>
"""
import os, sys, json
import numpy as np
import librosa

SR = 16000
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def chroma(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    C = librosa.feature.chroma_stft(y=y, sr=SR)
    return np.concatenate([C.mean(axis=1), C.std(axis=1)])  # 24 dim


def main():
    if len(sys.argv) != 3:
        sys.exit("uso: features_nota.py <dir_data> <prefix_output>")
    data, prefix = sys.argv[1], sys.argv[2]
    X, y = [], []
    for split in ("valid", "test"):
        jsondir = os.path.join(data, f"nsynth-{split}")
        with open(os.path.join(jsondir, "examples.json"), encoding="utf-8") as f:
            notes = json.load(f)
        Xs, ys = [], []
        for i, (note_str, md) in enumerate(notes.items()):
            wav = os.path.join(jsondir, "audio", note_str + ".wav")
            if os.path.exists(wav):
                Xs.append(chroma(wav))
                ys.append(md["pitch"] % 12)
            if (i + 1) % 4000 == 0:
                print(f"  {split} {i+1}/{len(notes)}")
        X.extend(Xs)
        y.extend(ys)
        np.save(f"{prefix}_{split}_X.npy", np.asarray(Xs, np.float32))
        np.save(f"{prefix}_{split}_y.npy", np.asarray(ys, np.int64))
        print(f"split {split}: {len(Xs)} clip salvate")
    with open(prefix + "_note.txt", "w") as f:
        f.write("\n".join(NOTE))
    print("OK")


if __name__ == "__main__":
    main()
