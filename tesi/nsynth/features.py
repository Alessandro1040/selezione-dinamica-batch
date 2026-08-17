#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estrazione delle features audio dal dataset NSynth (split jsonwav).

Uso:
  python3 features.py <dir_jsonwav> <prefix_output>

dove <dir_jsonwav> è la cartella estratta (es. nsynth-valid.jsonwav) e
<prefix_output> il prefisso dei file .npy generati (X, y, names, families).

Per ogni clip: spettrogramma mel (40 bande) in dB -> media e deviazione
standard nel tempo -> vettore di 80 features.
"""
import os, sys, json
import numpy as np
import librosa

SR = 16000
MEL_BANDS = 40
N_FFT = 1024
HOP = 512


def extract_features(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=MEL_BANDS,
                                       n_fft=N_FFT, hop_length=HOP)
    logS = librosa.power_to_db(S, ref=np.max)
    return np.concatenate([logS.mean(axis=1), logS.std(axis=1)])


def main():
    if len(sys.argv) != 3:
        sys.exit("uso: features.py <dir_jsonwav> <prefix_output>")
    jsondir, prefix = sys.argv[1], sys.argv[2]
    meta = os.path.join(jsondir, "examples.json")
    audio = os.path.join(jsondir, "audio")
    with open(meta, encoding="utf-8") as f:
        notes = json.load(f)          # dict note_str -> metadata

    items = []
    for note_str, md in notes.items():
        fam = md["instrument_family_str"]
        items.append((note_str, fam))
    print(f"note totali: {len(items)}")

    X, y, names = [], [], []
    families = sorted({fam for _, fam in items})
    fam2idx = {f: i for i, f in enumerate(families)}
    print("famiglie:", families)

    for i, (note_str, fam) in enumerate(items):
        wav = os.path.join(audio, note_str + ".wav")
        if not os.path.exists(wav):
            continue
        try:
            X.append(extract_features(wav))
            y.append(fam2idx[fam])
            names.append(note_str)
        except Exception as e:   # noqa: BLE001
            print("skip", note_str, e)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(items)}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    names = np.asarray(names)
    np.save(prefix + "_X.npy", X)
    np.save(prefix + "_y.npy", y)
    np.save(prefix + "_names.npy", names)
    with open(prefix + "_families.txt", "w") as f:
        f.write("\n".join(families))
    print(f"salvato: X={X.shape}, classi={len(families)}, ok")


if __name__ == "__main__":
    main()
