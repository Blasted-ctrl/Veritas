"""Adapters that turn on-disk datasets into a Veritas manifest.

Each adapter's only job is to enumerate media files and assign a correct
``(label, subject_id)`` pair.  Getting ``subject_id`` right is what makes the
downstream split identity-safe, so the dataset-specific knowledge lives here.

Supported sources:

* ``synthetic``     – generated fixtures (see :mod:`veritas.data.synthetic`).
* ``directory``     – a generic ``real/`` + ``fake/`` folder layout.
* ``faceforensics`` – FaceForensics++ image/frame layout.
* ``dfdc``          – DFDC video layout with ``metadata.json``.
* ``asvspoof``      – ASVspoof protocol-file layout (audio).

The real-dataset adapters are best-effort parsers of the public layouts; they
are exercised against fixtures in the tests and documented in the README, but
the heavyweight datasets themselves are gated and must be supplied by the user.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path

from veritas.data.manifest import FAKE, REAL, Sample

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def _iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _exts_for(modality: str) -> set[str]:
    return IMAGE_EXTS if modality == "image" else AUDIO_EXTS


# --------------------------------------------------------------------------- #
# Generic ``real/`` + ``fake/`` directory layout
# --------------------------------------------------------------------------- #
def from_directory(
    root: str | Path,
    modality: str,
    *,
    source: str = "directory",
    subject_from: Callable[[Path], str] | None = None,
) -> list[Sample]:
    """Load a dataset laid out as ``root/real/*`` and ``root/fake/*``.

    Args:
        root: Directory containing ``real`` and ``fake`` subfolders.
        modality: ``"image"`` or ``"audio"``.
        source: Value recorded in each sample's ``source`` field.
        subject_from: Optional callable mapping a file path to a subject id.
            Defaults to the filename stem up to the first underscore, so e.g.
            ``subject_0007_03.png`` groups under ``subject_0007``.
    """
    root = Path(root)
    exts = _exts_for(modality)
    subject_from = subject_from or _default_subject_from
    samples: list[Sample] = []
    for label, sub in ((REAL, "real"), (FAKE, "fake")):
        folder = root / sub
        if not folder.exists():
            continue
        for f in _iter_files(folder, exts):
            samples.append(
                Sample(
                    path=str(f),
                    label=label,
                    subject_id=subject_from(f),
                    modality=modality,
                    source=source,
                )
            )
    if not samples:
        raise FileNotFoundError(
            f"no {modality} files found under {root}/real or {root}/fake "
            f"(expected extensions: {sorted(exts)})"
        )
    return samples


def _default_subject_from(path: Path) -> str:
    """``subject_0007_03.png`` -> ``subject_0007`` (stem up to last ``_``)."""
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


# --------------------------------------------------------------------------- #
# FaceForensics++
# --------------------------------------------------------------------------- #
# Layout (extracted frames):
#   <root>/original_sequences/youtube/<quality>/frames/<id>/*.png      (REAL)
#   <root>/manipulated_sequences/<method>/<quality>/frames/<id1_id2>/* (FAKE)
# The subject identity for a manipulated clip is the *target* identity (the
# first id of ``<id1_id2>``); grouping on it keeps a person's real and faked
# frames together in one split.
_FFPP_FAKE_DIRS = (
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection",
)


def from_faceforensics(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []

    real_root = root / "original_sequences"
    for clip_dir in _clip_dirs(real_root):
        sid = clip_dir.name  # e.g. "033"
        for f in _iter_files(clip_dir, IMAGE_EXTS):
            samples.append(_img(f, REAL, _ffpp_subject(sid), "faceforensics"))

    manip_root = root / "manipulated_sequences"
    for method in _FFPP_FAKE_DIRS:
        for clip_dir in _clip_dirs(manip_root / method):
            sid = clip_dir.name  # e.g. "033_097" -> target id 033
            for f in _iter_files(clip_dir, IMAGE_EXTS):
                samples.append(_img(f, FAKE, _ffpp_subject(sid), "faceforensics"))

    if not samples:
        raise FileNotFoundError(
            f"no FaceForensics++ frames found under {root}; expected "
            f"original_sequences/ and manipulated_sequences/ with extracted frames"
        )
    return samples


def _clip_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    # Clip directories are the leaf folders that directly contain image frames.
    out = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(c.suffix.lower() in IMAGE_EXTS for c in d.iterdir() if c.is_file()):
            out.append(d)
    return out


def _ffpp_subject(clip_id: str) -> str:
    # Target identity is the first numeric id of "<target>_<source>".
    return f"ffpp_{clip_id.split('_', 1)[0]}"


# --------------------------------------------------------------------------- #
# DFDC (Deepfake Detection Challenge)
# --------------------------------------------------------------------------- #
# Each part folder has a ``metadata.json`` mapping
#   "<file>.mp4" -> {"label": "REAL"|"FAKE", "original": "<src>.mp4"|null, ...}
# Frames are expected pre-extracted alongside as ``<file>/<frame>.png``.  A fake
# video's subject is its ``original`` (the source identity being impersonated),
# which groups the fake with the real it was derived from.
def from_dfdc(root: str | Path) -> list[Sample]:
    root = Path(root)
    samples: list[Sample] = []
    for meta_path in sorted(root.rglob("metadata.json")):
        part = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for fname, info in meta.items():
            label = FAKE if str(info.get("label", "")).upper() == "FAKE" else REAL
            origin = info.get("original") or fname
            subject = f"dfdc_{Path(origin).stem}"
            frame_dir = part / Path(fname).stem
            frames = list(_iter_files(frame_dir, IMAGE_EXTS)) if frame_dir.exists() else []
            for f in frames:
                samples.append(_img(f, label, subject, "dfdc"))
    if not samples:
        raise FileNotFoundError(
            f"no DFDC frames found under {root}; expected metadata.json files and "
            f"per-video extracted frame folders"
        )
    return samples


# --------------------------------------------------------------------------- #
# ASVspoof (audio anti-spoofing)
# --------------------------------------------------------------------------- #
# Protocol lines (LA): "<speaker> <utt-id> - <system-id> <bonafide|spoof>".
# Grouping on speaker id prevents a speaker's bona-fide and spoofed utterances
# from straddling splits.
_ASV_LINE = re.compile(r"^(?P<speaker>\S+)\s+(?P<utt>\S+)\s+\S+\s+\S+\s+(?P<key>bonafide|spoof)\b")


def from_asvspoof(
    root: str | Path,
    *,
    protocol: str | Path | None = None,
    audio_subdir: str = "flac",
) -> list[Sample]:
    root = Path(root)
    protocol_path = Path(protocol) if protocol else _find_asv_protocol(root)
    if protocol_path is None or not protocol_path.exists():
        raise FileNotFoundError(
            f"could not locate an ASVspoof protocol (.txt) file under {root}; pass protocol=<path> explicitly"
        )

    audio_root = root / audio_subdir
    samples: list[Sample] = []
    for line in protocol_path.read_text(encoding="utf-8").splitlines():
        m = _ASV_LINE.match(line.strip())
        if not m:
            continue
        utt = m.group("utt")
        label = REAL if m.group("key") == "bonafide" else FAKE
        audio = _resolve_audio(audio_root if audio_root.exists() else root, utt)
        if audio is None:
            continue
        samples.append(
            Sample(
                path=str(audio),
                label=label,
                subject_id=f"asv_{m.group('speaker')}",
                modality="audio",
                source="asvspoof",
            )
        )
    if not samples:
        raise FileNotFoundError(f"protocol {protocol_path} matched no audio files under {root}")
    return samples


def _find_asv_protocol(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("*.txt") if "protocol" in p.name.lower() or "trial" in p.name.lower()]
    if not candidates:
        candidates = list(root.rglob("*cm_protocol*")) + list(root.rglob("*trial*.txt"))
    return sorted(candidates)[0] if candidates else None


def _resolve_audio(audio_root: Path, utt: str) -> Path | None:
    for ext in (".flac", ".wav"):
        cand = audio_root / f"{utt}{ext}"
        if cand.exists():
            return cand
    hits = list(audio_root.rglob(f"{utt}.*"))
    return hits[0] if hits else None


def _img(path: Path, label: int, subject: str, source: str) -> Sample:
    return Sample(path=str(path), label=label, subject_id=subject, modality="image", source=source)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def load_source(source: str, modality: str, input_dir: str | Path) -> list[Sample]:
    """Load any supported real dataset by ``source`` name."""
    loaders = {
        "directory": lambda: from_directory(input_dir, modality),
        "faceforensics": lambda: from_faceforensics(input_dir),
        "dfdc": lambda: from_dfdc(input_dir),
        "asvspoof": lambda: from_asvspoof(input_dir),
    }
    if source not in loaders:
        raise ValueError(f"unknown source {source!r}; choose from {sorted(loaders) + ['synthetic']}")
    return loaders[source]()


__all__ = [
    "from_directory",
    "from_faceforensics",
    "from_dfdc",
    "from_asvspoof",
    "load_source",
    "IMAGE_EXTS",
    "AUDIO_EXTS",
]
