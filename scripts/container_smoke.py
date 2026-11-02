"""Exercise the production Compose stack using only Python's standard library."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

root = Path(__file__).resolve().parents[1]


def get_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


for _attempt in range(60):
    try:
        health = get_json("http://127.0.0.1:8000/health")
        if health["image_model"]:
            break
    except OSError:
        pass
    time.sleep(2)
else:
    raise RuntimeError("The production image detector did not become ready")

with urlopen("http://127.0.0.1:3000", timeout=30) as response:
    page = response.read()
    assert b'id="analyze"' in page
    assert b'id="media-file"' in page and b'type="file"' in page

for filename, content_type, modality in [
    ("generated-lakehouse.png", "image/png", "image"),
    ("generated-demo.mp4", "video/mp4", "video"),
]:
    boundary = "veritas-production-smoke-boundary"
    data = (root / "frontend/public/samples" / filename).read_bytes()
    body = (
        (
            f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f'Content-Type: {content_type}\r\n\r\n'
        ).encode()
        + data
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = Request(
        "http://127.0.0.1:8000/verify",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    assert result["modality"] == modality
    assert 0 <= result["fake_probability"] <= 1
    assert (
        result["model_info"]["artifact_sha256"]
        == "44cb205f596f2530686bf4895e3678f08f9483c926b6bdc0e02998d5b54978a2"
    )
    if modality == "video":
        assert result["frames_analyzed"] == 4
        assert len(result["frames"]) == 4
    print(filename, result["verdict"], result["fake_probability"], flush=True)
print("Production container smoke checks passed", flush=True)
