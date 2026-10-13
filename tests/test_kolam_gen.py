"""kolam_gen (Rust) against the Python reference answers and kolam_feed's independent checks.

Runs when the binary is built: cd kolam_gen && cargo build --release
"""
import json
import subprocess
from pathlib import Path

import pytest

from kolam_feed import verify

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "kolam_gen" / "target" / "release" / "kolam_gen"
pytestmark = pytest.mark.skipif(not BIN.exists(), reason="kolam_gen not built (cargo build --release)")


def test_geometry_check_matches_the_python_reference(tmp_path, kolams):
    """Same kolams (21 failing) -> same verdict and stats as the Python generator's answers."""
    f = tmp_path / "in.jsonl"
    f.write_text("".join(json.dumps({"encoding": k["encoding"]}) + "\n" for k in kolams))
    out = subprocess.run([BIN, "check", f], capture_output=True, text=True, check=True).stdout.splitlines()
    for k, line in zip(kolams, out, strict=True):
        r = json.loads(line)
        assert (r["ok"], r["n_strokes"], r["n_crossings"]) == (k["ok"], k["n_strokes"], k["n_crossings"])
        assert r["min_dot"] == pytest.approx(k["min_dot"], abs=1e-3)
        if k["min_strand"] is None:
            assert r["min_strand"] is None
        else:
            assert r["min_strand"] == pytest.approx(k["min_strand"], abs=1e-3)
        assert r["length"] == pytest.approx(k["length"], abs=1e-3)


def test_generated_encodings_pass_python_checks(tmp_path):
    subprocess.run([BIN, "generate", "-n", "200", "-o", tmp_path, "--shard", "120"], check=True, capture_output=True)
    r = verify(tmp_path, n=60, log=lambda *_: None)
    assert r["ok"], r["examples"]
    assert r["kolams"] == 200 and set(r["dist"]["system"]) == {"random", "substitution", "switching", "wfc"}
    # resume: finished shards are kept, the result is the same
    before = (tmp_path / "shard_00000.jsonl").read_text()
    (tmp_path / "shard_00001.jsonl").unlink()
    subprocess.run([BIN, "generate", "-n", "200", "-o", tmp_path, "--shard", "120"], check=True, capture_output=True)
    assert (tmp_path / "shard_00000.jsonl").read_text() == before
    assert verify(tmp_path, n=0)["kolams"] == 200


def test_big_kolam(tmp_path):
    out = tmp_path / "big.jsonl"
    subprocess.run([BIN, "big", "--system", "switching", "--size", "40", "-o", out], check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert len(r["encoding"]) == 2 * r["rows"] + 1 and r["rows"] >= 30
