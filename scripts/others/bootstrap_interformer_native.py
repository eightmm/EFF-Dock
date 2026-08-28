#!/usr/bin/env python3
"""Materialize Interformer's pinned non-Python tools below its model root."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


PACKAGES = {
    "boost_headers": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "libboost-headers-1.84.0-ha770c72_5.conda",
        "85257c026772e47054683aa466c358fbd229d54055f42126974a77d577b0192f",
    ),
    "boost": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "libboost-1.84.0-h0ccab89_5.conda",
        "f66f16ce2402eedb58f4cf1a09b4edfab644e540bb287fe65a7d52f529fe00d1",
    ),
    "openbabel": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "openbabel-3.1.1-py312hbfe4552_9.conda",
        "c1dda107511a20b043dfe94c562cf607833cbd29ffcf612ed4b65b5b04fa8a24",
    ),
    "reduce": (
        "https://conda.anaconda.org/bioconda/linux-64/"
        "reduce-4.14-py312h719dbc0_2.tar.bz2",
        "0045dd56d02d563695bcfb8fc2581e8a3d79c34b1585359cfad6af03d0a38806",
    ),
    "libstdcxx": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "libstdcxx-ng-14.1.0-hc0a3c3a_0.conda",
        "88c42b388202ffe16adaa337e36cf5022c63cf09b0405cf06fc6aeacccbe6146",
    ),
    "libgcc": (
        "https://conda.anaconda.org/conda-forge/linux-64/"
        "libgcc-ng-14.1.0-h77fa898_0.conda",
        "b8e869ac96591cda2704bf7e77a301025e405227791a0bddf14a3dac65125538",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, expected_sha256: str, cache: Path) -> Path:
    destination = cache / url.rsplit("/", 1)[-1]
    if destination.is_file() and sha256(destination) == expected_sha256:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    actual = sha256(partial)
    if actual != expected_sha256:
        raise ValueError(
            f"Native artifact digest mismatch: {partial.name} "
            f"expected={expected_sha256} actual={actual}"
        )
    partial.replace(destination)
    return destination


def extract(package: Path, destination: Path) -> None:
    if package.suffixes[-2:] == [".tar", ".bz2"]:
        with tarfile.open(package, "r:bz2") as archive:
            archive.extractall(destination, filter="data")
        return
    with zipfile.ZipFile(package) as archive:
        archive.extractall(destination)
    payloads = list(destination.glob("pkg-*.tar.zst"))
    if len(payloads) != 1:
        raise ValueError(f"Expected one conda payload in {package}, found {payloads}")
    if shutil.which("zstd") is None:
        raise RuntimeError("zstd is required to unpack pinned .conda native artifacts")
    subprocess.run(
        ["tar", "--zstd", "-xf", str(payloads[0]), "-C", str(destination)],
        check=True,
    )


def copy_file(source: Path, destination: Path, *, executable: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(), destination)
    if executable:
        destination.chmod(destination.stat().st_mode | 0o111)


def link(target: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.readlink() == Path(target):
        return
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(target)


def verify(native_root: Path) -> None:
    required_files = [
        native_root / "reduce",
        native_root / "obrms.real",
        native_root / "obrms-lib/libopenbabel.so.7.0.0",
        native_root / "lib/libboost_system.so.1.84.0",
        native_root / "lib/libstdc++.so.6.0.33",
        native_root / "lib/libgcc_s.so.1",
        native_root / "include/boost/version.hpp",
        native_root / "share/openbabel/3.1.0/atomtyp.txt",
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete Interformer native runtime: {missing}")
    if not (native_root / "reduce").stat().st_mode & 0o111:
        raise PermissionError(f"Reduce is not executable: {native_root / 'reduce'}")
    if not (native_root / "obrms.real").stat().st_mode & 0o111:
        raise PermissionError(f"obrms is not executable: {native_root / 'obrms.real'}")


def main() -> None:
    args = parse_args()
    model_root = args.model_root.resolve()
    native_root = model_root / "bin"
    if args.verify_only:
        verify(native_root)
        print(f"Interformer native runtime verified: {native_root}")
        return

    cache = model_root / ".cache/native"
    cache.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="extract_", dir=cache) as temporary:
        temporary_root = Path(temporary)
        for name, (url, digest) in PACKAGES.items():
            package = download(url, digest, cache)
            package_root = temporary_root / name
            package_root.mkdir()
            extract(package, package_root)
            extracted[name] = package_root

        copy_file(extracted["reduce"] / "bin/reduce", native_root / "reduce", executable=True)
        reduce_dictionary = extracted["reduce"] / "reduce_wwPDB_het_dict.txt"
        if reduce_dictionary.is_file():
            copy_file(reduce_dictionary, native_root / "share/reduce_wwPDB_het_dict.txt")
        copy_file(
            extracted["openbabel"] / "bin/obrms",
            native_root / "obrms.real",
            executable=True,
        )
        copy_file(
            extracted["openbabel"] / "lib/libopenbabel.so.7.0.0",
            native_root / "obrms-lib/libopenbabel.so.7.0.0",
        )
        shutil.copytree(
            extracted["openbabel"] / "share/openbabel/3.1.0",
            native_root / "share/openbabel/3.1.0",
            dirs_exist_ok=True,
        )
        shutil.copytree(
            extracted["boost_headers"] / "include/boost",
            native_root / "include/boost",
            dirs_exist_ok=True,
        )
        copy_file(
            extracted["boost"] / "lib/libboost_system.so.1.84.0",
            native_root / "lib/libboost_system.so.1.84.0",
        )
        copy_file(
            extracted["libstdcxx"] / "lib/libstdc++.so.6.0.33",
            native_root / "lib/libstdc++.so.6.0.33",
        )
        copy_file(
            extracted["libgcc"] / "lib/libgcc_s.so.1",
            native_root / "lib/libgcc_s.so.1",
        )

    link("libopenbabel.so.7.0.0", native_root / "obrms-lib/libopenbabel.so.7")
    link("libboost_system.so.1.84.0", native_root / "lib/libboost_system.so")
    link("libstdc++.so.6.0.33", native_root / "lib/libstdc++.so.6")
    verify(native_root)
    print(f"Interformer native runtime materialized: {native_root}")


if __name__ == "__main__":
    main()
