# OSS Scanner build image (https://github.com/anthropics/oss-scanner).
# The scanner builds this with the repository as the build context, then runs
# its audit offline, so every crate, toolchain and tool the build, its tests and
# its fuzz targets need is fetched here. Rust matches rust-toolchain.toml
# (1.99.0); the fuzz lane uses nightly, as scripts/fuzz-ci.sh does.
FROM rust:1.99.0-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    CARGO_TERM_COLOR=never \
    # Full DWARF for the scanner; the crate's dev profile keeps line tables only.
    CARGO_PROFILE_DEV_DEBUG=full \
    RUSTFLAGS="-C force-frame-pointers=yes"
# wgpu needs no system headers to compile. The Vulkan loader and Mesa's
# lavapipe give it a software (CPU) adapter at run time, so the GPU analysis
# path can be exercised without a GPU, slowly; without them GPU tests skip.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git jq python3 libvulkan1 mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# Install the pinned toolchain from rust-toolchain.toml (a no-op when the base
# image already matches), fetch every crate against the committed lockfile, and
# build the library and every test target (the regression harness included)
# with debug info, so the offline audit can `cargo test` without the network.
# A failure here fails the build.
RUN rustup show && cargo --version \
    && cargo fetch --locked \
    && cargo build --all-features --locked \
    && cargo test --all-features --locked --no-run

# Fuzz lane (fuzz/, cargo-fuzz): nightly floats by design (see
# quality/cargo_install_pinning.sh); cargo-fuzz is pinned like fuzz-ci.sh pins
# it. fuzz/Cargo.lock lags the root manifest, so it is resolved here rather than
# with --locked. Optional, so a failure warns rather than aborting the build.
RUN (rustup toolchain install nightly --profile minimal \
      && cargo +nightly install --locked --version 0.13.2 cargo-fuzz \
      && cargo fetch --manifest-path fuzz/Cargo.toml \
      && cargo +nightly fuzz build \
      || echo "WARNING: fuzz lane setup failed")

# Smoke-run the FFI integration suite and the panic-response contract; a
# failure warns rather than aborting the build. The full suite is
# `cargo test --lib --tests --all-features -- --test-threads=2` from /src.
RUN (timeout 1200 cargo test --all-features --locked -q \
      --test ffi --test issue_2089_panic_response_json -- --test-threads=2 \
      || echo "WARNING: smoke tests reported failures")
