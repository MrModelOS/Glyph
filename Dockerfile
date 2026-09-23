FROM rust:1.85

# System deps for building and running glyphc (gcc needed for Glyph's C backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache deps first: copy manifests and build dummy
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY stdlib ./stdlib
# Ensure build succeeds even if other top-level files are absent in early layer
RUN cargo build --release && rm -rf target/release/deps target/release/.fingerprint

# Copy the rest of the repo and do a full build
COPY . .
RUN cargo build --release

# Make binary available on PATH
RUN cp target/release/glyphc /usr/local/bin/glyphc

ENTRYPOINT ["glyphc"]
CMD ["--help"]
