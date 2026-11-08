# syntax=docker/dockerfile:1
#
# AnyCount Studio — portable multi-stage image.
#
#   stage 1  node:22-bookworm-slim  -> build the React SPA into web/dist
#   stage 2  rust:1.88-bookworm     -> compile the Axum single binary (rust-embed embeds web/dist)
#   stage 3  debian:bookworm-slim   -> minimal runtime: binary + python workers
#
# The server enforces a loopback-only ANYCOUNT_BIND (src/config.rs), so the container
# listens on 127.0.0.1:2030 exactly like the systemd unit. On Linux, run it with host
# networking (see compose.yml) so the port is reachable from the host / Caddy.
#
# Override the pinned toolchain images when Cargo.lock needs a newer MSRV, e.g.
#   docker build --build-arg RUST_IMAGE=rust:1.99-bookworm -t anycount-prototype .

ARG NODE_IMAGE=node:22-bookworm-slim
ARG RUST_IMAGE=rust:1.88-bookworm
ARG RUNTIME_IMAGE=debian:bookworm-slim

# ---------- stage 1: React SPA ----------
FROM ${NODE_IMAGE} AS web
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# ---------- stage 2: Rust binary ----------
FROM ${RUST_IMAGE} AS backend
WORKDIR /build
RUN apt-get update \
 && apt-get install -y --no-install-recommends pkg-config \
 && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY migrations ./migrations
COPY workers ./workers
COPY --from=web /build/web/dist ./web/dist
RUN cargo build --release --locked

# ---------- stage 3: runtime ----------
FROM ${RUNTIME_IMAGE} AS runtime
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=backend /build/target/release/anycount-prototype /usr/local/bin/anycount-prototype
COPY workers ./workers
COPY migrations ./migrations

ENV ANYCOUNT_BIND=127.0.0.1:2030 \
    ANYCOUNT_DATA_DIR=/var/lib/anycount-prototype \
    ANYCOUNT_REPO_ROOT=/app/repo \
    RUST_LOG=info,anycount_prototype=info

RUN mkdir -p /var/lib/anycount-prototype/uploads /var/lib/anycount-prototype/renders \
             /var/lib/anycount-prototype/packs /var/lib/anycount-prototype/tmp
VOLUME ["/var/lib/anycount-prototype"]
EXPOSE 2030
ENTRYPOINT ["/usr/local/bin/anycount-prototype"]
