#!/usr/bin/env bash
# Installs the Swift 6.2 toolchain for Linux into /opt/swift (idempotent).
# Usage: Tools/install-swift-linux.sh  (then run the printed export line in your shell)
set -euo pipefail

SWIFT_ROOT="/opt/swift/swift-6.2-RELEASE-ubuntu24.04"
SWIFT_BIN="${SWIFT_ROOT}/usr/bin/swift"
SWIFT_URL="https://download.swift.org/swift-6.2-release/ubuntu2404/swift-6.2-RELEASE/swift-6.2-RELEASE-ubuntu24.04.tar.gz"

SUDO=""
if [[ "$(id -u)" -ne 0 ]]; then
    SUDO="sudo"
fi

if [[ -x "${SWIFT_BIN}" ]]; then
    echo "Swift 6.2 already installed at ${SWIFT_ROOT}"
else
    workdir="$(mktemp -d)"
    trap 'rm -rf "${workdir}"' EXIT
    echo "Downloading ${SWIFT_URL}"
    curl -fsSL "${SWIFT_URL}" -o "${workdir}/swift.tar.gz"
    ${SUDO} mkdir -p /opt/swift
    ${SUDO} tar -xzf "${workdir}/swift.tar.gz" -C /opt/swift
    echo "Installed Swift 6.2 into /opt/swift"
fi

echo "Add to your shell:"
echo "export PATH=${SWIFT_ROOT}/usr/bin:\$PATH"
