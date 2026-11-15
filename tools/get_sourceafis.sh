#!/usr/bin/env bash
# SourceAFIS 3.18.1 ve bağımlılıklarını Maven Central'dan tools/lib/ altına indirir (SHA-1 doğrulamalı).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lib
REPO=https://repo1.maven.org/maven2
ARTIFACTS=(
  com.machinezoo.sourceafis:sourceafis:3.18.1
  com.machinezoo.stagean:stagean:1.3.0
  com.machinezoo.closeablescope:closeablescope:1.0.1
  com.machinezoo.noexception:noexception:1.9.1
  com.machinezoo.fingerprintio:fingerprintio:1.3.1
  it.unimi.dsi:fastutil:8.5.12
  commons-io:commons-io:2.15.0
  com.google.code.gson:gson:2.10.1
  com.fasterxml.jackson.core:jackson-databind:2.15.3
  com.fasterxml.jackson.core:jackson-core:2.15.3
  com.fasterxml.jackson.core:jackson-annotations:2.15.3
  com.fasterxml.jackson.dataformat:jackson-dataformat-cbor:2.15.3
  com.github.mhshams:jnbis:2.1.2
  org.slf4j:slf4j-api:2.0.9
  org.slf4j:slf4j-nop:2.0.9
)
for a in "${ARTIFACTS[@]}"; do
  IFS=: read -r g n v <<<"$a"
  f="$n-$v.jar"; url="$REPO/${g//.//}/$n/$v/$f"
  [ -f "lib/$f" ] || curl -sfL -o "lib/$f" "$url"
  want=$(curl -sfL "$url.sha1" | cut -c1-40)
  got=$(sha1sum "lib/$f" | cut -c1-40)
  [ "$want" = "$got" ] || { echo "SHA-1 uyuşmuyor: $f"; rm -f "lib/$f"; exit 1; }
  echo "ok  $f"
done
