#!/bin/bash
set -e

QUARTO_VERSION="1.5.57"
QUARTO_BASE="https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}"

# Choose tarball by OS and architecture
case "$(uname -s)" in
  Darwin)
    QUARTO_TAR="quarto-${QUARTO_VERSION}-macos.tar.gz"
    ;;
  Linux)
    case "$(uname -m)" in
      x86_64|amd64) QUARTO_TAR="quarto-${QUARTO_VERSION}-linux-amd64.tar.gz" ;;
      aarch64|arm64) QUARTO_TAR="quarto-${QUARTO_VERSION}-linux-arm64.tar.gz" ;;
      *) echo "Unsupported Linux arch: $(uname -m)"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $(uname -s)"; exit 1
    ;;
esac

# Install Quarto without sudo
echo "Downloading Quarto (${QUARTO_TAR})..."
curl -LO "${QUARTO_BASE}/${QUARTO_TAR}"

echo "Extracting Quarto..."
rm -rf ~/quarto
mkdir -p ~/quarto
tar -xzf "${QUARTO_TAR}" -C ~/quarto --strip-components=1
rm -f "${QUARTO_TAR}"

echo "Adding Quarto to PATH..."
export PATH="$HOME/quarto/bin:$PATH"

echo "Quarto version:"
quarto --version

# Render the site
echo "Rendering Quarto site..."
quarto render --to html

echo "Build complete!"
