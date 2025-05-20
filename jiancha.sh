#!/bin/bash

echo "Checking required packages..."

packages=(
  numpy
  wandb
  matplotlib
  scipy
  pytorch_lightning
  shapely
  networkx
  cartopy
  pyproj
  tueplots
  plotly
  # dev tools
  codespell
  black
  isort
  flake8
  pylint
  pre_commit
)

for pkg in "${packages[@]}"; do
  version=$(pip show $pkg 2>/dev/null | grep -i version | awk '{print $2}')
  if [ -z "$version" ]; then
    echo "$pkg ❌ Not installed"
  else
    echo "$pkg ✅ Version $version"
  fi
done
