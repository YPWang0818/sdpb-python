#!/usr/bin/env bash
# Stamp a development version into the working tree (not for committing):
# the next patch version after the latest v* tag, as a PEP 440 dev release
# numbered by the commits since that tag, e.g. 0.3.1.dev12 for the 12th
# commit after v0.3.0. It sorts above that release and below any later one.
# Used by the wheels workflow for preview builds; prints the version.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tag="$(git -C "$ROOT" describe --tags --match 'v[0-9]*' --abbrev=0)"
count="$(git -C "$ROOT" rev-list --count "$tag"..HEAD)"
IFS=. read -r major minor patch <<< "${tag#v}"
version="$major.$minor.$((patch + 1)).dev$count"

sed -i -E "s/^version = \"[^\"]*\"/version = \"$version\"/" "$ROOT/pyproject.toml"
sed -i -E "s/^__version__ = \"[^\"]*\"/__version__ = \"$version\"/" "$ROOT/src/sdpb_python/__init__.py"
grep -q "^version = \"$version\"" "$ROOT/pyproject.toml"
grep -q "^__version__ = \"$version\"" "$ROOT/src/sdpb_python/__init__.py"
echo "$version"
