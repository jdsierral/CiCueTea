#!/bin/sh
# Apply clang-format (config: .clang-format at repo root) to the library sources.
# Only touches code owned by this repo — skips build/, BenchTests/, docs/, etc.

set -eu

cd "$(dirname "$0")"

DIRS="Include Source Tests"

find $DIRS \
    \( -iname '*.h' -o -iname '*.hpp' -o -iname '*.cpp' -o -iname '*.cc' \) \
    -print0 | xargs -0 clang-format -i --verbose