#!/bin/sh

find . -iname '*.h' -o -iname '*.cpp' -o -iname '*.cc' | xargs clang-format -i
