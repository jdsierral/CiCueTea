#!/bin/sh

find Plugins -iname '*.h' -o -iname '*.cpp' -o -iname '*.cc' | xargs clang-format -i
