#!/bin/sh
rm build -fr || true && mkdir -p build && cd build/ && cmake .. -DCMAKE_BUILD_TYPE=Release && make
