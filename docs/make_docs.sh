#!/bin/bash

# Generate API documentation
# sphinx-apidoc -o source ../src/relaxit

# Build the documentation
sphinx-build -b html source build/html