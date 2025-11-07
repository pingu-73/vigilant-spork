#!/bin/bash

set -v
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
