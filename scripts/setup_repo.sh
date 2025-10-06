#!/usr/bin/env bash
set -e
git init
git add .
git commit -m "Initial commit: scaffolding for Impossibility of Retrain Equivalence in Machine Unlearning"
git branch -M main
echo "Now run either GitHub CLI (gh repo create ...) or add a remote and push."
