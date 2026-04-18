#!/bin/bash
set -e

COMMON_EXCLUDES=(
    --exclude ".git/*"
    --exclude ".venv/*"
    --exclude "data/*"
    --exclude "data.bak*/*"
    --exclude ".env"
    --exclude "__pycache__/*"
    --exclude "*.pyc"
    --exclude "*.pyo"
    --exclude "mkexam/__pycache__/*"
    --exclude ".claude/*"
    --exclude "CLAUDE.md"
    --exclude "make-package.sh"
)

# ── Generator package (full app) ─────────────────────────────────────────────
GEN_OUT="mkexam-generate.zip"
rm -f "$GEN_OUT"
zip -r "$GEN_OUT" . "${COMMON_EXCLUDES[@]}"
echo "Generator package: $GEN_OUT ($(du -h "$GEN_OUT" | cut -f1))"

# ── Quiz-only package ─────────────────────────────────────────────────────────
QUIZ_OUT="mkexam-quiz.zip"
rm -f "$QUIZ_OUT"
zip "$QUIZ_OUT" \
    app_quiz.py \
    requirements_quiz.txt \
    run-quiz.sh \
    run-quiz.bat
zip -r "$QUIZ_OUT" \
    templates/ \
    mkexam/__init__.py \
    mkexam/storage.py \
    mkexam/spaced.py \
    "${COMMON_EXCLUDES[@]}"
echo "Quiz package:      $QUIZ_OUT ($(du -h "$QUIZ_OUT" | cut -f1))"
