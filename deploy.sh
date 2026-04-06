#!/bin/bash
# Usage: bash deploy.sh YOUR_HF_USERNAME
# Deploys ClusterOrch-Gym to a HuggingFace Space

HF_USERNAME=$1
SPACE_NAME="clusterorch-gym"

if [ -z "$HF_USERNAME" ]; then
  echo "Usage: bash deploy.sh YOUR_HF_USERNAME"
  echo "Make sure HF_TOKEN is set: export HF_TOKEN=your_token"
  exit 1
fi

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set. Run: export HF_TOKEN=your_token"
  exit 1
fi

echo "Logging into HuggingFace..."
huggingface-cli login --token "$HF_TOKEN"

echo "Creating HuggingFace Space..."
huggingface-cli repo create "$SPACE_NAME" --type space --space_sdk docker 2>/dev/null || \
  echo "Space already exists, continuing..."

echo "Pushing files..."
if [ ! -d ".git" ]; then
  git init
fi
git add .
git commit -m "ClusterOrch-Gym submission" --allow-empty
git remote remove origin 2>/dev/null
git remote add origin "https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
git push origin main --force

echo ""
echo "Done! Check your space at:"
echo "  https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"
