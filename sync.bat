@echo off
echo Stage all changes...
git add -A

echo Committing changes...
git commit -m "Auto-sync local changes"

echo Pushing to GitHub...
git push origin main

echo Sync complete!
