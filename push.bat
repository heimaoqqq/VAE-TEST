@echo off
cd /d "g:\vq-diffusion"
git add .
git commit -m "Fix conditional diffusion performance issues"
git remote set-url origin git@github.com:heimaoqqq/VAE-TEST.git
git push -f origin main
del push.bat
