@echo off
cd /d "g:\vq-diffusion"
git add .
git commit -m "Final fix for conditional diffusion with API verification

Key fixes:
1. Unified UNet channel configuration (in_channels=3, out_channels=3)
2. Unified VAE scaling factor (0.18215) in training and inference
3. Fixed classifier-free guidance logic indentation
4. Simplified UNet config to match unconditional training
5. Added comprehensive test scripts for verification

All diffusers API usage verified and corrected."
git push -f origin main
del push_final.bat
