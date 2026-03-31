#!/usr/bin/env python

from analysis.visual_lib import parallel_plot

from pathlib import Path
from PIL import Image
from tqdm import tqdm

StartTime_str = "2022-12-20_00:00"
EndTime_str = "2022-12-26_23:00"
dt_draw_int = 1
region_US = {
    "lon_s": -129,
    "lon_e": -64,
    "lat_s": 20,
    "lat_e": 55,
}
varName = "surface"
demoDir = "demo"
figDir = demoDir + "/.tmp_" + varName
n_core = 12
dpi = 100
# Only the GIF frames are downscaled; the source PNG files in figDir are left untouched.
gif_scale = 0.6

#parallel_plot(StartTime_str, EndTime_str, dt_draw_int, region_US, varName, figDir, n_core, dpi)

png_files = sorted(Path(figDir).glob("*.png"))
if png_files:
    frames = []
    for png_file in tqdm(png_files, desc=f"Loading frames for {varName}"):
        with Image.open(png_file) as img:
            # Resize in memory before encoding so the final GIF is smaller without modifying the PNGs.
            resized_frame = img.convert("RGB").resize(
                (int(img.width * gif_scale), int(img.height * gif_scale)),
                Image.Resampling.LANCZOS,
            )
        frames.append(resized_frame)
    output_gif = Path(demoDir) / f"demo_{varName}.gif"
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=300,
        loop=0,
        optimize=True,
    )