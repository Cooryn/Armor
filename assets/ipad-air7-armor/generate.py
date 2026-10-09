"""Regenerate the iPad Air 7 target from the current Solver geometry."""
from pathlib import Path
import json
import re
from PIL import Image, ImageDraw, PngImagePlugin

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = (ROOT / 'src' / 'solver.cpp').read_text(encoding='utf-8-sig')

def dimension(name):
    return float(re.search(rf'constexpr\s+float\s+{name}\s*=\s*([\d.]+)f', SOURCE).group(1)) * 1000

WIDTH_MM, HEIGHT_MM = dimension('armor_width_m'), dimension('armor_height_m')
PPI = 264
PPM = PPI / 25.4
BAR_WIDTH_MM = 10  # Display choice: no bar-width parameter exists in Solver.
DIGIT_POLYGON = [(-13, -23), (13, -23), (13, 23), (-13, 23), (-13, 16), (6, 16), (6, 3.5), (-9, 3.5), (-9, -3.5), (6, -3.5), (6, -16), (-13, -16)]
DIGIT_PATH = 'M' + ' L'.join(f'{x},{y}' for x, y in DIGIT_POLYGON) + ' Z'
CONFIG = dict(widthMm=round(WIDTH_MM, 6), heightMm=round(HEIGHT_MM, 6), barWidthMm=BAR_WIDTH_MM, ppi=PPI, digit3Path=DIGIT_PATH)

def image(size, number=True, calibration=False):
    ss = 4
    w, h = size
    img = Image.new('RGB', (w * ss, h * ss), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2
    def point(x, y):
        return ((cx + x * PPM) * ss, (cy + y * PPM) * ss)
    for x in [-WIDTH_MM / 2, WIDTH_MM / 2]:
        draw.rectangle([point(x - BAR_WIDTH_MM / 2, -HEIGHT_MM / 2), point(x + BAR_WIDTH_MM / 2, HEIGHT_MM / 2)], fill=(255, 0, 0))
    if number:
        draw.polygon([point(x, y) for x, y in DIGIT_POLYGON], fill=(255, 255, 255))
    if calibration:
        # The central target is unchanged; markers are for ruler verification only.
        for points in [ [(-50, 52), (50, 52)], [(-50, 49), (-50, 55)], [(50, 49), (50, 55)], [(-95, -25), (-95, 25)], [(-98, -25), (-92, -25)], [(-98, 25), (-92, 25)] ]:
            draw.line([point(x, y) for x, y in points], fill='white', width=ss * 2)
    return img.resize(size, Image.Resampling.LANCZOS)

outputs = [
    ('red-armor-air7-landscape.png', (2360, 1640), True, False),
    ('red-armor-air7-portrait.png', (1640, 2360), True, False),
    ('red-armor-air7-plain-landscape.png', (2360, 1640), False, False),
    ('calibration-air7-landscape.png', (2360, 1640), True, True),
]
for name, size, number, calibration in outputs:
    info = PngImagePlugin.PngInfo()
    info.add_text('Geometry', f'Light center spacing {WIDTH_MM:g} mm; light length {HEIGHT_MM:g} mm; light width {BAR_WIDTH_MM:g} mm')
    info.add_text('Display', 'iPad Air 7 11-inch M3; 264 ppi; native full-screen without zoom')
    image(size, number, calibration).save(HERE / name, dpi=(PPI, PPI), pnginfo=info)

svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_MM + BAR_WIDTH_MM:g}mm" height="{HEIGHT_MM:g}mm" viewBox="{- (WIDTH_MM + BAR_WIDTH_MM) / 2:g} {-HEIGHT_MM / 2:g} {WIDTH_MM + BAR_WIDTH_MM:g} {HEIGHT_MM:g}">
  <title>Red armor: light center spacing {WIDTH_MM:g} mm, light length {HEIGHT_MM:g} mm</title>
  <rect x="{- (WIDTH_MM + BAR_WIDTH_MM) / 2:g}" y="{-HEIGHT_MM / 2:g}" width="{WIDTH_MM + BAR_WIDTH_MM:g}" height="{HEIGHT_MM:g}" fill="black"/>
  <g fill="#ff0000">
    <rect x="{-WIDTH_MM / 2 - BAR_WIDTH_MM / 2:g}" y="{-HEIGHT_MM / 2:g}" width="{BAR_WIDTH_MM:g}" height="{HEIGHT_MM:g}"/>
    <rect x="{WIDTH_MM / 2 - BAR_WIDTH_MM / 2:g}" y="{-HEIGHT_MM / 2:g}" width="{BAR_WIDTH_MM:g}" height="{HEIGHT_MM:g}"/>
  </g>
  <path d="{DIGIT_PATH}" fill="white"/>
</svg>
'''
(HERE / 'red-armor-135x56mm.svg').write_text(svg, encoding='utf-8')
(HERE / 'geometry.json').write_text(json.dumps(CONFIG, indent=2) + '\n', encoding='utf-8')
template = (HERE / 'index.template.html').read_text(encoding='utf-8')
(HERE / 'index.html').write_text(template.replace('__ARMOR_CONFIG__', json.dumps(CONFIG)), encoding='utf-8')
print(json.dumps({'width_mm': WIDTH_MM, 'height_mm': HEIGHT_MM, 'ppi': PPI, 'center_spacing_pixels': WIDTH_MM * PPM, 'light_length_pixels': HEIGHT_MM * PPM, 'png_files': [item[0] for item in outputs]}, indent=2))
