from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from pathlib import Path


font_path = Path.home() / '.local/share/fonts/sgr-iosevka.ttc'
font = ImageFont.truetype(str(font_path), 96, index=30)


with open('frame_probs.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        path, _, prob = tuple(line.split(' '))
        prob = float(prob)  # type: ignore
        img = Image.open(path)
        draw = ImageDraw.Draw(img)
        new_path = Path('data/overlay_frames') / Path(path).name
        if prob > .5:  # type: ignore
            draw.text((300, 200), f'{prob*100:.2f}%', (255, 0, 0), font=font)
        else:
            draw.text((300, 200), f'{prob*100:.2f}%', (0, 255, 0), font=font)
        img.save(new_path)
