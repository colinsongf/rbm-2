from PIL import Image
import os

HORSE_PATH='weizmann_horse_db/figure_ground/'
OUT_PATH='weizmann/'

for f in next(os.walk(HORSE_PATH))[2]:
    img = Image.open(HORSE_PATH+f)
    img = img.resize((28, 28), Image.ANTIALIAS)
    img.save(OUT_PATH+f)


