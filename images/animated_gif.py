import imageio
from pathlib import Path
images = []
for filename in Path.cwd().glob('*.png'):
    images.append(imageio.imread(filename))
imageio.mimsave('linear_anim.gif', images, duration=1)