import metaworld
import numpy as np
import random
import time
from PIL import Image

# Initialize the MT1 benchmark for the 'assembly-v2' task
mt1 = metaworld.MT1('box-close-v2')

# Create an environment with task `assembly-v2`
env = mt1.train_classes['box-close-v2']()
task = random.choice(mt1.train_tasks)
env.set_task(task)

# High resolution render
img = env.sim.render(width=1920, height=1080, camera_name='corner2')

# Convert the image to RGB format and save it
image = Image.fromarray(img)
image.save('boxclose_env_render.pdf')

time.sleep(10)
env.close()
