import metaworld
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# Initialize the MT1 benchmark for the 'assembly-v2' task
mt1 = metaworld.MT1('assembly-v2')
env = mt1.train_classes['assembly-v2']()
task = random.choice(mt1.train_tasks)
env.set_task(task)

# Render the environment background
img = env.sim.render(width=1920, height=1080, camera_name='corner2')
image = Image.fromarray(img)

# Sample multiple initial positions
num_samples = 20
positions = []

for _ in range(num_samples):
    obs = env.reset()
    gripper_position = obs[0:3]  # Assuming first three entries are x, y, z coordinates of the gripper
    positions.append(gripper_position)

env.close()

# Convert list to numpy array for easier plotting
positions = np.array(positions)

# Create a plot overlaying the positions on the rendered image
fig, ax = plt.subplots(figsize=(19.2, 10.8))  # Match the size of the rendered image
ax.imshow(img, aspect='auto')  # Display the background image
ax.scatter(960 + positions[:, 0] * 1000, 540 - positions[:, 1] * 1000, color='red')  # Scale and translate points appropriately

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Initial Gripper Positions on Rendered Environment')

# Hide axes
ax.axis('off')

# Save the complete plot as a PDF
plt.savefig('overlay_gripper_positions.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
