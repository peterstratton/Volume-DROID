import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image as im

def save_traj_video(imgs, dir='./demo_video/animation.mp4'):
    imgs = np.transpose(imgs, (0, 2, 3, 1))    
    fig  = plt.figure()

    frames = []
    for i in range(poses.shape[0]):
        frames.append([plt.imshow(imgs[i], animated=True)])

    # Create the animation object
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=10000)

    # Save the animation to a file
    ani.save(dir, writer='ffmpeg')


if __name__ == '__main__':
    imgs = np.load('/opt/DROID-SLAM/reconstructions/demos/tartanair/images.npy')
    poses = np.load('/opt/DROID-SLAM/reconstructions/demos/tartanair/poses.npy')

    # save_traj_video(imgs)

    plt.plot(poses[:, 0], poses[:, 1])
    plt.savefig("./demo_video/pose_plot.png")

    

