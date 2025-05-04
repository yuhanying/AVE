import glob
import subprocess
import os
import multiprocessing
import argparse


parser = argparse.ArgumentParser(description=('Extract frames for the AVE dataset'))
parser.add_argument('--video_dir', default='/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/feature_extractors/video/AVE_Dataset/AVE',type=str, help='Path to .MP4 videos')
parser.add_argument('--out_dir', default='/home/server-05/YU/OGM-GE_CVPR2022/data/AVE', type=str, help='Path to save extracted frames')

def ffmpeg_extraction(videofile):
    basename = os.path.basename(videofile)[:-4]
    print(f'Extracting {videofile} to {basename}')

    # Change this to store the frames
    outdir = f'{args.out_dir}/{basename}'
    os.makedirs(outdir, exist_ok=True)

    command = f"ffmpeg -i {videofile} '{outdir}/frame_%10d.jpg'"
    subprocess.call(command, shell=True)

if __name__ == '__main__':
    args = parser.parse_args()
    # print(args.video_dir)

    ## Change the path to your own dataset path
    mp4files = glob.glob(f'{args.video_dir}/*.mp4') ##4097
    # print(f'Found {len(mp4files)} videos')
    with multiprocessing.Pool(32) as p:
        p.map(ffmpeg_extraction, mp4files)