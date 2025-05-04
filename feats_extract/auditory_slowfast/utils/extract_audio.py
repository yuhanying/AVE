import argparse
import subprocess
import os


def ffmpeg_extraction(input_video, output_audio, sampling_rate):

    ffmpeg_command = ['ffmpeg', '-i', input_video,
                      '-vn', '-acodec', 'pcm_s16le',
                      '-ac', '1', '-ar', sampling_rate,
                      output_audio]

    subprocess.call(ffmpeg_command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--videos_dir', default='/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/feature_extractors/video/AVE_Dataset/AVE', help='Directory of AVE videos with audio')
    parser.add_argument('--output_dir', default='/home/server-05/YU/OGM-GE_CVPR2022/TIM-main/feature_extractors/audio/ave', help='Directory to save AVE audio')
    parser.add_argument('--sampling_rate', default='16000', help='Rate to resample audio')

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for root, dirs, files in os.walk(args.videos_dir):
        for f in files:
            if f.endswith('.mp4'):
                ffmpeg_extraction(os.path.join(root, f),
                                  os.path.join(args.output_dir,
                                               os.path.splitext(f)[0] + '.wav'),
                                  args.sampling_rate)
