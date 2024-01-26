
import cv2 
import argparse
import glob
from pathlib import Path



def parse_config():
    # Settings.
    parser = argparse.ArgumentParser(description='Video2Images.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file_path', type=str, default='/data/chenghao/project/apollo/imgs', help='The path of video.')
    parser.add_argument('--output_dir', type=str, default='/data/chenghao/project/apollo/imgs/test.avi',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--show_video', action='store_true', help='The path of video.')

    args = parser.parse_args()
    
    return args 

    
def main():
    args = parse_config()
    w, h, fps = 1920, 1080, 30
    videowriter = cv2.VideoWriter(args.output_dir, 
                                  cv2.VideoWriter_fourcc('M','J','P','G'), 
                                  fps, 
                                  (w, h))
    
    
    pimg_path =  Path(args.file_path)
    data_fromat = '.jpg'
    data_file_list = glob.glob(str(pimg_path / f'*{data_fromat}')) if pimg_path.is_dir() else [pimg_path]
    data_file_list.sort() 
    # print(data_file_list)
    
    
    for idx, file_ in enumerate(data_file_list):
        print("Progress: {}/{}".format(idx, len(data_file_list)))
        img = cv2.imread(file_)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('img', img)
        # cv2.waitKey(1)
        videowriter.write(img)

    
    
if __name__ == '__main__':
    main()