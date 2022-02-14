from .utils import analyze_dir, reset
DOWNSAMPLED_DIR = './downsampled_audios'
if __name__ == '__main__':
    reset()
    analyze_dir(DOWNSAMPLED_DIR)