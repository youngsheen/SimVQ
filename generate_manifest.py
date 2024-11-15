import glob
import os


def save_audio_paths_to_txt(glob_pattern, base_dir, output_file):
    """
    Finds all .wav files using a glob pattern and saves their relative paths to a text file.
    
    :param glob_pattern: Glob pattern to match files.
    :param base_dir: Base directory to calculate relative paths, excluding parent directories.
    :param output_file: Path to the output text file.
    """
    # Use glob to match all .wav files
    audio_paths = glob.glob(glob_pattern, recursive=True)
    
    # Normalize paths and strip the base directory
    relative_paths = [os.path.relpath(path, base_dir) for path in audio_paths]

    # Save to text file
    with open(output_file, 'w') as f:
        for path in relative_paths:
            f.write(f"{path}\n")
    
    print(f"Saved {len(relative_paths)} audio file paths to {output_file}")

# Example usage:
libritts_root = "/data3/yongxinzhu/libritts/LibriTTS"
glob_pattern = os.path.join(libritts_root, "test-other/**/*.wav") 
output_txt = "/data3/yongxinzhu/libritts/LibriTTS/test-other.txt"

save_audio_paths_to_txt(glob_pattern, libritts_root, output_txt)