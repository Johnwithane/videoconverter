import cv2
import numpy as np
import os
from pytube import YouTube
from PIL import Image
import subprocess


youtube_url = ''  
base_directory = './output/'  # Replace with the base directory path

def download_youtube_video(url, base_path):
    print("Downloading YouTube Video...")
    yt = YouTube(url)
    video = yt.streams.get_highest_resolution()
    safe_title = yt.title.replace(' ', '').replace('/', '')  # Remove spaces and replace problematic characters
    output_dir = os.path.join(base_path, safe_title)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video.download(output_path=output_dir, filename=safe_title + '.mp4')
    return os.path.join(output_dir, safe_title + '.mp4'), output_dir

def compute_histogram(frame):
    hist = cv2.calcHist([frame], [0, 1, 2], None, [64, 64, 64], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def detect_no_cut_segments(video_path, threshold=.8, segment_length=1):
    print("Processing Samples...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error: Could not open video.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_required = int(fps * segment_length)
    prev_hist = None
    cuts = [0]
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        hist = compute_histogram(frame)
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if diff < threshold:
                cuts.append(i)
        prev_hist = hist
    cap.release()
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > frames_required], fps

def resize_and_crop(frame, target_size=560):
    height, width = frame.shape[:2]
    scale = target_size / min(height, width)
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    mid_y, mid_x = resized.shape[0] // 2, resized.shape[1] // 2
    cropped = resized[mid_y - target_size//2:mid_y + target_size//2, mid_x - target_size//2:mid_x + target_size//2]
    return cropped

def compress_and_overwrite_png(path):
    # Load the image
    image = Image.open(path)
    # Reduce the number of colors using the quantize method
    image = image.quantize(colors=64)
    # Overwrite the original image with the compressed version
    image.save(path, format='PNG', optimize=True)

def create_grid_from_segments(video_path, segments, fps, output_dir):
    cap = cv2.VideoCapture(video_path)
    video_filename = os.path.basename(video_path).rsplit('.', 1)[0]
    for idx, (start, end) in enumerate(segments):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end:
            ret, frame = cap.read()
            if not ret:
                break
            frame = resize_and_crop(frame)
            frames.append(frame)
        frames_to_use = frames[:25]
        if len(frames_to_use) < 25:
            print(f"Segment {idx+1} has fewer than 25 frames, skipping.")
            continue
        grid_size = 5
        frame_size = 560
        canvas = np.zeros((frame_size * grid_size, frame_size * grid_size, 3), dtype=np.uint8)
        for i, frame in enumerate(frames_to_use):
            row, col = divmod(i, grid_size)
            canvas[row*frame_size:(row+1)*frame_size, col*frame_size:(col+1)*frame_size] = frame
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, f"{video_filename}_smpl_{idx+1}.png")
        cv2.imwrite(output_path, canvas)
        # Compress and overwrite the PNG file right after it's created
        compress_and_overwrite_png(output_path)
        print(f"Created PNG: {video_filename}_smpl_{idx+1}.png")
    cap.release()

def extract_sprites(sheet, sprite_width, sprite_height):
    """Extract sprites from a sprite sheet."""
    sprites = []
    for y in range(0, sheet.height, sprite_height):
        for x in range(0, sheet.width, sprite_width):
            sprite = sheet.crop((x, y, x + sprite_width, y + sprite_height))
            sprites.append(sprite)
    return sprites

def create_gif(sprites, output_path, resize_to=None, num_colors=64, frame_duration=10):
    """Create a GIF from a list of sprites, with quantization and adjustable frame rate."""
    if resize_to:
        sprites = [sprite.resize(resize_to, Image.ANTIALIAS) for sprite in sprites]
    sprites = [sprite.quantize(colors=num_colors) for sprite in sprites]
    sprites[0].save(
        output_path,
        save_all=True,
        append_images=sprites[1:],
        optimize=False,
        duration=frame_duration,  # Frame duration in milliseconds
        loop=1
    )

def process_sprite_sheets(input_dir, output_dir, sprite_size=(560, 560), gif_size=(100, 100)):
    """Process all sprite sheets in the specified directory."""
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            path = os.path.join(input_dir, filename)
            sheet = Image.open(path)
            sprites = extract_sprites(sheet, *sprite_size)
            gif_path = os.path.join(output_dir, filename.replace('.png', '.gif').replace('smpl', 'thumb'))
            create_gif(sprites, gif_path, resize_to=gif_size)
            print(f"Created GIF: {filename.replace('.png', '.gif').replace('smpl', 'thumb')}")
            # show_menu()

def show_menu():
    # Ask the user to enter a number
    user_input = input("Enter YouTube URL: ")

    # Convert the input to an integer (you could add error checking here)
    youtube_url = str(user_input)

    # Call the function and store the result
    video_path, output_directory = download_youtube_video(youtube_url, base_directory)
    segments, fps = detect_no_cut_segments(video_path)
    input_directory = output_directory
    print("Detected segments:", segments)
    create_grid_from_segments(video_path, segments, fps, output_directory)
    print("Frame rate (FPS):", fps)
    # compress_pngs(input_directory)
    process_sprite_sheets(input_directory, output_directory)
    show_menu()

if __name__ == "__main__":
    show_menu()

