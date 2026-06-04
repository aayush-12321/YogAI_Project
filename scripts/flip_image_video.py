import os
import cv2

def flip_media(parent_folder):
    # Path for the new flipped directory
    output_folder = os.path.join(parent_folder, "flipped")
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Supported extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')

    # Iterate through all files in the parent folder
    for filename in os.listdir(parent_folder):
        file_path = os.path.join(parent_folder, filename)
        
        # Skip directories (like the 'flipped' folder itself)
        if os.path.isdir(file_path):
            continue

        ext = os.path.splitext(filename)[1].lower()
        output_path = os.path.join(output_folder, filename)

        # --- PROCESS IMAGES ---
        if ext in image_extensions:
            print(f"Processing image: {filename}...")
            img = cv2.imread(file_path)
            if img is not None:
                # 1 means horizontal flipping (left-right)
                flipped_img = cv2.flip(img, 1) 
                cv2.imwrite(output_path, flipped_img)
            else:
                print(f"Warning: Could not read image {filename}")

        # --- PROCESS VIDEOS ---
        elif ext in video_extensions:
            print(f"Processing video: {filename} (this may take a moment)...")
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {filename}")
                continue

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define codec and create VideoWriter object
            # mp4v works well for MP4; change codec if using different formats
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip the frame horizontally
                flipped_frame = cv2.flip(frame, 1)
                out.write(flipped_frame)

            # Release resources for this video
            cap.release()
            out.release()
            print(f"Finished video: {filename}")

    print("\n All processing complete! Check the 'flipped' folder.")

if __name__ == "__main__":
    # Input path of the folder containing your media
    folder_path = input("Enter the path to the media folder: ").strip()
    
    if os.path.exists(folder_path):
        flip_media(folder_path)
    else:
        print("Error: The specified folder path does not exist.")