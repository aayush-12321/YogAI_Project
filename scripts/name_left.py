import os


def add_prefix_to_files():
    # 1. Get the folder path from the user
    folder_path = (
        input("Enter the path to the folder: ").strip().strip('"')
    )

    # 2. Validate if the path exists
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' is invalid.")
        return

    prefix = "left_"
    rename_count = 0
    skipped_count = 0

    print(f"\nAdding prefix '{prefix}' to files in: {folder_path}\n")

    # 3. Loop through all items in the directory
    for filename in os.listdir(folder_path):
        old_file_path = os.path.join(folder_path, filename)

        # Ensure we are only renaming files (skip subfolders)
        if os.path.isfile(old_file_path):
            # Skip if the file already has the prefix
            if filename.startswith(prefix):
                print(f"Skipped (already has prefix): {filename}")
                skipped_count += 1
                continue

            # Create the new filename and full path
            new_filename = prefix + filename
            new_file_path = os.path.join(folder_path, new_filename)

            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
                rename_count += 1
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

    # 4. Final Summary
    print("\n--- Summary ---")
    print(f"Successfully renamed: {rename_count} files")
    if skipped_count > 0:
        print(f"Skipped: {skipped_count} files")


if __name__ == "__main__":
    add_prefix_to_files()