import csv
import os


def delete_matching_files():
    # 1. Get the CSV file path and folder path from the user
    csv_path = (
        input("Enter the path/name of your CSV file (e.g., file_list.csv): ")
        .strip()
        .strip('"')
    )
    folder_path = (
        input("Enter the path to the folder where files should be deleted: ")
        .strip()
        .strip('"')
    )

    # 2. Validation checks
    if not os.path.exists(csv_path):
        print(f"Error: The CSV file '{csv_path}' was not found.")
        return
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: The folder path '{folder_path}' is invalid.")
        return

    # 3. Read the file names from the CSV into a set for quick lookup
    files_to_delete = set()
    try:
        with open(csv_path, mode="r", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader, None)  # Skip the header row ('File Name')

            for row in reader:
                if row:  # Ensure the row isn't empty
                    files_to_delete.add(row[0].strip())
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    if not files_to_delete:
        print("No file names found in the CSV.")
        return

    # 4. Safety Confirmation
    print(f"\nFound {len(files_to_delete)} unique file names in the CSV.")
    confirm = (
        input(
            f"Are you sure you want to delete matching files in '{folder_path}'? (yes/no): "
        )
        .strip()
        .lower()
    )

    # Updated to accept 'y' or 'yes'
    if confirm not in ["yes", "y"]:
        print("Operation cancelled. No files were deleted.")
        return

    # 5. Perform the deletion
    deleted_count = 0
    errors_count = 0

    print("\nProcessing deletions...")
    for filename in files_to_delete:
        file_path = os.path.join(folder_path, filename)

        # Check if the file actually exists in the target folder
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
                errors_count += 1

    # 6. Final Summary
    print("\n--- Summary ---")
    print(f"Successfully deleted: {deleted_count} files")
    if errors_count > 0:
        print(f"Failed to delete: {errors_count} files (check permissions)")
    if deleted_count == 0 and errors_count == 0:
        print("No matching files were found in the target folder.")


if __name__ == "__main__":
    delete_matching_files()