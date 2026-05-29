import csv
import os


def list_files_to_csv():
    # 1. Get the folder path from the user
    folder_path = input("Enter the full path to the folder: ").strip()

    # 2. Validate if the path exists and is a directory
    if not os.path.exists(folder_path):
        print(f"Error: The path '{folder_path}' does not exist.")
        return
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a folder.")
        return

    # 3. Define the output CSV file name
    output_csv = "file_list.csv"

    try:
        # 4. Open the CSV file for writing
        with open(
            output_csv, mode="w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)

            # Write the header row
            writer.writerow(["File Name"])

            file_count = 0
            # 5. Loop through the items in the directory
            for item in os.listdir(folder_path):
                # Join path to check if it's actually a file (and not a subfolder)
                if os.path.isfile(os.path.join(folder_path, item)):
                    writer.writerow([item])
                    file_count += 1

        print(
            f"Success! Found {file_count} files and saved them to '{output_csv}'."
        )

    except Exception as e:
        print(f"An error occurred while saving the CSV: {e}")


if __name__ == "__main__":
    list_files_to_csv()