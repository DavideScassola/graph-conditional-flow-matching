import os
import pandas as pd
import shutil

def most_recent_artifact_folder():
    folder_path = "artifacts/models"
    subdirectories = [
        d
        for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d)) and d[0] != "_"
    ]
    subdirectories.sort()
    if subdirectories:
        print("Evaluating", subdirectories[-1])
        return os.path.join(folder_path, subdirectories[-1])
    else:
        return None

def remove_duplicates(input_folder, output_folder, table_name, columns):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Copy all files from input_folder to output_folder
    for file_name in os.listdir(input_folder):
        full_file_name = os.path.join(input_folder, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, output_folder)

    # Path to the specific table (CSV file)
    table_path = os.path.join(output_folder, table_name)

    # Check if the table exists
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table '{table_name}' not found in the input folder.")

    # Load the table as a DataFrame
    df = pd.read_csv(table_path)

    # Remove duplicate rows based on the foreign key and categorical column
    df = df.drop_duplicates(subset=columns)

    # Save the modified table back to the output folder
    df.to_csv(table_path, index=False)

    print(f"Duplicates removed and updated table saved in '{output_folder}'.")

def remove_duplicates_from_artifact_folder(artifact_folder, table_name, columns):
    
    if artifact_folder is None:
        raise ValueError("No artifact folder found.")
    
    output_folder = artifact_folder + '/report/generated' # Replace with the path to the input folder
    # rename generated -> generated_old
    input_folder = artifact_folder + '/report/generated_old'
    os.rename(output_folder, input_folder)

    remove_duplicates(input_folder, output_folder, table_name, columns)

# Example usage
if __name__ == "__main__":
    
    artifact_folder = most_recent_artifact_folder()
    
    if artifact_folder is None:
        raise ValueError("No artifact folder found.")
       
    remove_duplicates_from_artifact_folder(artifact_folder,
                                           table_name = "content.csv",
                                           columns = ["paper_id", "word_cited_id"])