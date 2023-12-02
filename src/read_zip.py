import os
import zipfile
import requests

def read_zip(url, directory):
    """
    Downloads and extracts a ZIP file from a specified URL into a given directory.

    Parameters:
    ----------
    url : str
        URL of the ZIP file to download.
    directory : str
        Directory where the ZIP file's contents will be extracted.

    Returns:
    -------
    None
    """

    # Send a GET request to the provided URL to download the file.
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    # Check if the URL is valid and accessible. If not, raise an error.
    if request.status_code != 200:
        raise ValueError('The URL provided does not exist.')

    # Verify that the URL points to a ZIP file based on the file extension.
    if filename_from_url[-4:] != '.zip':
        raise ValueError('The URL provided does not point to a zip file.')
    
    # Ensure the specified directory exists. If not, raise an error.
    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')

    # Save the downloaded ZIP file to the specified directory.
    path_to_zip_file = os.path.join(directory, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)

    # List the current files in the directory to check later if new files were added.
    original_files = os.listdir(directory)
    original_timestamps = [os.path.getmtime(os.path.join(directory, filename)) for filename in original_files]

    # Extract the contents of the ZIP file into the directory.
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory)

    # Compare the directory's contents before and after extraction.
    # If no new files were added, it indicates the ZIP file was empty or extraction failed.
    current_files = os.listdir(directory)
    current_timestamps = [os.path.getmtime(os.path.join(directory, filename)) for filename in current_files]
    if (len(current_files) == len(original_files)) and (original_timestamps == current_timestamps):
        raise ValueError('The ZIP file is empty or extraction failed.')
