# download the omniobject3d dataset from their google drive

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
import io
import tarfile
import os
import cv2
import numpy as np
import shutil

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Obtain your Google credentials
def get_credentials():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

# Build the downloader
creds = get_credentials()
drive_downloader = build('drive', 'v3', credentials=creds)

# Replace 'FOLDER_ID' with your actual Google Drive folder ID
folder_id = '1QVTXrpZVG9vHVWAqQfxlv5-iPfJFti34'

# query = f"Folder ID '{folder_id}'"  # you may get error for this line
query = f"'{folder_id}' in parents"  # this works  ref https://stackoverflow.com/q/73119251/248616

results = drive_downloader.files().list(q=query, orderBy = 'name asc, modifiedTime desc', pageSize=1000).execute()
items = results.get('files', [])

# Download the files
for n, item in enumerate(items):
    request = drive_downloader.files().get_media(fileId=item['id'])
    print(item['name'])
    f = io.FileIO(item['name'], 'w+')
    print(f)
    downloader = MediaIoBaseDownload(f, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {item['name']}: {int(status.progress() * 100)}% complete.")
    f.close()

    # extract tarfiles
    object_name = item["name"][:item["name"].find(".")]
    if tarfile.is_tarfile(os.path.join(os.getcwd(), item["name"])):
        with tarfile.open(os.path.join(os.getcwd(), item["name"])) as f:
            f.extractall(object_name)
    object_top_level_path = os.path.join(os.getcwd(), object_name)
    print(f"Extracting image files for {object_name}.")
    for _, dirnames, _ in os.walk(object_name):
        for dir in dirnames:
            subimage_path = os.path.join(object_top_level_path, dir)

            json_file = os.path.join(subimage_path, "transforms.json")
            if not os.path.isdir(os.path.join(os.getcwd(), "rotation_data", object_name)):
                os.mkdir(os.path.join(os.getcwd(), "rotation_data", object_name))
            shutil.copy(json_file, os.path.join(os.getcwd(), "rotation_data", object_name, dir + "_transforms.json"))

            # individual image files for a particular model render
            image_files = [img for img in os.listdir(subimage_path) if os.path.isfile(os.path.join(subimage_path, img)) and os.path.splitext(img)[1] != ".json"]
            if not os.path.isdir(os.path.join(os.getcwd(), "img_data", object_name)):
                os.mkdir(os.path.join(os.getcwd(), "img_data", object_name))

            # for the first 15 images of each render, save them
            for img in image_files[:min(15, len(image_files))]:
                os.rename(os.path.join(subimage_path, img), os.path.join(os.getcwd(), "img_data", object_name, dir + "_" + img))

                img_path = os.path.join(os.getcwd(), "img_data", object_name, dir + "_" + img)
                image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                h, w = image.shape[:2]
                scale = min(192 / h, 256 / w)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # transparent canvas
                canvas = np.zeros((192, 256, 4), dtype=np.uint8)

                y_offset = (192 - new_h) // 2
                x_offset = (256 - new_w) // 2
                canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

                cv2.imwrite(img_path, canvas)
    shutil.rmtree(object_top_level_path)
    os.remove(os.path.join(os.getcwd(), object_name + ".tar.gz"))
    print(f"Completed {object_name} download, object #{n}.")