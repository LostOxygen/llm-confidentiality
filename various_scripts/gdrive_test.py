"""test script to test the google drive search tool"""

import os
from langchain_googledrive.utilities.google_drive import GoogleDriveAPIWrapper
from langchain_googledrive.tools.google_drive.tool import GoogleDriveSearchTool

os.environ["GOOGLE_ACCOUNT_FILE"] = "google_credentials.json"

# By default, search only in the filename.
tool = GoogleDriveSearchTool(
    api_wrapper=GoogleDriveAPIWrapper(
        folder_id="12QFKRqhM_R6-MMryXuOMIMGzIqaYGupt",
        num_results=1,
        template="gdrive-query-in-folder",  # Search in the body of documents
        mode="documents",
    )
)

print(tool.run("soup"))
