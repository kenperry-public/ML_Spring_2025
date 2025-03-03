#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys


from pathlib import Path
import os

def IN_COLAB():
    try:
      from google.colab import drive
      IN_COLAB=True
    except:
      IN_COLAB=False
    
    return IN_COLAB

def upload(data_file, file_desc="", data_dir=None):
  """
  Upload a file to Colab.
  Parameters
  ---------
  data_file: String.  File name: name of file to be stored on Colab.
      WARNING: this name is used ONLY to detect an existing file on Colab

     If the file already exists on Colab: it will not be uploaded again.

     If it does not exist:
     - user will be prompted to identify on file (on the local -- non-Colab -- filesystem) to uploaded.
     - The name of the local file will be used as the name on Colab


  file_desc: String.  Description of file that will be displayed in helpful messages.

  data-dir: String.  Name of Colab directory where file will be uploaded to.

  Returns
  -------
  None
  """
  
  # What directory are we in ?
  notebook_dir = os.getcwd()

  print("Current directory is: ", notebook_dir)

  # Check that the notebook directory is in sys.path
  # This is needed for the import of the data_file to succeed
  if not notebook_dir in sys.path:
    print(f"Adding {notebook_dir} to sys.path")

    sys.path.append(notebook_dir)

  if data_dir is not None:
    data_path     = Path(notebook_dir) / data_dir
    datafile_path = Path(notebook_dir) / data_dir / data_file
  else:
    data_path = Path(notebook_dir)
    datafile_path = Path(notebook_dir) / data_file


  if not data_path.is_dir():
    print(f"Creating the {data_dir} directory")
    os.makedirs(data_path)

  if not datafile_path.exists():
    print(f"Upload the {file_desc} file: {data_file} to directory {data_path}")
    print("\tIf file is large it may take a long time to upload.  Make sure it is completely uploaded before proceeding")

    print()
    print("\tAs an alternative: place the file on a Google Drive and mount the drive.")
    print("\t\tYou will have to add the path to the directory to sys.path -- see code above for modifying sys.path")

    # We will upload to the directory stored in variable data_path
    # This will necessitate changing the current directory; we will save it and restore it after the upload
    current_dir = os.getcwd()
    os.chdir(data_path)

    if IN_COLAB():
        from google.colab import files
        _= files.upload()
    else:
        print(f"Upload the {file_desc} file: {data_file} to directory {data_path}.")
        uploaded = input("Press ENTER when done.")

    # Restore the current working directory to the original directory
    os.chdir(current_dir)

from pathlib import Path


notebook_dir = os.getcwd()

def get_API_token(token_file=f"/{notebook_dir}/hf.token"):
    """
    Read file containing token

    Paramters
    ---------
    token_file: String.   Name of file

    Returns
    -------
    token: String.   Content of the file
    """
    
    # Check for file containing API token to HuggingFace
    p = Path(token_file).expanduser()
    if not p.exists():
      print(f"Token file {p} not found.")
      return

    with open(token_file, 'r') as fp:
        token = fp.read()

    # Remove trailing newline
    token = token.rstrip()

    return token

def install_required(required_list):
    """
    Install required packages via pip install

    The packages will be installed if they are not already present

    Parameters
    ----------
    required_list: List  of packages that are necessary
    
    """
    # Derived from: https://stackoverflow.com/a/44210735

    import pkg_resources
    import sys
    import subprocess

    # Convert required_list to a set
    # Use set difference to find packages that need to be installed
    required = set(required_list)
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        python = sys.executable
        rc = subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)

        if rc == 0:
            print("Installed: ", ", ".join( list(missing) ) )
    else:
        print("All packages already present: ", ", ".join(required_list) )
