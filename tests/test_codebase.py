import subprocess
import os
    import ast
import glob

def test_only_exception_files_modified():
    # print(f"Current working directory: {os.getcwd()}")  # Debugging line

    EXCEPTION_FILES = ["multigrid/envs/competative_red_blue_door.py",
     "multigrid/rllib/models.py", "multigrid/scripts/train.py",
     "multigrid/scripts/visualize.py", 
     "multigrid/scripts/train_ppo_cleanrl.py",
     "multigrid/scripts/visualize.py",
     "multigrid/utils/training_utilis.py"
     ]
    
    EXCEPTION_FOLDERS = ["submission/**", "notebooks/**"]

    for folder in EXCEPTION_FOLDERS:
        globbed_files = glob.glob(folder, recursive=True)
        # print(f"Adding files from folder {folder}: {globbed_files}")  # Debugging line
        EXCEPTION_FILES.extend(globbed_files)
        
    EXCEPTION_FILES = set(EXCEPTION_FILES)  # Converting to set for faster look-up

    # Get list of all files in the repository.
    all_files_result = subprocess.run(
        ["git", "ls-files"],
        capture_output=True,
        text=True
    )
    all_files_result.check_returncode()
    
    all_files = set(all_files_result.stdout.splitlines())
    
    # Remove exception files from all_files to create the locked_files set.
    locked_files = all_files - EXCEPTION_FILES


    # Get list of changed files between HEAD and its previous commit.
    changed_files_result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        capture_output=True,
        text=True
    )
    
    # If the git command fails, the test should fail.
    changed_files_result.check_returncode()

    changed_files = set(changed_files_result.stdout.splitlines())

    # Check if any locked file is in the changed files list.
    modified_locked_files = changed_files & locked_files
    assert not modified_locked_files, f"Locked files were modified: {', '.join(modified_locked_files)}"



import subprocess

ALLOWED_FUNCTIONS_IN_CLASS = {
    "MyClass": ["allowed_function_1", "allowed_function_2"],
}