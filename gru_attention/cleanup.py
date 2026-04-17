import os

def clean_specific_folders():
    """
    Strict cleanup: Deletes files ONLY within 'gru_attention/data' and 'gru_attention/report'.
    """
    # Get the absolute path of the directory where this script is located (gru_attention)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the specific target subfolders
    targets = ['trained_models', 'report']
    
    print(f"🔍 Current Workspace: {base_dir}")
    print("🧹 Initializing strict cleanup of 'trained_models' and 'report' folders...\n")

    for folder_name in targets:
        target_path = os.path.join(base_dir, folder_name)
        
        if os.path.exists(target_path) and os.path.isdir(target_path):
            files = os.listdir(target_path)
            
            if not files:
                print(f"✅ {folder_name.upper()}: Already empty.")
                continue
            
            print(f"🗑️  Clearing files in {folder_name.upper()}...")
            file_count = 0
            
            for filename in files:
                file_path = os.path.join(target_path, filename)
                try:
                    # We only delete files to be safe
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        file_count += 1
                except Exception as e:
                    print(f"❌ Could not delete {filename}: {e}")
            
            print(f"✨ Successfully removed {file_count} files from {folder_name}.")
        else:
            print(f"⚠️  Folder not found, skipping: {folder_name}")

    print("\n🎉 Cleanup finished. Project folders are reset.")

if __name__ == "__main__":
    # Security confirmation
    print("--- PROJECT CLEANUP TOOL ---")
    confirm = input("Confirm: Delete ALL files in 'trained_models' and 'report'? (y/n): ")
    if confirm.lower() == 'y':
        clean_specific_folders()
    else:
        print("🚫 Operation aborted by user.")