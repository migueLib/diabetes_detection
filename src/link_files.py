import os
from pathlib import Path
from tqdm import tqdm

def link_files(source, destin, fnames):
    # Make source and destination paths absolute
    source = os.path.abspath(source)
    destin = os.path.abspath(destin)
        
    # Create directory for destination if it does not exist
    Path(destin).mkdir(parents=True, exist_ok=True)
    
    # Iterate over filelist and link files
    for f in tqdm(fnames):
        src = os.path.join(source, f)
        dst = os.path.join(destin, f)
        os.symlink(src, dst)
    