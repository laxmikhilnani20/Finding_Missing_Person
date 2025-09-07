#!/usr/bin/env python3
import json

def fix_notebook():
    # Read the notebook
    with open('PBL-3.ipynb', 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Remove the widgets metadata that's causing validation issues
    if 'widgets' in notebook['metadata']:
        del notebook['metadata']['widgets']
        print("Removed widgets metadata")
    
    # Write the fixed notebook
    with open('PBL-3.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print("Fixed notebook saved successfully")

if __name__ == "__main__":
    fix_notebook()
