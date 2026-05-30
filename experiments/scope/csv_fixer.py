# csv_fixer.py - Run this first to clean your CSV files
import pandas as pd

def fix_csv_encoding(filename):
    """Fix CSV encoding issues"""
    try:
        # Read file in binary mode and replace problematic bytes
        with open(filename, 'rb') as f:
            content = f.read()
        
        # Replace the problematic byte 0xac with a space or remove it
        content = content.replace(b'\xac', b' ')  # Replace with space
        content = content.replace(b'\x92', b"'")  # Replace smart quote
        content = content.replace(b'\x93', b'"')  # Replace smart quote
        content = content.replace(b'\x94', b'"')  # Replace smart quote
        
        # Write back as UTF-8
        fixed_filename = f"fixed_{filename}"
        with open(fixed_filename, 'wb') as f:
            f.write(content)
        
        # Test if it works
        df = pd.read_csv(fixed_filename, encoding='utf-8')
        print(f"? Fixed {filename} -> {fixed_filename} ({len(df)} rows)")
        return fixed_filename
        
    except Exception as e:
        print(f"? Error fixing {filename}: {e}")
        return None

# Fix your files
fix_csv_encoding('scope_fixed.csv')
fix_csv_encoding('exp1b_base_dataset.csv')