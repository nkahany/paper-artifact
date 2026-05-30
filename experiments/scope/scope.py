#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Scope Ambiguity Resolution Script (scope.py)
This is the main script that gets called by runall_scope.sh
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the complete system
try:
    from scope_complete_system import main as run_scope_resolution
    
    if __name__ == "__main__":
        run_scope_resolution()
        
except ImportError as e:
    print(f"❌ Error importing scope resolution system: {e}")
    print("Please ensure scope_complete_system.py is in the same directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running scope resolution: {e}")
    sys.exit(1)
