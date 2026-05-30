#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Scope Evaluation Script (scope_eval.py)
This is the main evaluation script that gets called by runall_scope.sh
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the evaluation system
try:
    from scope_evaluation import main as run_evaluation
    
    if __name__ == "__main__":
        run_evaluation()
        
except ImportError as e:
    print(f"❌ Error importing evaluation system: {e}")
    print("Please ensure scope_evaluation.py is in the same directory")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error running evaluation: {e}")
    sys.exit(1)
