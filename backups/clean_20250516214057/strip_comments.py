#!/usr/bin/env python3
"""
Script to remove comments and docstrings from Python code.
This preserves functionality while removing explanations.
"""

import sys
import tokenize
import io
import re
import os

def remove_docstrings_and_comments(source):
    """Remove docstrings and comments from Python source code."""
    io_obj = io.StringIO(source)
    out = io.StringIO()
    
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    
    tokgen = tokenize.generate_tokens(io_obj.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if toktype == tokenize.COMMENT:
            # Completely ignore comments
            continue
        
        if toktype == tokenize.STRING:
            if prev_toktype == tokenize.INDENT:
                # This is likely a docstring - strip it
                continue
            
            # Check if this is a multiline string but not a docstring
            if slineno > last_lineno:
                if scol > 0 and ttext.startswith(('"""', "'''")):
                    # Check if this has the form of a docstring
                    is_docstring = False
                    
                    # If indented and at the start of a function/class/method
                    if prev_toktype == tokenize.INDENT or prev_toktype == tokenize.NEWLINE:
                        is_docstring = True
                    
                    if is_docstring:
                        continue
        
        # Add a newline if needed
        if slineno > last_lineno:
            last_col = 0
        
        # Handle spaces before the current token
        if scol > last_col:
            out.write(" " * (scol - last_col))
        
        # Write the token's text
        out.write(ttext)
        
        # Update position tracking
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno
    
    # Process result to remove remaining docstrings
    result = out.getvalue()
    
    # Handle triple-quoted docstrings that may have been missed
    result = re.sub(r'(?m)^(\s*)("""|\'\'\').*?("""|\'\'\')', r'\1pass', result, flags=re.DOTALL)
    
    # Remove empty lines resulting from comment removal (up to 2 consecutive empty lines)
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    
    return result

def process_file(file_path):
    """Process a single Python file to remove comments."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Skip empty files
        if not source.strip():
            return False, "File is empty"
        
        cleaned_source = remove_docstrings_and_comments(source)
        
        # Write output
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_source)
        
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    if len(sys.argv) < 2:
        print("Usage: python strip_comments.py <file.py>")
        return
    
    file_path = sys.argv[1]
    success, error = process_file(file_path)
    
    if success:
        print(f"Successfully processed: {file_path}")
    else:
        print(f"Error processing {file_path}: {error}")

if __name__ == "__main__":
    main()
