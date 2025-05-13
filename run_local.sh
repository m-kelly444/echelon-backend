#!/bin/bash
export FLASK_APP=app.py
export FLASK_ENV=development

# Use the absolute path to Python3 on your system
/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m gunicorn 'app:create_app()' --bind 0.0.0.0:5000 --reload
