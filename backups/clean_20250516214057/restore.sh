#!/bin/bash
# Restore files from backup

echo "Restoring files from backup backups/clean_20250516214057..."
cp -r backups/clean_20250516214057/* ./
echo "Restore complete!"
