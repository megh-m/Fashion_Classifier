#!/bin/bash
# periodic_retrain.sh

# This script will be called by cron to periodically retrain the model

# Set working directory
cd "$(dirname "$0")"

# Check if we have enough new user images since last retraining
LAST_RETRAIN_TIME=$(stat -c %Y models/model_metadata.json 2>/dev/null || echo 0)
CURRENT_TIME=$(date +%s)
TIME_DIFF=$((CURRENT_TIME - LAST_RETRAIN_TIME))

# Count new images since last retraining
NEW_IMAGES=$(find data/user-images -type f -newermt "@$LAST_RETRAIN_TIME" | wc -l)

echo "Time since last retraining: $TIME_DIFF seconds"
echo "New images since last retraining: $NEW_IMAGES"

# Retrain if either:
# 1. More than 3 hours since last retraining OR
# 2. At least 10 new images since last retraining
if [ $TIME_DIFF -gt 10800 ] || [ $NEW_IMAGES -ge 10 ]; then
    echo "Initiating model retraining..."
    
    # Call the retraining API endpoint
    curl -X POST "http://localhost:5000/retrain" \
      -H "Content-Type: application/x-www-form-urlencoded" \
      -d "api_key=your-secret-key"
    
    # Alternatively, call the retraining script directly
    # python retrain.py
    
    echo "Retraining initiated successfully."
else
    echo "Skipping retraining - not enough new data or too recent."
fi
