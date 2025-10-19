#!/bin/bash

# Test script for Squash Coaching API
# Usage: ./test_api.sh path/to/video.mp4

set -e

API_URL="http://localhost:8000"
VIDEO_PATH="$1"

if [ -z "$VIDEO_PATH" ]; then
    echo "Usage: ./test_api.sh path/to/video.mp4"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "==================================="
echo "Squash Coaching API Test"
echo "==================================="

# 1. Health check
echo -e "\n1. Testing health check..."
curl -s $API_URL/ | jq .

# 2. Upload video
echo -e "\n2. Uploading video..."
UPLOAD_RESPONSE=$(curl -s -X POST $API_URL/upload -F "file=@$VIDEO_PATH;type=video/mp4")
echo $UPLOAD_RESPONSE | jq .

JOB_ID=$(echo $UPLOAD_RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 3. Trigger analysis
echo -e "\n3. Triggering analysis..."
curl -s -X POST $API_URL/analyze/$JOB_ID | jq .

# 4. Monitor status
echo -e "\n4. Monitoring status..."
while true; do
    STATUS_RESPONSE=$(curl -s $API_URL/status/$JOB_ID)
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')

    echo "Current status: $STATUS"

    if [ "$STATUS" == "completed" ]; then
        echo "✓ Analysis completed!"
        break
    elif [ "$STATUS" == "failed" ]; then
        echo "✗ Analysis failed!"
        echo $STATUS_RESPONSE | jq .
        exit 1
    fi

    sleep 5
done

# 5. Get results
echo -e "\n5. Fetching results..."
RESULTS=$(curl -s $API_URL/results/$JOB_ID)
echo $RESULTS | jq '.results | {total_rallies, avg_rally_duration}'

TOTAL_RALLIES=$(echo $RESULTS | jq -r '.results.total_rallies')
echo "Total rallies found: $TOTAL_RALLIES"

# 6. Download first rally video
if [ "$TOTAL_RALLIES" -gt 0 ]; then
    echo -e "\n6. Downloading rally 1 video..."
    curl -s $API_URL/videos/$JOB_ID/rallies/1 -o rally_1.mp4
    echo "✓ Rally video saved to rally_1.mp4"
fi

echo -e "\n==================================="
echo "Test completed successfully!"
echo "==================================="
