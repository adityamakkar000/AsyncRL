#!/bin/bash 

if [ -d "profile" ]; then
    echo "Profile directory already exists. Skipping download."
else
    mkdir profile
    echo "Downloading profile data from Google Cloud Storage..."
fi 

gcloud storage cp -r "gs://arl-experiments/profile/$1" ./profile
