#!/bin/bash

set_environment() {
    if [[ "$1" == "--test" ]]; then
        export ENVIRONMENT="TEST"
    else
        export ENVIRONMENT="PROD"
    fi
}
set_environment "$1"
echo "You are running in the '$ENVIRONMENT' environment."

cd "$(dirname "$0")/.."

# Check if the .env file exists
if [ ! -f .env ]; then
    echo "Error: .env file not found."
    exit 1
fi

export $(grep -v '^#' .env | xargs)

# run API bot
python3 bot.py