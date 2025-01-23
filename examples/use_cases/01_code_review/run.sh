#!/bin/bash

# Default values
CODE_PATH="."
OUTPUT_FILE=""
EXTENSIONS="py,js,ts,java,cpp,go,rb"

# Help message
show_help() {
    echo "Usage: $0 [OPTIONS] [PATH]"
    echo
    echo "Run automated code review on specified directory"
    echo
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  -o, --output FILE    Write results to FILE (default: stdout)"
    echo "  -e, --ext LIST       Comma-separated list of file extensions to process"
    echo "                       (default: $EXTENSIONS)"
    echo
    echo "Example:"
    echo "  $0 --ext py,js --output review.json ./src"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -e|--ext)
            EXTENSIONS="$2"
            shift 2
            ;;
        *)
            CODE_PATH="$1"
            shift
            ;;
    esac
done

# Validate code path
if [ ! -d "$CODE_PATH" ]; then
    echo "Error: Directory '$CODE_PATH' does not exist"
    exit 1
fi

# Build the command
CMD="ostruct"
CMD="$CMD --task @task.j2"
CMD="$CMD --schema schema.json"
CMD="$CMD --system-prompt @system_prompt.txt"
CMD="$CMD --dir code=$CODE_PATH"
CMD="$CMD --ext $EXTENSIONS"
CMD="$CMD --recursive"

# Add output file if specified
if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output-file $OUTPUT_FILE"
fi

# Run the command
echo "Running code review..."
echo "Directory: $CODE_PATH"
echo "Extensions: $EXTENSIONS"
if [ -n "$OUTPUT_FILE" ]; then
    echo "Output: $OUTPUT_FILE"
fi
echo

eval "$CMD" 