import subprocess
import time
import pandas as pd
import os
import sys
import io
import re

# Force the standard output and error to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Define the real value (reference) for WER calculation
real_value = "مرحباً، هذا تسجيل صوتي نحن نحتاج النجدة في مدينة بيروت في لبنان"

# Models to simulate (ignoring .en models)
models = [
    'tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'
]

# Function to sanitize the transcription by removing numbers, timestamps, and special characters
def sanitize_transcription(text):
    # Remove numbers and anything that looks like a timestamp
    cleaned_text = re.sub(r'\d+\.\d+|\d+', '', text)  # Remove numbers (e.g., 1546.1102)
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Remove newlines
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace multiple spaces with a single space
    cleaned_text = cleaned_text.strip()  # Remove leading/trailing spaces
    return cleaned_text

# Function to calculate WER using a brute-force approach
def calculate_wer_bruteforce(reference, hypothesis):
    # Split reference and hypothesis into words
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()

    # Initialize the counts for substitutions, insertions, and deletions
    substitutions = 0
    deletions = 0
    insertions = 0

    # Make two pointers to track the current word in each list
    i, j = 0, 0
    
    # Compare words from both reference and hypothesis
    while i < len(reference_words) and j < len(hypothesis_words):
        # If words match, move to the next word in both lists
        if reference_words[i] == hypothesis_words[j]:
            i += 1
            j += 1
        # If words don't match, count it as a substitution and move to the next word in both lists
        elif reference_words[i] != hypothesis_words[j]:
            substitutions += 1
            i += 1
            j += 1

    # Handle cases where one of the lists is longer (insertions or deletions)
    while i < len(reference_words):  # Remaining reference words are deletions
        deletions += 1
        i += 1

    while j < len(hypothesis_words):  # Remaining hypothesis words are insertions
        insertions += 1
        j += 1

    # Total number of errors
    total_errors = substitutions + deletions + insertions

    # Calculate WER
    wer = total_errors / len(reference_words) if len(reference_words) > 0 else 0.0
    return wer

# Function to run the model using the simulstreaming command
def run_model(model_name):
    print(f"Running model: {model_name}")
    
    # Command to run the model
    command = f"python simulstreaming_whisper.py output.wav --language ar --model {model_name} -l ERROR"
    
    # Start the timer
    start_time = time.time()

    try:
        # Set the environment to use UTF-8 encoding for subprocess
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        # Run the command with the updated environment
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True, env=env, encoding='utf-8')
        
        # Check if we have output in stdout, if not, handle the case
        if result.stdout:
            transcribed_text = result.stdout.strip()
        else:
            transcribed_text = "No transcription received"
        
        # Print both stdout and stderr for debugging
        print("Standard Output:")
        print(result.stdout)
        print("Standard Error (if any):")
        print(result.stderr)
        
    except subprocess.CalledProcessError as e:
        # If subprocess fails, print the error and capture stderr
        print(f"Error running command {command}: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return {"Model": model_name, "Transcription": "Error", "Time (s)": 0, "WER": 1.0}

    # Measure time taken
    end_time = time.time()
    duration = end_time - start_time

    # Sanitize the transcription (clean it from timestamps, numbers, and special characters)
    sanitized_text = sanitize_transcription(transcribed_text)

    # Calculate WER using brute-force method
    wer = calculate_wer_bruteforce(real_value, sanitized_text)

    return {
        "Model": model_name,
        "Transcription": sanitized_text,
        "Time (s)": duration,
        "WER": wer
    }

# Collect results for each model
benchmark_results = []

# Run simulations for each model
for model_name in models:
    print(f"Processing model {model_name}...")
    model_result = run_model(model_name)
    benchmark_results.append(model_result)

# Convert the results to a DataFrame
df = pd.DataFrame(benchmark_results)

# Add the expected value to the DataFrame
df['Expected Transcription'] = real_value

# File path for the output Excel file
output_file = 'model_benchmark_results_bruteforce.xlsx'

# Check if the file exists, and if so, remove it
if os.path.exists(output_file):
    print(f"File {output_file} already exists. Overwriting...")
    os.remove(output_file)

# Save the DataFrame to an Excel file
df.to_excel(output_file, index=False)

print(f"Benchmarking complete! Results saved to '{output_file}'")
