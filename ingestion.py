import whisper
import os

def transcribe_to_data_folder(audio_path, model_size="base"):
    """
    Transcribes audio and saves it to the /data folder for the RAG pipeline.
    Model sizes: 'tiny', 'base', 'small', 'medium', 'large'
    """
    print(f"--- Loading Whisper {model_size} model ---")
    model = whisper.load_model(model_size)
    
    print(f"--- Transcribing: {audio_path} ---")
    # fp16=False is safer if you don't have a high-end GPU
    result = model.transcribe(audio_path, fp16=False) 
    
    # Create the filename for the text output
    base_name = os.path.basename(audio_path).split('.')[0]
    output_path = f"data/{base_name}.txt"
    
    # Ensure /data directory exists
    os.makedirs("data", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
        
    print(f"--- Done! Transcript saved to {output_path} ---")
    return output_path

# bla bla bla

# Example usage:
# transcribe_to_data_folder("meeting_recordings/weekly_sync.mp3")