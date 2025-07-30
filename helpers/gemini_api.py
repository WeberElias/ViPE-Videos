import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv  # pip install python-dotenv

def setup_gemini():
    """Initialize Gemini with API key from environment"""
    print("Initialize Gemini")
    load_dotenv()  # Load .env file
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_AI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def call_gemini(model, prompt):
    """Basic Gemini API call"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None

def generate_characters(model, json_file_path, output_dir="./"):
    """
    Generate characters from lyrics interpretation file using Gemini
    
    Args:
        model: Gemini model instance
        json_file_path: Path to lyrics interpretation JSON (lyric2prompt)
        output_dir: Directory to save output files
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        lyrics_data = json.load(f)
    
    # Convert to string for the prompt
    json_content = json.dumps(lyrics_data, indent=2)
    
    prompt = f"""I'm going to provide you the lyrics of a song and an interpretation of those lyrics. The interpretations are a descriptive interpretation to be used as prompts for image generation. The lyrics are marked as "text" and the interpretations as "prompt". Your task is to:

Create a short list of no more than five characters with very brief visual descriptions, like:

[
  {{
    "name": "Alice",
    "description": "Woman, middle-aged, brown hair, green eyes"
  }}
]

Use this exact JSON structure. Do not use real persons. Characters must be named (e.g., <Alice>) and referenced by name in the prompts. Reuse existing characters whenever possible. Adjust prompts to incorporate characters and improve coherence, while keeping the original descriptive intent.

If an interpretation does not clearly connect to any character, adjust the prompt to make it fit one of the existing or newly created characters.

Return your output in the exact same format and structure as the interpretation file I sent, with only the adjusted prompts changed.

At the end of the file, include the character list in the JSON format specified above.

Constraints:

- No more than 5 characters total.
- No real persons.
- No explanations or extra commentary.
- Character descriptions must be very short and visual only.
- Final character list must be in JSON format as shown.
- DO NOT create more than five characters

Here is the lyrics interpretation file:

{json_content}"""
    
    # Call Gemini
    gemini_response = call_gemini(model, prompt)
    
    if gemini_response is None:
        print("Error: Failed to get response from Gemini")
        return False, None, None
    
    print("Gemini response: \n" + gemini_response)
    # Validate and save the response
    return validate_and_save_gemini_response(gemini_response, json_file_path, output_dir)

def validate_and_save_gemini_response(gemini_response, original_json_path, output_dir="./"):
    """
    Validate Gemini response and split into two JSON files
    
    Args:
        gemini_response: Raw response from Gemini
        original_json_path: Path to original JSON file (for fallback)
        output_dir: Directory to save output files
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    try:
        # Clean the response - remove markdown code blocks if present
        cleaned_response = gemini_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Try to find the character list at the end
        # Look for the last JSON array in the response
        character_pattern = r'\[\s*\{[^}]*"name"[^}]*"description"[^}]*\}[^\]]*\]'
        character_matches = list(re.finditer(character_pattern, cleaned_response, re.DOTALL))
        
        if not character_matches:
            print("Error: No character list found in response")
            return False, None, None
        
        # Get the last character list match
        last_character_match = character_matches[-1]
        character_json_str = last_character_match.group()
        
        # Remove character list from response to get updated prompts
        updated_prompts_str = cleaned_response[:last_character_match.start()].strip()
        
        # Validate character JSON
        try:
            characters = json.loads(character_json_str)
            if not isinstance(characters, list):
                raise ValueError("Characters must be a list")
            
            if len(characters) > 5:
                print(f"Warning: Found {len(characters)} characters, expected max 5")
            
            # Validate character structure
            for char in characters:
                if not isinstance(char, dict) or "name" not in char or "description" not in char:
                    raise ValueError("Invalid character structure")
                    
        except json.JSONDecodeError as e:
            print(f"Error: Invalid character JSON: {e}")
            return False, None, None
        
        # Validate updated prompts JSON
        try:
            updated_prompts = json.loads(updated_prompts_str)
            if not isinstance(updated_prompts, list):
                raise ValueError("Updated prompts must be a list")
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid updated prompts JSON: {e}")
            return False, None, None
        
        # Save files
        base_name = os.path.splitext(os.path.basename(original_json_path))[0]
        
        updated_prompts_path = os.path.join(output_dir, f"{base_name}_with_characters.json")
        characters_path = os.path.join(output_dir, f"{base_name}_characters.json")
        
        # Save updated prompts
        with open(updated_prompts_path, 'w') as f:
            json.dump(updated_prompts, f, indent=2)
        
        # Save characters
        with open(characters_path, 'w') as f:
            json.dump(characters, f, indent=2)
        
        print(f"Successfully saved:")
        print(f"  Updated prompts: {updated_prompts_path}")
        print(f"  Characters: {characters_path}")
        print(f"  Found {len(characters)} characters")
        
        return True, updated_prompts_path, characters_path
        
    except Exception as e:
        print(f"Error processing Gemini response: {e}")
        return False, None, None

def process_lyrics_with_characters(model, json_file_path, output_dir="./"):
    """
    Complete workflow: generate characters and save results
    
    Args:
        model: Gemini model instance
        json_file_path: Path to lyrics interpretation JSON
        output_dir: Directory to save output files
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    # Generate characters
    gemini_response = generate_characters_from_lyrics(model, json_file_path)
    
    if gemini_response is None:
        print("Error: Failed to get response from Gemini")
        return False, None, None
    
    # Validate and save
    return validate_and_save_gemini_response(gemini_response, json_file_path, output_dir)

