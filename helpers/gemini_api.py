import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv

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
        response_text = response.text.strip()
        return response_text
    except Exception as e:
        print(f"Gemini API error: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def load_prompt_template(prompt_file_path=None):
    """
    Load the character generation prompt template from file
    
    Args:
        prompt_file_path: Path to prompt file. If None, uses default path.
        
    Returns:
        str: The prompt template with {json_content} placeholder
    """
    if prompt_file_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(os.path.dirname(current_dir), "prompts", "single_character_generation_prompt.txt") # single_character... or character_generation_prompt.txt
    
    try:
        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Warning: Prompt file not found at {prompt_file_path}, using default prompt")
        # Fallback to original hardcoded prompt
        return """I'm going to provide you the lyrics of a song and an interpretation of those lyrics. The interpretations are a descriptive interpretation to be used as prompts for image generation. The lyrics are marked as "text" and the interpretations as "prompt". Your task is to:

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

{{json_content}}"""

def generate_characters(model, json_file_path, output_dir="./", prompt_file_path=None):
    """
    Generate characters from lyrics interpretation file using Gemini
    
    Args:
        model: Gemini model instance
        json_file_path: Path to lyrics interpretation JSON (lyric2prompt)
        output_dir: Directory to save output files
        prompt_file_path: Path to custom prompt file (optional)
        
    Returns:
        tuple: (success: bool, updated_prompts_path: str, characters_path: str)
    """
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as f:
            lyrics_data = json.load(f)
        
        # Convert to string for the prompt
        json_content = json.dumps(lyrics_data, indent=2)
        
        # Load prompt template from file
        prompt_template = load_prompt_template(prompt_file_path)
        
        # Format the prompt with the JSON content
        prompt = prompt_template.format(json_content=json_content)
        
        # Call Gemini
        gemini_response = call_gemini(model, prompt)
        
        if gemini_response is None:
            print("Error: Failed to get response from Gemini")
            return False, None, None
        
        # Validate and save the response
        return validate_and_save_gemini_response(gemini_response, json_file_path, output_dir)
        
    except Exception as e:
        print(f"ERROR in generate_characters: {e}")
        print(f"Exception type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False, None, None

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
        # First, try to extract JSON from markdown code blocks
        pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(pattern, gemini_response, re.DOTALL)
        
        updated_prompts = None
        characters = None
        
        if len(matches) == 2:
            # New format: Two separate JSON blocks in markdown
            print("Processing response with two JSON blocks in markdown")
            try:
                updated_prompts = json.loads(matches[0].strip())
                characters = json.loads(matches[1].strip())
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in two-block markdown format: {e}")
                return False, None, None
                
        elif len(matches) == 1:
            # Old format: Single JSON block with characters at the end
            print("Processing response with single JSON block in markdown (legacy format)")
            try:
                cleaned_response = matches[0].strip()
                
                # Try to find the character list at the end
                character_pattern = r'\[\s*\{[^}]*"name"[^}]*"description"[^}]*\}[^\]]*\]'
                character_matches = list(re.finditer(character_pattern, cleaned_response, re.DOTALL))
                
                if not character_matches:
                    print("Error: No character list found in single block response")
                    return False, None, None
                
                # Get the last character list match
                last_character_match = character_matches[-1]
                character_json_str = last_character_match.group()
                
                # Remove character list from response to get updated prompts
                updated_prompts_str = cleaned_response[:last_character_match.start()].strip()
                
                # Parse both parts
                characters = json.loads(character_json_str)
                updated_prompts = json.loads(updated_prompts_str)
                
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in single-block markdown format: {e}")
                return False, None, None
        else:
            # No markdown code blocks found - try to parse as raw JSON
            print("No markdown code blocks found, attempting to parse as raw JSON")
            
            # Look for two separate JSON arrays in the raw response
            json_array_pattern = r'\[\s*\{.*?\}\s*\]'
            json_arrays = re.findall(json_array_pattern, gemini_response, re.DOTALL)
            
            if len(json_arrays) == 2:
                print("Found two JSON arrays in raw response")
                try:
                    updated_prompts = json.loads(json_arrays[0])
                    characters = json.loads(json_arrays[1])
                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in raw format: {e}")
                    return False, None, None
            else:
                print(f"Error: Expected 2 JSON arrays in raw format, found {len(json_arrays)}")
                print("Available arrays:", [arr[:100] + "..." if len(arr) > 100 else arr for arr in json_arrays])
                return False, None, None
        
        # Validate updated prompts
        if not isinstance(updated_prompts, list):
            print("Error: Updated prompts must be a list")
            return False, None, None
            
        # Validate characters
        if not isinstance(characters, list):
            print("Error: Characters must be a list")
            return False, None, None
        
        if len(characters) > 5:
            print(f"Warning: Found {len(characters)} characters, expected max 5")
        
        # Validate character structure
        for char in characters:
            if not isinstance(char, dict) or "name" not in char or "description" not in char:
                print(f"Error: Invalid character structure: {char}")
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

