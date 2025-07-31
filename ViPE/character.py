import json
import re
import os

class Character:
    """A character object that tracks character information and in which lines they appear"""
    
    def __init__(self, name, description, line_occurrences=None, regularization_images=None, training_images=None):
        """
        Initialize a Character object
        
        Args:
            name (str): Name of the character
            description (str): Description of the character
            line_occurrences (list): List of integers representing line numbers where character appears
            regularization_images (str): Path to the regularization images
            training_images (str): Path to the training images
        """
        self.name = name
        self.description = description
        self.line_occurrences = line_occurrences if line_occurrences is not None else []
        self.regularization_images = regularization_images
        self.training_images = training_images
        self.model_path = None
    
    def add_occurrence(self, line_number):
        """Add a line number where this character appears"""
        if line_number not in self.line_occurrences:
            self.line_occurrences.append(line_number)
            self.line_occurrences.sort()  # Sort occurrences
    
    def appears_in_line(self, line_number):
        """Check if character appears in a specific line"""
        return line_number in self.line_occurrences
    
    def to_dict(self):
        """Return a dictionary representation of the Character object."""
        return {
            'name': self.name,
            'description': self.description,
            'line_occurrences': self.line_occurrences,
            'regularization_images': self.regularization_images,
            'training_images': self.training_images
        }
    
    def set_training_images_path(self, path):
        """Set the path to training images for this character"""
        self.training_images = path
    
    def set_regularization_images_path(self, path):
        """Set the path to regularization images for this character"""
        self.regularization_images = path
    
    def get_training_images_path(self):
        """Get the path to training images for this character"""
        return self.training_images
    
    def get_regularization_images_path(self):
        """Get the path to regularization images for this character"""
        return self.regularization_images
    
    def has_trained_model(self):
        """Check if this character has a trained model"""
        return self.model_path and os.path.exists(self.model_path)

    def get_model_path(self):
        """Get the path to the trained model"""
        return getattr(self, 'model_path', None)


def load_characters_from_json(json_file_path):
    """
    Load characters from JSON file and convert to Character objects
    
    Args:
        json_file_path (str): Path to the characters JSON file
        
    Returns:
        list: List of Character objects
    """
    try:
        with open(json_file_path, 'r') as f:
            characters_data = json.load(f)
        
        characters = []
        for char_data in characters_data:
            # Extract data from JSON
            name = char_data.get('name', '')
            description = char_data.get('description', '')
            line_occurrences = char_data.get('line_occurrences', [])
            regularization_images = char_data.get('regularization_images', None)
            training_images = char_data.get('training_images', None)
            
            # Create Character object with name
            character = Character(
                name=name,
                description=description,
                line_occurrences=line_occurrences,
                regularization_images=regularization_images,
                training_images=training_images
            )
            
            characters.append(character)
        
        print(f"Loaded {len(characters)} characters from {json_file_path}")
        return characters
        
    except FileNotFoundError:
        print(f"Characters file not found: {json_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing characters JSON: {e}")
        return []
    except Exception as e:
        print(f"Error loading characters: {e}")
        return []

def save_characters_to_json(characters, json_file_path):
    """
    Save Character objects to JSON file
    
    Args:
        characters (list): List of Character objects
        json_file_path (str): Path where to save the JSON file
    """
    try:
        characters_data = []
        for char in characters:
            char_dict = char.to_dict()
            characters_data.append(char_dict)
        
        with open(json_file_path, 'w') as f:
            json.dump(characters_data, f, indent=2)
        
        print(f"Saved {len(characters)} characters to {json_file_path}")
        
    except Exception as e:
        print(f"Error saving characters: {e}")

def update_character_occurrences(characters, lyric2prompt):
    """
    Scan prompts for character names in brackets and update their line occurrences
    
    Args:
        characters (list): List of Character objects
        lyric2prompt (list): List of dictionaries with 'text' and 'prompt' keys
        
    Returns:
        list: Updated list of Character objects with line occurrences
    """
    
    for line_index, line_data in enumerate(lyric2prompt):
        prompt = line_data.get('prompt', '')
        
        for character in characters:
            # Check for character name in angle brackets <CharacterName>
            bracketed_name = f"<{character.name}>"
            if bracketed_name in prompt:
                character.add_occurrence(line_index)
                print(f"Found '{bracketed_name}' in line {line_index}: {prompt[:50]}...")
    
    # Print summary
    for character in characters:
        print(f"Character '{character.name}' appears in {len(character.line_occurrences)} lines: {character.line_occurrences}")
    
    return characters

#    # generate characters using the ViPE interpretation and adjust the interpretation
#    def get_characters(lyric2prompt):
#    """Generate characters using the ViPE interpretation and adjust the interpretation"""
#    
#    #ask an AI to generate characters
#
#    #generate character objects for it
#        #name
#        #description
#        #occurences
#        #ask for training images or get them using get_training_images
#        #ask for regularization images, generate them or use public sets
#    
#    #adjust ViPE interpretations
#
#    return list_of_characters