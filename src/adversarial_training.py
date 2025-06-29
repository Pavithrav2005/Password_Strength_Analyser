"""
Adversarial Training Module
Generates adversarial examples to improve model robustness
"""

import random
import string
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class AdversarialPasswordGenerator:
    """Generate adversarial password examples for robust training"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Common character substitutions that users think make passwords stronger
        self.leet_substitutions = {
            'a': ['@', '4'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['$', '5'],
            'l': ['1'],
            't': ['7'],
            'g': ['9'],
            'b': ['6'],
            'z': ['2']
        }
        
        # Common password suffixes that users add
        self.common_suffixes = [
            '123', '!', '123!', '1', '12', '1234', '2024', '2023', '2022',
            '21', '22', '23', '24', '!@#', '***', '111', '000', '999'
        ]
        
        # Common password prefixes
        self.common_prefixes = [
            'my', 'the', 'a', 'i', 'we', '123', '!', '@'
        ]
        
        # Predictable transformations
        self.case_transformations = [
            'capitalize_first',
            'capitalize_all',
            'alternate_case',
            'capitalize_last'
        ]
        
    def generate_leet_speak_adversarial(self, password: str) -> str:
        """Convert password to leet speak (looks stronger but isn't)"""
        adversarial = password.lower()
        
        # Apply leet substitutions randomly
        for char, substitutes in self.leet_substitutions.items():
            if char in adversarial and random.random() < 0.7:
                substitute = random.choice(substitutes)
                # Replace only some occurrences to make it more realistic
                char_positions = [i for i, c in enumerate(adversarial) if c == char]
                if char_positions:
                    pos_to_replace = random.choice(char_positions)
                    adversarial = adversarial[:pos_to_replace] + substitute + adversarial[pos_to_replace+1:]
        
        return adversarial
    
    def add_predictable_suffix(self, password: str) -> str:
        """Add common suffixes that users think add security"""
        suffix = random.choice(self.common_suffixes)
        return password + suffix
    
    def add_predictable_prefix(self, password: str) -> str:
        """Add common prefixes"""
        prefix = random.choice(self.common_prefixes)
        return prefix + password
    
    def apply_case_transformation(self, password: str) -> str:
        """Apply predictable case transformations"""
        transformation = random.choice(self.case_transformations)
        
        if transformation == 'capitalize_first':
            return password.capitalize()
        elif transformation == 'capitalize_all':
            return password.upper()
        elif transformation == 'alternate_case':
            return ''.join(c.upper() if i % 2 == 0 else c.lower() 
                          for i, c in enumerate(password))
        elif transformation == 'capitalize_last':
            if len(password) > 0:
                return password[:-1] + password[-1].upper()
        
        return password
    
    def keyboard_shift_attack(self, password: str) -> str:
        """Simulate keyboard shift patterns"""
        # Define keyboard layout shifts
        shift_map = {
            '1': '!', '2': '@', '3': '#', '4': '$', '5': '%',
            '6': '^', '7': '&', '8': '*', '9': '(', '0': ')',
            'q': 'w', 'w': 'e', 'e': 'r', 'r': 't', 't': 'y',
            'a': 's', 's': 'd', 'd': 'f', 'f': 'g', 'g': 'h',
            'z': 'x', 'x': 'c', 'c': 'v', 'v': 'b', 'b': 'n'
        }
        
        shifted = ''
        for char in password.lower():
            if char in shift_map and random.random() < 0.3:
                shifted += shift_map[char]
            else:
                shifted += char
        
        return shifted
    
    def year_injection_attack(self, password: str) -> str:
        """Inject years that seem to add complexity but are predictable"""
        years = [str(year) for year in range(1950, 2025)]
        birth_years = [str(year) for year in range(1960, 2005)]  # More likely birth years
        
        # Choose more predictable years
        if random.random() < 0.7:
            year = random.choice(birth_years)
        else:
            year = random.choice(years)
        
        # Insert year at random position
        positions = ['beginning', 'middle', 'end']
        position = random.choice(positions)
        
        if position == 'beginning':
            return year + password
        elif position == 'end':
            return password + year
        else:  # middle
            mid = len(password) // 2
            return password[:mid] + year + password[mid:]
    
    def common_word_injection(self, password: str) -> str:
        """Inject common words that seem to add complexity"""
        common_words = [
            'love', 'hate', 'cool', 'hot', 'new', 'old', 'big', 'small',
            'good', 'bad', 'happy', 'sad', 'fast', 'slow', 'high', 'low'
        ]
        
        word = random.choice(common_words)
        
        # Randomly capitalize or apply leet speak to the word
        if random.random() < 0.5:
            word = word.capitalize()
        if random.random() < 0.3:
            word = self.generate_leet_speak_adversarial(word)
        
        # Insert at random position
        if random.random() < 0.5:
            return word + password
        else:
            return password + word
    
    def padding_attack(self, password: str) -> str:
        """Add padding characters that don't add real security"""
        padding_chars = ['0', '1', '!', '*', '#', '=', '-', '_']
        padding_length = random.randint(1, 4)
        
        left_padding = ''.join(random.choices(padding_chars, k=padding_length//2))
        right_padding = ''.join(random.choices(padding_chars, k=padding_length - padding_length//2))
        
        return left_padding + password + right_padding
    
    def generate_multiple_adversarials(self, password: str, n_variants: int = 5) -> List[str]:
        """Generate multiple adversarial variants of a password"""
        adversarials = []
        
        # Define transformation functions
        transformations = [
            self.generate_leet_speak_adversarial,
            self.add_predictable_suffix,
            self.add_predictable_prefix,
            self.apply_case_transformation,
            self.keyboard_shift_attack,
            self.year_injection_attack,
            self.common_word_injection,
            self.padding_attack
        ]
        
        for _ in range(n_variants):
            # Apply 1-3 random transformations
            n_transforms = random.randint(1, 3)
            current_password = password
            
            selected_transforms = random.sample(transformations, n_transforms)
            for transform in selected_transforms:
                current_password = transform(current_password)
            
            adversarials.append(current_password)
        
        return list(set(adversarials))  # Remove duplicates
    
    def create_adversarial_dataset(self, original_passwords: List[str], 
                                 original_labels: List[str]) -> Tuple[List[str], List[str]]:
        """Create adversarial dataset from original passwords"""
        adversarial_passwords = []
        adversarial_labels = []
        
        for password, label in zip(original_passwords, original_labels):
            # Generate adversarial variants
            variants = self.generate_multiple_adversarials(password, n_variants=3)
            
            for variant in variants:
                adversarial_passwords.append(variant)
                # Keep original label - these should still be classified correctly
                adversarial_labels.append(label)
        
        return adversarial_passwords, adversarial_labels
    
    def generate_deceptive_strong_passwords(self, n_passwords: int = 100) -> List[str]:
        """Generate passwords that look strong but are actually weak"""
        deceptive_passwords = []
        
        # Base weak passwords
        weak_bases = [
            'password', 'admin', 'user', 'login', 'welcome',
            'hello', 'world', 'test', '123456', 'qwerty'
        ]
        
        for _ in range(n_passwords):
            base = random.choice(weak_bases)
            
            # Apply multiple "strengthening" transformations
            deceptive = base
            
            # Leet speak
            if random.random() < 0.8:
                deceptive = self.generate_leet_speak_adversarial(deceptive)
            
            # Add year
            if random.random() < 0.7:
                deceptive = self.year_injection_attack(deceptive)
            
            # Add suffix
            if random.random() < 0.9:
                deceptive = self.add_predictable_suffix(deceptive)
            
            # Case transformation
            if random.random() < 0.6:
                deceptive = self.apply_case_transformation(deceptive)
            
            # Padding
            if random.random() < 0.4:
                deceptive = self.padding_attack(deceptive)
            
            deceptive_passwords.append(deceptive)
        
        return deceptive_passwords
    
    def evaluate_adversarial_robustness(self, model, password_list: List[str], 
                                      true_labels: List[str]) -> Dict[str, float]:
        """Evaluate model robustness against adversarial examples"""
        from feature_extraction import PasswordFeatureExtractor
        import pandas as pd
        
        extractor = PasswordFeatureExtractor()
        
        # Generate adversarial examples
        adversarial_passwords, adversarial_labels = self.create_adversarial_dataset(
            password_list, true_labels
        )
        
        # Extract features for adversarial examples
        adversarial_features = []
        for password in adversarial_passwords:
            features = extractor.extract_features(password)
            adversarial_features.append(features)
        
        adversarial_df = pd.DataFrame(adversarial_features)
        
        # Predict on adversarial examples
        adversarial_predictions = model.predict(adversarial_df)
        
        # Calculate accuracy
        correct_predictions = sum(1 for pred, true in zip(adversarial_predictions, adversarial_labels) 
                                if pred == true)
        accuracy = correct_predictions / len(adversarial_labels)
        
        # Calculate robustness metrics
        original_features = []
        for password in password_list:
            features = extractor.extract_features(password)
            original_features.append(features)
        
        original_df = pd.DataFrame(original_features)
        original_predictions = model.predict(original_df)
        
        original_accuracy = sum(1 for pred, true in zip(original_predictions, true_labels) 
                              if pred == true) / len(true_labels)
        
        robustness_drop = original_accuracy - accuracy
        
        return {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': accuracy,
            'robustness_drop': robustness_drop,
            'total_adversarial_examples': len(adversarial_passwords)
        }


def augment_training_data_with_adversarials(df, augmentation_factor=0.3):
    """Augment training dataset with adversarial examples"""
    adversarial_gen = AdversarialPasswordGenerator()
    
    # Sample passwords to create adversarials from
    n_adversarials = int(len(df) * augmentation_factor)
    sample_df = df.sample(n=min(n_adversarials, len(df)))
    
    adversarial_passwords, adversarial_labels = adversarial_gen.create_adversarial_dataset(
        sample_df['Password'].tolist(),
        sample_df['Strength_Label'].tolist()
    )
    
    # Create adversarial dataframe
    adversarial_df = pd.DataFrame({
        'Password': adversarial_passwords,
        'Strength_Label': adversarial_labels
    })
    
    # Calculate crack times for adversarial passwords (they should still be weak)
    from data_generator import PasswordDataGenerator
    gen = PasswordDataGenerator()
    
    adversarial_df['Crack_Time_Sec'] = adversarial_df['Password'].apply(gen.calculate_crack_time)
    adversarial_df['Strength_Label'] = adversarial_df['Crack_Time_Sec'].apply(gen.classify_strength)
    
    # Combine with original dataset
    augmented_df = pd.concat([df, adversarial_df], ignore_index=True)
    augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    return augmented_df


if __name__ == "__main__":
    # Example usage
    adversarial_gen = AdversarialPasswordGenerator()
    
    # Test password
    test_password = "password"
    
    print(f"Original password: {test_password}")
    print("Adversarial variants:")
    
    variants = adversarial_gen.generate_multiple_adversarials(test_password, n_variants=5)
    for i, variant in enumerate(variants, 1):
        print(f"{i}. {variant}")
    
    print("\nDeceptive 'strong' passwords:")
    deceptive = adversarial_gen.generate_deceptive_strong_passwords(n_passwords=5)
    for i, pwd in enumerate(deceptive, 1):
        print(f"{i}. {pwd}")
