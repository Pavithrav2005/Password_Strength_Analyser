"""
Data Generator Module
Generates synthetic password dataset with strength labels
"""

import random
import string
import pandas as pd
import numpy as np
from faker import Faker
import math


class PasswordDataGenerator:
    """Generate synthetic password data for training"""
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)
        
        # Common weak passwords
        self.weak_passwords = [
            "123456", "password", "123456789", "12345678", "12345",
            "1234567", "1234567890", "qwerty", "abc123", "111111",
            "123123", "admin", "letmein", "welcome", "monkey",
            "password123", "qwerty123", "123qwe", "password1",
            "iloveyou", "princess", "rockyou", "1234", "12345678901",
            "abc", "password12", "computer", "123321", "1q2w3e4r"
        ]
        
        # Common words for medium strength passwords
        self.common_words = [
            "love", "happy", "family", "friend", "world", "home",
            "music", "heart", "peace", "smile", "dream", "hope",
            "trust", "faith", "power", "strong", "brave", "smart"
        ]
        
        # Keyboard patterns
        self.keyboard_patterns = [
            "qwerty", "asdf", "zxcv", "qwertyuiop", "asdfghjkl",
            "zxcvbnm", "1234567890", "0987654321"
        ]
        
    def generate_weak_password(self):
        """Generate a weak password"""
        strategies = [
            self._generate_from_weak_list,
            self._generate_simple_numeric,
            self._generate_simple_alphabetic,
            self._generate_keyboard_pattern,
            self._generate_repeated_chars,
            self._generate_common_word_simple
        ]
        
        strategy = random.choice(strategies)
        return strategy()
    
    def generate_medium_password(self):
        """Generate a medium strength password"""
        strategies = [
            self._generate_common_word_with_numbers,
            self._generate_name_with_year,
            self._generate_simple_substitution,
            self._generate_two_words_combined,
            self._generate_predictable_pattern
        ]
        
        strategy = random.choice(strategies)
        return strategy()
    
    def generate_strong_password(self):
        """Generate a strong password"""
        strategies = [
            self._generate_random_complex,
            self._generate_passphrase,
            self._generate_mixed_case_symbols,
            self._generate_long_random,
            self._generate_pronounceable_complex
        ]
        
        strategy = random.choice(strategies)
        return strategy()
    
    def _generate_from_weak_list(self):
        """Pick from known weak passwords"""
        return random.choice(self.weak_passwords)
    
    def _generate_simple_numeric(self):
        """Generate simple numeric passwords"""
        length = random.randint(4, 8)
        return ''.join(random.choices(string.digits, k=length))
    
    def _generate_simple_alphabetic(self):
        """Generate simple alphabetic passwords"""
        length = random.randint(4, 8)
        return ''.join(random.choices(string.ascii_lowercase, k=length))
    
    def _generate_keyboard_pattern(self):
        """Generate keyboard pattern passwords"""
        pattern = random.choice(self.keyboard_patterns)
        if random.random() < 0.3:
            pattern = pattern[::-1]  # Reverse pattern
        
        # Sometimes add numbers or symbols
        if random.random() < 0.4:
            pattern += str(random.randint(1, 999))
        
        return pattern
    
    def _generate_repeated_chars(self):
        """Generate passwords with repeated characters"""
        char = random.choice(string.ascii_lowercase + string.digits)
        length = random.randint(6, 12)
        return char * length
    
    def _generate_common_word_simple(self):
        """Generate common word with simple additions"""
        word = random.choice(self.common_words)
        
        # Add simple modifications
        if random.random() < 0.5:
            word += str(random.randint(1, 99))
        if random.random() < 0.3:
            word = word.capitalize()
        
        return word
    
    def _generate_common_word_with_numbers(self):
        """Generate common word with numbers"""
        word = random.choice(self.common_words)
        
        # Add year or numbers
        additions = [
            str(random.randint(1990, 2025)),
            str(random.randint(1, 999)),
            str(random.randint(10, 99)) + str(random.randint(10, 99))
        ]
        
        word += random.choice(additions)
        
        # Sometimes capitalize
        if random.random() < 0.6:
            word = word.capitalize()
        
        return word
    
    def _generate_name_with_year(self):
        """Generate name with birth year"""
        name = self.faker.first_name().lower()
        year = random.randint(1960, 2010)
        
        patterns = [f"{name}{year}", f"{name.capitalize()}{year}"]
        return random.choice(patterns)
    
    def _generate_simple_substitution(self):
        """Generate password with simple character substitutions"""
        word = random.choice(self.common_words)
        
        # Apply common substitutions
        substitutions = {'a': '@', 'o': '0', 's': '$', 'e': '3', 'i': '1'}
        
        for char, sub in substitutions.items():
            if random.random() < 0.7 and char in word:
                word = word.replace(char, sub)
        
        # Add numbers
        if random.random() < 0.8:
            word += str(random.randint(1, 999))
        
        return word
    
    def _generate_two_words_combined(self):
        """Combine two common words"""
        word1 = random.choice(self.common_words)
        word2 = random.choice(self.common_words)
        
        # Combine with different patterns
        patterns = [
            f"{word1}{word2}",
            f"{word1.capitalize()}{word2}",
            f"{word1}{word2.capitalize()}",
            f"{word1}_{word2}",
            f"{word1}-{word2}"
        ]
        
        result = random.choice(patterns)
        
        # Sometimes add numbers
        if random.random() < 0.6:
            result += str(random.randint(1, 99))
        
        return result
    
    def _generate_predictable_pattern(self):
        """Generate predictable patterns"""
        patterns = [
            lambda: "Password" + str(random.randint(1, 999)) + "!",
            lambda: "Welcome" + str(random.randint(1, 999)),
            lambda: "Admin" + str(random.randint(1, 999)),
            lambda: self.faker.first_name() + str(random.randint(1950, 2025)),
            lambda: random.choice(self.common_words).capitalize() + str(random.randint(10, 99)) + "!"
        ]
        
        pattern_func = random.choice(patterns)
        return pattern_func()
    
    def _generate_random_complex(self):
        """Generate truly random complex password"""
        length = random.randint(12, 20)
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choices(chars, k=length))
    
    def _generate_passphrase(self):
        """Generate passphrase-style strong password"""
        words = [self.faker.word() for _ in range(random.randint(3, 5))]
        separators = ['', '-', '_', '.']
        separator = random.choice(separators)
        
        passphrase = separator.join(words)
        
        # Sometimes add numbers and symbols
        if random.random() < 0.7:
            passphrase += str(random.randint(1, 999))
        if random.random() < 0.5:
            passphrase += random.choice('!@#$%')
        
        return passphrase
    
    def _generate_mixed_case_symbols(self):
        """Generate password with good mix of character types"""
        length = random.randint(10, 16)
        
        # Ensure we have each character type
        password = []
        password.extend(random.choices(string.ascii_lowercase, k=length//4))
        password.extend(random.choices(string.ascii_uppercase, k=length//4))
        password.extend(random.choices(string.digits, k=length//4))
        password.extend(random.choices(string.punctuation, k=length//4))
        
        # Fill remaining with random characters
        remaining = length - len(password)
        all_chars = string.ascii_letters + string.digits + string.punctuation
        password.extend(random.choices(all_chars, k=remaining))
        
        # Shuffle to avoid predictable patterns
        random.shuffle(password)
        return ''.join(password)
    
    def _generate_long_random(self):
        """Generate long random password"""
        length = random.randint(16, 25)
        chars = string.ascii_letters + string.digits + string.punctuation
        return ''.join(random.choices(chars, k=length))
    
    def _generate_pronounceable_complex(self):
        """Generate pronounceable but complex password"""
        consonants = 'bcdfghjklmnpqrstvwxyz'
        vowels = 'aeiou'
        
        password = ''
        for i in range(random.randint(3, 5)):
            # Add consonant-vowel pairs
            password += random.choice(consonants)
            password += random.choice(vowels)
        
        # Add complexity
        password += str(random.randint(10, 999))
        password += random.choice('!@#$%^&*')
        
        # Random capitalization
        password = ''.join(c.upper() if random.random() < 0.3 else c for c in password)
        
        return password
    
    def calculate_crack_time(self, password):
        """Estimate crack time based on password characteristics"""
        # Character set size
        charset_size = 0
        if any(c.islower() for c in password):
            charset_size += 26
        if any(c.isupper() for c in password):
            charset_size += 26
        if any(c.isdigit() for c in password):
            charset_size += 10
        if any(c in string.punctuation for c in password):
            charset_size += 32
        
        # Total combinations
        total_combinations = charset_size ** len(password)
        
        # Assume 1 billion guesses per second
        guesses_per_second = 1e9
        
        # Average time to crack (half the search space)
        crack_time_seconds = total_combinations / (2 * guesses_per_second)
        
        # Apply penalties for common patterns
        penalty_factor = 1.0
        
        # Check for weak patterns
        password_lower = password.lower()
        
        # Dictionary word penalty
        for word in self.common_words:
            if word in password_lower:
                penalty_factor *= 0.01
                break
        
        # Keyboard pattern penalty
        for pattern in self.keyboard_patterns:
            if pattern in password_lower:
                penalty_factor *= 0.001
                break
        
        # Repeated characters penalty
        if len(set(password)) < len(password) * 0.5:
            penalty_factor *= 0.1
        
        # Common substitutions don't help much
        substitution_chars = '@0$31179'
        if any(c in substitution_chars for c in password):
            penalty_factor *= 0.5
        
        # Year penalty
        for year in range(1950, 2030):
            if str(year) in password:
                penalty_factor *= 0.1
                break
        
        return max(0.001, crack_time_seconds * penalty_factor)
    
    def classify_strength(self, crack_time_seconds):
        """Classify password strength based on crack time"""
        if crack_time_seconds < 60:  # Less than 1 minute
            return "Weak"
        elif crack_time_seconds < 86400:  # Less than 1 day
            return "Medium"
        else:  # More than 1 day
            return "Strong"
    
    def generate_dataset(self, n_samples=10000, distribution=None):
        """Generate a balanced dataset of passwords"""
        if distribution is None:
            distribution = {"Weak": 0.4, "Medium": 0.35, "Strong": 0.25}
        
        passwords = []
        crack_times = []
        strengths = []
        
        # Generate passwords for each strength category
        for strength, proportion in distribution.items():
            n_category = int(n_samples * proportion)
            
            for _ in range(n_category):
                if strength == "Weak":
                    password = self.generate_weak_password()
                elif strength == "Medium":
                    password = self.generate_medium_password()
                else:  # Strong
                    password = self.generate_strong_password()
                
                crack_time = self.calculate_crack_time(password)
                predicted_strength = self.classify_strength(crack_time)
                
                passwords.append(password)
                crack_times.append(crack_time)
                strengths.append(predicted_strength)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Password': passwords,
            'Crack_Time_Sec': crack_times,
            'Strength_Label': strengths
        })
        
        # Shuffle the dataset
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df


def create_sample_dataset():
    """Create a sample dataset for testing"""
    generator = PasswordDataGenerator()
    df = generator.generate_dataset(n_samples=5000)
    
    # Add some real weak passwords for variety
    real_weak_passwords = [
        "123456", "password", "123456789", "12345678", "12345",
        "qwerty", "abc123", "password123", "admin", "letmein",
        "welcome", "monkey", "1234567890", "111111", "password1"
    ]
    
    real_weak_data = []
    for pwd in real_weak_passwords:
        crack_time = generator.calculate_crack_time(pwd)
        strength = generator.classify_strength(crack_time)
        real_weak_data.append({
            'Password': pwd,
            'Crack_Time_Sec': crack_time,
            'Strength_Label': strength
        })
    
    real_weak_df = pd.DataFrame(real_weak_data)
    df = pd.concat([df, real_weak_df], ignore_index=True)
    
    # Shuffle again
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    dataset = create_sample_dataset()
    print(f"Generated dataset with {len(dataset)} passwords")
    print(f"Strength distribution:")
    print(dataset['Strength_Label'].value_counts())
    
    # Save dataset
    dataset.to_csv("../data/password_dataset.csv", index=False)
    print("Dataset saved to ../data/password_dataset.csv")
