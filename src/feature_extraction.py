"""
Password Feature Extraction Module
Extracts various features from passwords for ML training
"""

import re
import math
import string
from collections import Counter
import numpy as np


class PasswordFeatureExtractor:
    """Extract features from passwords for ML model training"""
    
    def __init__(self):
        self.keyboard_patterns = [
            'qwerty', 'asdf', 'zxcv', '123456', 'abcdef',
            'qwertyuiop', 'asdfghjkl', 'zxcvbnm',
            '1234567890', '0987654321'
        ]
        
        self.common_substitutions = {
            'a': '@', 'o': '0', 's': '$', 'e': '3', 
            'i': '1', 'l': '1', 't': '7', 'g': '9'
        }
        
    def extract_features(self, password):
        """Extract all features from a password"""
        features = {}
        
        # Basic length features
        features['length'] = len(password)
        features['length_squared'] = len(password) ** 2
        
        # Character composition
        features['uppercase_count'] = sum(1 for c in password if c.isupper())
        features['lowercase_count'] = sum(1 for c in password if c.islower())
        features['digit_count'] = sum(1 for c in password if c.isdigit())
        features['symbol_count'] = sum(1 for c in password if c in string.punctuation)
        
        # Ratios
        if len(password) > 0:
            features['uppercase_ratio'] = features['uppercase_count'] / len(password)
            features['lowercase_ratio'] = features['lowercase_count'] / len(password)
            features['digit_ratio'] = features['digit_count'] / len(password)
            features['symbol_ratio'] = features['symbol_count'] / len(password)
        else:
            features['uppercase_ratio'] = 0
            features['lowercase_ratio'] = 0
            features['digit_ratio'] = 0
            features['symbol_ratio'] = 0
        
        # Character set diversity
        features['char_set_size'] = len(set(password))
        features['unique_char_ratio'] = features['char_set_size'] / len(password) if len(password) > 0 else 0
        
        # Entropy calculation
        features['entropy'] = self._calculate_entropy(password)
        
        # Pattern detection
        features['has_keyboard_pattern'] = self._has_keyboard_pattern(password)
        features['keyboard_pattern_length'] = self._get_keyboard_pattern_length(password)
        
        # Sequential patterns
        features['has_sequential_chars'] = self._has_sequential_chars(password)
        features['sequential_char_count'] = self._count_sequential_chars(password)
        
        # Repetition patterns
        features['has_repeated_chars'] = self._has_repeated_chars(password)
        features['repeated_char_count'] = self._count_repeated_chars(password)
        features['max_repeated_length'] = self._max_repeated_length(password)
        
        # Common substitutions
        features['substitution_count'] = self._count_substitutions(password)
        features['has_common_substitutions'] = features['substitution_count'] > 0
        
        # Dictionary words
        features['contains_dictionary_word'] = self._contains_dictionary_word(password)
        features['dictionary_word_count'] = self._count_dictionary_words(password)
        
        # Date patterns
        features['contains_date'] = self._contains_date_pattern(password)
        features['contains_year'] = self._contains_year(password)
        
        # Position-based features
        features['starts_with_uppercase'] = password[0].isupper() if len(password) > 0 else False
        features['ends_with_digit'] = password[-1].isdigit() if len(password) > 0 else False
        features['ends_with_symbol'] = password[-1] in string.punctuation if len(password) > 0 else False
        
        # Complexity score
        features['complexity_score'] = self._calculate_complexity_score(password)
        
        return features
    
    def _calculate_entropy(self, password):
        """Calculate Shannon entropy of password"""
        if not password:
            return 0
        
        # Count character frequencies
        char_counts = Counter(password)
        length = len(password)
        
        # Calculate entropy
        entropy = 0
        for count in char_counts.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _has_keyboard_pattern(self, password):
        """Check if password contains keyboard patterns"""
        password_lower = password.lower()
        for pattern in self.keyboard_patterns:
            if pattern in password_lower or pattern[::-1] in password_lower:
                return True
        return False
    
    def _get_keyboard_pattern_length(self, password):
        """Get length of longest keyboard pattern"""
        password_lower = password.lower()
        max_length = 0
        
        for pattern in self.keyboard_patterns:
            if pattern in password_lower:
                max_length = max(max_length, len(pattern))
            if pattern[::-1] in password_lower:
                max_length = max(max_length, len(pattern))
        
        return max_length
    
    def _has_sequential_chars(self, password):
        """Check for sequential characters (abc, 123, etc.)"""
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i+1]) + 1):
                return True
        return False
    
    def _count_sequential_chars(self, password):
        """Count sequential character sequences"""
        count = 0
        i = 0
        while i < len(password) - 2:
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i+1]) + 1):
                seq_length = 3
                j = i + 3
                while j < len(password) and ord(password[j]) == ord(password[j-1]) + 1:
                    seq_length += 1
                    j += 1
                count += seq_length
                i = j
            else:
                i += 1
        return count
    
    def _has_repeated_chars(self, password):
        """Check for repeated characters"""
        for i in range(len(password) - 1):
            if password[i] == password[i+1]:
                return True
        return False
    
    def _count_repeated_chars(self, password):
        """Count repeated character sequences"""
        count = 0
        i = 0
        while i < len(password) - 1:
            if password[i] == password[i+1]:
                repeat_length = 2
                j = i + 2
                while j < len(password) and password[j] == password[i]:
                    repeat_length += 1
                    j += 1
                count += repeat_length
                i = j
            else:
                i += 1
        return count
    
    def _max_repeated_length(self, password):
        """Get maximum length of repeated character sequence"""
        max_length = 0
        i = 0
        while i < len(password) - 1:
            if password[i] == password[i+1]:
                repeat_length = 2
                j = i + 2
                while j < len(password) and password[j] == password[i]:
                    repeat_length += 1
                    j += 1
                max_length = max(max_length, repeat_length)
                i = j
            else:
                i += 1
        return max_length
    
    def _count_substitutions(self, password):
        """Count common character substitutions"""
        count = 0
        for char, sub in self.common_substitutions.items():
            count += password.count(sub)
        return count
    
    def _contains_dictionary_word(self, password):
        """Check if password contains common dictionary words"""
        common_words = [
            'password', 'admin', 'user', 'login', 'welcome',
            'hello', 'world', 'test', 'qwerty', 'abc',
            'love', 'god', 'sex', 'secret', 'money'
        ]
        
        password_lower = password.lower()
        for word in common_words:
            if word in password_lower:
                return True
        return False
    
    def _count_dictionary_words(self, password):
        """Count dictionary words in password"""
        common_words = [
            'password', 'admin', 'user', 'login', 'welcome',
            'hello', 'world', 'test', 'qwerty', 'abc',
            'love', 'god', 'sex', 'secret', 'money'
        ]
        
        count = 0
        password_lower = password.lower()
        for word in common_words:
            if word in password_lower:
                count += 1
        return count
    
    def _contains_date_pattern(self, password):
        """Check for date patterns"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\d{2,4}\d{2}\d{2}'         # YYYYMMDD
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, password):
                return True
        return False
    
    def _contains_year(self, password):
        """Check if password contains a year (1900-2030)"""
        for year in range(1900, 2031):
            if str(year) in password:
                return True
        return False
    
    def _calculate_complexity_score(self, password):
        """Calculate a complexity score based on various factors"""
        score = 0
        
        # Length bonus
        score += len(password) * 2
        
        # Character diversity bonus
        if any(c.isupper() for c in password):
            score += 5
        if any(c.islower() for c in password):
            score += 5
        if any(c.isdigit() for c in password):
            score += 5
        if any(c in string.punctuation for c in password):
            score += 10
        
        # Penalty for patterns
        if self._has_keyboard_pattern(password):
            score -= 20
        if self._has_sequential_chars(password):
            score -= 15
        if self._has_repeated_chars(password):
            score -= 10
        if self._contains_dictionary_word(password):
            score -= 25
        
        return max(0, score)  # Ensure non-negative score


def extract_features_batch(passwords):
    """Extract features for a batch of passwords"""
    extractor = PasswordFeatureExtractor()
    features_list = []
    
    for password in passwords:
        features = extractor.extract_features(password)
        features_list.append(features)
    
    return features_list
