import pandas as pd
import numpy as np
import torch

class BasicDatasetForBook:
    def __init__(self, data) -> None:
        self._data = data
    
    @classmethod
    def load(cls, file_path):
        data = pd.read_csv(file_path)
        draft_data = []
        genre_to_id = {'Cookbooks, Food, Wine': 0, 'Self Help': 1, 'Science Fiction, Fantasy': 2, 'Mystery, Thriller, Suspense': 3, 'Health, Fitness, Dieting': 4, 'Humor, Entertainment': 5, 'Religion, Spirituality': 6, 'Computers, Technology': 7, 'Reference': 8, 'Medical Books': 9, 'Test Preparation': 10, 'Engineering, Transportation': 11, 'Childrens Books': 12, 'Arts, Photography': 13, 'Science, Math': 14, 'Parenting, Relationships': 15, 'Literature, Fiction': 16, 'Calendars': 17, 'Sports, Outdoors': 18, 'Travel': 19, 'Crafts, Hobbies, Home': 20, 'Law': 21, 'Biographies, Memoirs': 22, 'Education, Teaching': 23}
        
        # Filename, Title, label
        
        for title, genre in zip(data['Title'], data['label']): 
            genre_id = genre_to_id[genre]
            draft_data.append((title, genre_id))
            
        return cls(draft_data)

    def __getitem__(self, index) -> dict:
        title, genre_id = self._data[index]
        
        return {
            'title':title,
            'label':genre_id,
        }
        
    def __len__(self):
        return len(self._data)