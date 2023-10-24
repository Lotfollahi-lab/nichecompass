import json


class Config:
    """Object that stores configuration for a NicheCompass run"""

    def __init__(self, filepath):
        with open(filepath) as f:
            data = json.load(f)
        self.options = data
