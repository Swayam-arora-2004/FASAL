import os
import json

train_dir = 'train'

class_names = sorted(os.listdir(train_dir))

class_name_dict = {str(i): class_name for i, class_name in enumerate(class_names)}

with open('class_names.json', 'w') as f:
    json.dump(class_name_dict, f)

print("Class names saved to class_names.json")
