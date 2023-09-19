import pathlib
import pandas as pd

# folder of dataset, dataset_dir should contain s1/, s2/, ...
dataset_dir = pathlib.Path('/Users/marwei/Datasets/encodedGdumb/core50_128x128/')
class_names= {
    0: 'plug adapters',
    1: 'mobile phones',
    2: 'scissors',
    3: 'light bulbs',
    4: 'cans',
    5: 'glasses',
    6: 'balls',
    7: 'markers',
    8: 'cups',
    9: 'remote controls'
}
records = []

# create a df of all records in the dataset
session_paths = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
for session_path in session_paths:
    this_session = int(session_path.stem[1:])
    object_paths = sorted([d for d in session_path.iterdir() if d.is_dir()])
    for object_path in object_paths:
        this_object = int(object_path.stem[1:])
        frame_paths = sorted(object_path.glob('*.png'))
        for frame_path in frame_paths:
            records.append(
                {
                    'session': this_session,
                    'object': this_object,
                    'frame': int(frame_path.stem.split('_')[3]),
                    'file': frame_path.relative_to(dataset_dir)
                }
            )
df = pd.DataFrame.from_records(records)
df['class'] = (df['object']-1) // 5
df['split'] = df['session'].apply(lambda x: 'test' if x in [3, 7, 10] else 'train')
df['class_name'] = df['class'].map(class_names)

# save the dfs
df.set_index('file', inplace=True)
trainset = df.loc[df['split'] == 'train'].copy()
testset = df.loc[df['split'] == 'test'].copy()
trainset.reset_index(inplace=True)
testset.reset_index(inplace=True)
trainset.to_csv(dataset_dir / 'train.csv')
testset.to_csv(dataset_dir / 'test.csv')
