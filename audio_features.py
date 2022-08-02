import pandas
from pliers.extractors import ChromaSTFTExtractor
import s3fs

fs = s3fs.S3FileSystem(anon=True)
fs.get('openneuro.org/ds003097/stimuli/task-moviewatching_desc-koyaanisqatsi_movie.mp4',
       'ds003097/stimuli/task-moviewatching_desc-koyaanisqatsi_movie.mp4')

n_chroma, hop_length = 5, 97020
ext = ChromaSTFTExtractor(n_chroma=n_chroma, hop_length=hop_length)
result: pandas.DataFrame = ext.transform('ds003097/stimuli/task-moviewatching_desc-koyaanisqatsi_movie.mp4').to_df()
result.drop('object_id', axis=1, inplace=True)
result.to_csv(f'ds003097/stimuli/stft_{n_chroma}_{hop_length}.csv')
