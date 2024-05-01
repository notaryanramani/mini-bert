import pandas as pd

class FineTuneDataProcessing:

    def __init__(self,):
        self.out_df = pd.DataFrame(
            columns=['x', 'y']
        )

    def preprocess(self, df: pd.DataFrame):
        '''
        returns a DataFrame with columns x and y

        Parameters:
            -df - pd.DataFrame : DataFrame with columns x1, x2, y

        Returns:   
            -self.out_df - pd.DataFrame : processed DataFrame with columns x, y
        '''


        assert not df.isnull().values.any(), 'Input DataFrame should not have any missing values'
        assert all([col in df.columns for col in ['x1', 'x2', 'y']]), 'Input DataFrame should have columns x1, x2, y'
        self.out_df['x'] = '<|cls|>' + df['x1'] + '<|sep|>' + df['x2'] + '<|endoftext|>'
        self.out_df['y'] = df['y']

        return self.out_df