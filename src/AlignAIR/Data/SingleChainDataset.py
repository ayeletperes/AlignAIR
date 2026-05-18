# In your SingleChainDataset.py file

import numpy as np
import pandas as pd

from .columnSet import ColumnSet
from ..Data.datasetBase import DatasetBase
from GenAIRR.dataconfig import DataConfig


class SingleChainDataset(DatasetBase):
    """
    A dataset class for a single chain (e.g., heavy or light).
    """

    def __init__(self, data_path, dataconfig: DataConfig, batch_size=64, max_sequence_length=576, use_streaming=False,
                 nrows=None, seperator=',', evaluation_only=False,
                 use_aa_stream=False, max_aa_sequence_length=None):
        if evaluation_only:
            # AA mode reads from the 'sequence_aa' column; nt mode from 'sequence'.
            self.required_data_columns = ['sequence_aa'] if use_aa_stream else ['sequence']
        else:
            self.required_data_columns = ColumnSet(has_d=dataconfig.metadata.has_d)

        super().__init__(data_path, dataconfig, batch_size, max_sequence_length, use_streaming, nrows, seperator,
                         required_data_columns=self.required_data_columns,
                         use_aa_stream=use_aa_stream,
                         max_aa_sequence_length=max_aa_sequence_length)