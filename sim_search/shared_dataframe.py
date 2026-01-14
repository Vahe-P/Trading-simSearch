import dataclasses
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.typing import DTypeLike
import pandas as pd


"""
The classes in this module create readonly DataFrames backed by shared memory. It uses two buffers, one for the values,
and one for the index. Each buffer must be homogeneously typed, so the values in the DataFrame must have the same type.
Also strings/objects are not supported. 

This is not useful on Linux, because Linux uses fork. Large numpy arrays are allocated in their own pages, which can be
shared with child processes. However on Mac the default behavior now is to use spawn, and Windows doesn't support fork.
For both of those OS, the data sent to child processes must be pickled, so this shared memory wrapper is an improvement.
"""


@dataclasses.dataclass
class SharedNumpyModel:
    """A model of a numpy array stored in shared memory."""
    name: str
    shape: tuple
    nbytes: int
    dtype: DTypeLike


class SharedNumpyArray:
    """
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    https://e-dorigatti.github.io/python/2020/06/19/multiprocessing-large-objects.html
    """

    def __init__(self, array):
        """Creates the shared memory and copies the array into it."""
        self._shared = SharedMemory(create=True, size=array.nbytes)
        self._array = np.ndarray(array.shape, dtype=array.dtype, buffer=self._shared.buf)
        self._array[:] = array[:]
        self._array.writable = False

    def get_model(self) -> SharedNumpyModel:
        """Returns a model of the array stored in shared memory."""
        return SharedNumpyModel(
            name=self._shared.name,
            shape=self._array.shape,
            nbytes=self._array.nbytes,
            dtype=self._array.dtype
        )

    def unlink(self):
        """Releases the memory. Call when finished using the data, or when the data was copied somewhere else."""
        del self._array
        self._shared.close()
        self._shared.unlink()


@dataclasses.dataclass
class SharedDataFrameModel:
    """A model of a pandas dataframe stored in shared memory."""
    values: SharedNumpyModel
    index: SharedNumpyModel
    index_name: str
    columns: list

    def to_frame(self):
        """Returns a view of the dataframe stored in shared memory."""
        values = np.ndarray(self.values.shape, dtype=self.values.dtype, buffer=SharedMemory(self.values.name).buf)
        index_array = np.ndarray(self.index.shape, dtype=self.index.dtype, buffer=SharedMemory(self.index.name).buf)
        index = pd.DatetimeIndex(index_array, name=self.index_name)
        values.writable = False
        index.writable = False
        return pd.DataFrame(values, index=index, columns=self.columns)


class SharedDataFrame:
    """Supports sharing a pandas dataframe among processes."""

    def __init__(self, df):
        if not df.dtypes.eq(df.dtypes.iloc[0]).all():
            raise Exception('SharedDataFrame should have a single dtype.')
        if df.dtypes[0] == 'O':
            raise Exception('SharedDataFrame should not contain object values.')
        if not isinstance(df.index, pd.DatetimeIndex):
            raise Exception('SharedDataFrame should have a DatetimeIndex.')
        self._df = df
        self._values = SharedNumpyArray(self._df.values)
        self._index = SharedNumpyArray(self._df.index.values)

    def get_model(self) -> SharedDataFrameModel:
        """Returns a dataframe model backed by shared memory. Sent to child processes in place of the dataframe."""
        return SharedDataFrameModel(
            values=self._values.get_model(),
            index=self._index.get_model(),
            index_name=self._df.index.name,
            columns=self._df.columns,
        )

    def unlink(self):
        self._values.unlink()
        self._index.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlink()
