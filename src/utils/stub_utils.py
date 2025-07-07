import os
import pickle
from typing import Any

def save_stub(stub_path: str, obj: Any) -> None:
    """
    Saves a Python object to a file using pickle.

    Args:
        stub_path (str): File path where the object should be saved.
        obj (Any): Python object to serialize and store.

    Returns:
        None    
    """
    # Ensure parent directory exists
    output_dir = os.path.dirname(stub_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the object if the path is valid
    if (stub_path is not None):
        with open(stub_path, "wb") as f:
            pickle.dump(obj, f)

def read_stub(read_from_stub: bool, stub_path: str) -> Any:
    """
    Reads a Python object from a file using pickle.

    Args:
        read_from_stub (bool): If True, attempt to read the object from the stub file.
        stub_path (str): File path where the pickled object is stored.

    Returns:
        Any: The deserialized Python object if read is successful; otherwise, None.
    """
    if (read_from_stub and stub_path is not None and os.path.exists(stub_path)):
        with open(stub_path, "rb") as f:
            obj = pickle.load(f)
            return obj

    return None  