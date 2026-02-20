"""
Unit tests for ops.chunking module.
"""
import numpy as np
import pytest
import xarray as xr

from mllam_data_prep.ops.chunking import check_chunk_size, chunk_dataset


@pytest.fixture
def small_dataset():
    """Create a small test dataset."""
    return xr.Dataset(
        {
            "var1": (["x", "y"], np.random.random((10, 10))),
            "var2": (["x", "y"], np.random.random((10, 10))),
        },
        coords={"x": range(10), "y": range(10)},
    )


@pytest.fixture
def large_dataset():
    """Create a dataset that will exceed chunk size warning."""
    # Create dataset with large chunks that exceed 1GB warning
    # Using float64 (8 bytes), need > 1GB / 8 = 134217728 elements
    # For simplicity, create a smaller but still large dataset
    size = 5000
    return xr.Dataset(
        {
            "large_var": (["x", "y"], np.random.random((size, size))),
        },
        coords={"x": range(size), "y": range(size)},
    )


def test_check_chunk_size_small_chunks(small_dataset, caplog):
    """Test check_chunk_size with small chunks (should not warn)."""
    chunks = {"x": 5, "y": 5}
    check_chunk_size(small_dataset, chunks)
    # Should not log any warnings
    assert len(caplog.records) == 0


def test_check_chunk_size_large_chunks(large_dataset, caplog):
    """Test check_chunk_size with large chunks (should warn)."""
    # Use chunks that will create large memory usage
    chunks = {"x": 1000, "y": 1000}
    check_chunk_size(large_dataset, chunks)
    # Should log a warning
    assert len(caplog.records) > 0
    assert "exceeds" in caplog.records[0].message.lower()


def test_check_chunk_size_missing_dimension(small_dataset):
    """Test check_chunk_size when dimension doesn't exist in variable."""
    chunks = {"x": 5, "z": 10}  # z doesn't exist
    # Should not raise, just skip the missing dimension
    check_chunk_size(small_dataset, chunks)


def test_chunk_dataset_success(small_dataset):
    """Test chunk_dataset successfully chunks a dataset."""
    chunks = {"x": 5, "y": 5}
    chunked = chunk_dataset(small_dataset, chunks)
    assert isinstance(chunked, xr.Dataset)
    # Check that chunking was applied
    assert chunked["var1"].chunks is not None


def test_chunk_dataset_invalid_chunks(small_dataset):
    """Test chunk_dataset with invalid chunk specification."""
    chunks = {"x": -1}  # Invalid chunk size
    with pytest.raises(Exception, match="Error chunking dataset"):
        chunk_dataset(small_dataset, chunks)
