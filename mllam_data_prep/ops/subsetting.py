from typing import Any, Dict, List, Optional, Union

import xarray as xr


def extract_variable(
    ds: xr.Dataset,
    var_name: Union[str, Dict[str, Any], List[str]],
    coords_to_sample: Optional[Dict[str, Any]] = None,
) -> xr.DataArray:
    """
    Extract specified variable from the provided input dataset. If
    coordinates for subsetting are defined, then subset the variable along
    them and check coordinate units.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    var_name : Union[Dict, List]
        Either a list or dictionary with variables to extract.
        If a dictionary the keys are the variable name and the values are
        entries for each coordinate and coordinate values to extract
    coords_to_sample: Dict
        Optional argument for subsetting/sampling along the specified
        coordinates

    Returns
    ----------
    da: xr.DataArray
        Extracted variable (subsetted along the specified coordinates)
    """
    if coords_to_sample is None:
        coords_to_sample = {}

    try:
        da = ds[var_name]
    except KeyError as ex:
        raise KeyError(
            f"Could not find the variable `{var_name}` in the dataset. "
            f"The available variables are {list(ds.data_vars)}"
        ) from ex

    for coord, sampling in coords_to_sample.items():
        coord_values = sampling.values
        try:
            da = da.sel(**{coord: coord_values})
        except KeyError as ex:
            raise KeyError(
                f"Could not find the all coordinate values `{coord_values}` in "
                f"coordinate `{coord}` in the dataset"
            ) from ex
        expected_units = sampling.units
        coord_units = da[coord].attrs.get("units", None)
        if coord_units is not None and coord_units != expected_units:
            raise ValueError(
                f"Expected units {expected_units} for coordinate {coord}"
                f" in variable {var_name} but got {coord_units}"
            )

    return da
