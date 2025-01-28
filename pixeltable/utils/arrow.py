import datetime
from typing import Any, Iterator, Optional, Union

import numpy as np
import pyarrow as pa

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.io.globals import _find_or_create_table, _normalize_pxt_col_name, _normalize_schema_names

PA_TO_PXT_TYPES: dict[pa.DataType, ts.ColumnType] = {
    pa.string(): ts.StringType(nullable=True),
    pa.large_string(): ts.StringType(nullable=True),
    pa.date32(): ts.TimestampType(nullable=True),
    pa.timestamp('us', tz=datetime.timezone.utc): ts.TimestampType(nullable=True),
    pa.bool_(): ts.BoolType(nullable=True),
    pa.uint8(): ts.IntType(nullable=True),
    pa.int8(): ts.IntType(nullable=True),
    pa.uint32(): ts.IntType(nullable=True),
    pa.uint64(): ts.IntType(nullable=True),
    pa.int32(): ts.IntType(nullable=True),
    pa.int64(): ts.IntType(nullable=True),
    pa.float32(): ts.FloatType(nullable=True),
    pa.float64(): ts.FloatType(nullable=True),
}

PXT_TO_PA_TYPES: dict[type[ts.ColumnType], pa.DataType] = {
    ts.StringType: pa.string(),
    ts.TimestampType: pa.timestamp('us', tz=datetime.timezone.utc),  # postgres timestamp is microseconds
    ts.BoolType: pa.bool_(),
    ts.IntType: pa.int64(),
    ts.FloatType: pa.float32(),
    ts.JsonType: pa.string(),  # TODO(orm) pa.struct() is possible
    ts.ImageType: pa.binary(),  # inline image
    ts.AudioType: pa.string(),  # path
    ts.VideoType: pa.string(),  # path
    ts.DocumentType: pa.string(),  # path
}


def to_pixeltable_type(arrow_type: pa.DataType) -> Optional[ts.ColumnType]:
    """Convert a pyarrow DataType to a pixeltable ColumnType if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if isinstance(arrow_type, pa.TimestampType):
        return ts.TimestampType(nullable=True)
    elif arrow_type in PA_TO_PXT_TYPES:
        return PA_TO_PXT_TYPES[arrow_type]
    elif isinstance(arrow_type, pa.FixedShapeTensorType):
        dtype = to_pixeltable_type(arrow_type.value_type)
        if dtype is None:
            return None
        return ts.ArrayType(shape=arrow_type.shape, dtype=dtype)
    else:
        return None


def to_arrow_type(pixeltable_type: ts.ColumnType) -> Optional[pa.DataType]:
    """Convert a pixeltable DataType to a pyarrow datatype if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if pixeltable_type.__class__ in PXT_TO_PA_TYPES:
        return PXT_TO_PA_TYPES[pixeltable_type.__class__]
    elif isinstance(pixeltable_type, ts.ArrayType):
        return pa.fixed_shape_tensor(pa.from_numpy_dtype(pixeltable_type.numpy_dtype()), pixeltable_type.shape)
    else:
        return None


def ar_infer_schema(
    arrow_schema: pa.Schema, schema_overrides: Optional[dict[str, ts.ColumnType]] = None
) -> dict[str, pxt.ColumnType]:
    """Convert a pyarrow Schema to a schema using pyarrow names and pixeltable types."""
    if schema_overrides is None:
        schema_overrides = {}
    ar_schema = {
        field.name: to_pixeltable_type(field.type)
        if field.name not in schema_overrides
        else schema_overrides[field.name]
        for field in arrow_schema
    }
    return ar_schema


def to_arrow_schema(pixeltable_schema: dict[str, Any]) -> pa.Schema:
    return pa.schema((name, to_arrow_type(typ)) for name, typ in pixeltable_schema.items())  # type: ignore[misc]


def to_pydict(batch: Union[pa.Table, pa.RecordBatch]) -> dict[str, Union[list, np.ndarray]]:
    """Convert a RecordBatch to a dictionary of lists, unlike pa.lib.RecordBatch.to_pydict,
    this function will not convert numpy arrays to lists, and will preserve the original numpy dtype.
    """
    out: dict[str, Union[list, np.ndarray]] = {}
    for k, name in enumerate(batch.schema.names):
        col = batch.column(k)
        if isinstance(col.type, pa.FixedShapeTensorType):
            # treat array columns as numpy arrays to easily preserve numpy type
            out[name] = col.to_numpy(zero_copy_only=False)  # type: ignore[call-arg]
        else:
            # for the rest, use pydict to preserve python types
            out[name] = col.to_pylist()

    return out


def iter_tuples(batch: Union[pa.Table, pa.RecordBatch]) -> Iterator[dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        yield {col_name: values[i] for col_name, values in pydict.items()}


def iter_tuples2(
    batch: Union[pa.Table, pa.RecordBatch], mapping: dict[str, str], schema: dict[str, pxt.ColumnType]
) -> Iterator[dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        # Convert a row to insertable format
        yield {
            mapping[col_name]: schema[mapping[col_name]].create_literal(values[i])
            for col_name, values in pydict.items()
        }
