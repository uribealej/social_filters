"""Capture plotted values and structure without retaining large image fixtures."""

import hashlib
import json
import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


def compact_figure_contract(snapshot):
    """Keep a readable topology beside a digest covering every captured property."""
    fields = dict(snapshot["dict"])
    axes = []
    for encoded_axis in fields["axes"]["list"]:
        axis = dict(encoded_axis["dict"])
        axes.append({key: axis[key] for key in ["title", "xlabel", "ylabel", "xscale", "yscale"]})
        axes[-1].update({key: len(axis[key]["list"]) for key in ["images", "lines", "collections", "patches"]})
    return dict(sha256=hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest(), axes=axes)


def array_contract(value):
    """Encode exact numeric values and masks, including shape and dtype."""
    array = np.ma.asarray(value)
    data = np.ascontiguousarray(array.data)
    if data.dtype.kind in "OUS":
        return dict(shape=list(data.shape), values=data.tolist())
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                mask=hashlib.sha256(np.ma.getmaskarray(array).tobytes()).hexdigest())


def value_contract(value):
    """Preserve return containers and artist types as well as numerical results."""
    if isinstance(value, Figure):
        return {"artist": "Figure", "axes": len(value.axes)}
    if isinstance(value, Axes):
        return {"artist": "Axes", "index": value.figure.axes.index(value)}
    if isinstance(value, Artist):
        return {"artist": type(value).__name__}
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return {"shape": list(value.shape), "items": [value_contract(v) for v in value.flat]}
        return array_contract(value)
    if isinstance(value, np.generic):
        return value_contract(value.item())
    if isinstance(value, dict):
        return {"dict": [[str(k), value_contract(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {type(value).__name__: [value_contract(v) for v in value]}
    if isinstance(value, Path):
        return {"filename": value.name}
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def text_contract(text):
    """Record exact text and its explicit formatting."""
    return [text.get_text(), list(text.get_position()), text.get_fontsize(),
            text.get_rotation(), text.get_ha(), text.get_va(), text.get_visible()]


def figure_contract(figure):
    """Snapshot image, line, collection, patch, legend, axes and layout contracts."""
    result = dict(size=figure.get_size_inches().tolist(), dpi=figure.dpi,
                  texts=[text_contract(t) for t in figure.texts], axes=[],
                  legends=[[t.get_text() for t in l.get_texts()] for l in figure.legends])
    for ax in figure.axes:
        legend = ax.get_legend()
        item = dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                    xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                    xscale=ax.get_xscale(), yscale=ax.get_yscale(), aspect=ax.get_aspect(),
                    xticks=list(ax.get_xticks()), yticks=list(ax.get_yticks()),
                    xticklabels=[text_contract(t) for t in ax.get_xticklabels()],
                    yticklabels=[text_contract(t) for t in ax.get_yticklabels()],
                    texts=[text_contract(t) for t in ax.texts],
                    legend=None if legend is None else [t.get_text() for t in legend.get_texts()],
                    images=[], lines=[], collections=[], patches=[])
        for im in ax.images:
            item["images"].append(dict(data=array_contract(im.get_array()),
                extent=list(im.get_extent()), clim=list(im.get_clim()), cmap=im.get_cmap().name,
                origin=im.origin, interpolation=im.get_interpolation()))
        for line in ax.lines:
            item["lines"].append(dict(x=array_contract(line.get_xdata()), y=array_contract(line.get_ydata()),
                color=value_contract(line.get_color()), style=line.get_linestyle(),
                width=line.get_linewidth(), marker=line.get_marker(), label=line.get_label(),
                alpha=line.get_alpha()))
        for collection in ax.collections:
            item["collections"].append(dict(kind=type(collection).__name__,
                offsets=array_contract(collection.get_offsets()),
                paths=[array_contract(p.vertices) for p in collection.get_paths()],
                array=None if collection.get_array() is None else array_contract(collection.get_array()),
                face=array_contract(collection.get_facecolors()), edge=array_contract(collection.get_edgecolors()),
                clim=list(collection.get_clim()), cmap=collection.get_cmap().name))
        for patch in ax.patches:
            item["patches"].append(dict(kind=type(patch).__name__,
                vertices=array_contract(patch.get_path().vertices),
                bounds=patch.get_extents().bounds, face=patch.get_facecolor(), edge=patch.get_edgecolor()))
        result["axes"].append(item)
    return value_contract(result)
