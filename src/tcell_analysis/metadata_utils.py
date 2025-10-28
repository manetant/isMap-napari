
def get_pixel_sizes_and_units(img):
    """
    Extract physical pixel size and unit from an image’s OME metadata.

    Returns a standardized dict:
        {
            "PhysicalSizeX": float or None,
            "PhysicalSizeY": float or None,
            "PhysicalSizeUnit": str or None,
        }

    If unavailable, returns values as None.
    Works with BioImage (OME) objects.
    """
    try:
        ome = getattr(img, "metadata", None)
        if not ome or not getattr(ome, "images", None):
            return {"PhysicalSizeX": None, "PhysicalSizeY": None, "PhysicalSizeUnit": None}

        pixels = getattr(ome.images[0], "pixels", None)
        if not pixels:
            return {"PhysicalSizeX": None, "PhysicalSizeY": None, "PhysicalSizeUnit": None}

        x = getattr(pixels, "physical_size_x", None)
        y = getattr(pixels, "physical_size_y", None)
        ux = getattr(pixels, "physical_size_x_unit", None)
        uy = getattr(pixels, "physical_size_y_unit", None)

        unit = ux or uy or None

        px_meta_clean = {}
        if x is not None:
            px_meta_clean["PhysicalSizeX"] = float(x)
        else:
            px_meta_clean["PhysicalSizeX"] = None

        if y is not None:
            px_meta_clean["PhysicalSizeY"] = float(y)
        else:
            px_meta_clean["PhysicalSizeY"] = None

        if unit is not None:
            # some BioFormats units are enum-like, so use .value if present
            px_meta_clean["PhysicalSizeUnit"] = getattr(unit, "value", str(unit))
        else:
            px_meta_clean["PhysicalSizeUnit"] = None

        return px_meta_clean

    except Exception:
        return {"PhysicalSizeX": None, "PhysicalSizeY": None, "PhysicalSizeUnit": None}


def avg_px_um(px_size: dict | None) -> float | None:
    """Return average pixel size in µm/px, or None if unavailable."""
    if not isinstance(px_size, dict):
        return None
    unit = (px_size.get("PhysicalSizeUnit") or "µm").lower()
    sx = px_size.get("PhysicalSizeX")
    sy = px_size.get("PhysicalSizeY")
    if sx is None or sy is None:
        return None

    def _to_um(v):
        if v is None:
            return None
        v = float(v)
        if unit in ("µm", "um", "micrometer", "micrometre"):
            return v
        if unit in ("nm", "nanometer", "nanometre"):
            return v / 1000.0
        if unit in ("mm", "millimeter", "millimetre"):
            return v * 1000.0
        return v

    sx = _to_um(sx); sy = _to_um(sy)
    if sx is None or sy is None:
        return None
    return (sx + sy) / 2.0