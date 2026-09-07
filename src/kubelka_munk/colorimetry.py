"""Turning reflectance into colour: spectrum to XYZ to sRGB, and CIELAB differences.

This is the boundary layer of the library. Kubelka-Munk mixing happens entirely in
spectral space; colour spaces appear only when a result has to be shown to a human or
compared against a target. Every conversion here is standard colorimetry with no
Kubelka-Munk content at all.
"""

from __future__ import annotations

import numpy as np

from .cie_data import CIE_1931_2_DEGREE_OBSERVER, ILLUMINANT_D65
from .spectrum import BAND_COUNT, WAVELENGTHS_NM, Spectrum

# The observer as a (bands, 3) array of x-bar, y-bar, z-bar, and D65 as (bands,).
_OBSERVER = np.array(
    [CIE_1931_2_DEGREE_OBSERVER[int(nm)] for nm in WAVELENGTHS_NM], dtype=float
)
_D65_POWER = np.array([ILLUMINANT_D65[int(nm)] for nm in WAVELENGTHS_NM], dtype=float)

# Scaling chosen so a perfect reflecting diffuser (reflectance 1 everywhere) has Y = 1.
_LUMINANCE_NORMALISATION = 1.0 / float(_D65_POWER @ _OBSERVER[:, 1])

# Multiplying a reflectance vector by this (bands, 3) matrix integrates it against the
# illuminant and the observer in one step, giving XYZ directly.
REFLECTANCE_TO_XYZ = (
    _OBSERVER * _D65_POWER[:, np.newaxis] * _LUMINANCE_NORMALISATION
)

# IEC 61966-2-1 with the 2003 amendment's higher-precision coefficients.
# https://en.wikipedia.org/wiki/SRGB
XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2406255, -1.5372080, -0.4986286],
        [-0.9689307, 1.8757561, 0.0415175],
        [0.0557101, -0.2040211, 1.0569959],
    ]
)
LINEAR_SRGB_TO_XYZ = np.linalg.inv(XYZ_TO_LINEAR_SRGB)

# The colour of the illuminant itself -- a perfect white under D65.
D65_WHITE_POINT_XYZ = np.ones(BAND_COUNT) @ REFLECTANCE_TO_XYZ


def spectrum_to_xyz(spectrum: Spectrum) -> np.ndarray:
    """CIE XYZ tristimulus values of a reflecting surface seen under D65, with Y in [0, 1]."""
    return spectrum.reflectance @ REFLECTANCE_TO_XYZ


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    """Apply the sRGB transfer function, mapping linear light to display values in [0, 1]."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
    )


def decode_srgb(encoded: np.ndarray) -> np.ndarray:
    """Undo the sRGB transfer function, mapping display values back to linear light."""
    encoded = np.clip(encoded, 0.0, 1.0)
    return np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )


def xyz_to_srgb(xyz: np.ndarray) -> np.ndarray:
    """Display-ready sRGB in [0, 1]. Out-of-gamut colours are clipped, not gamut-mapped.

    Accepts one colour or any stack of them, the last axis being the three components,
    so a whole image can be converted at once.
    """
    return encode_srgb(np.asarray(xyz, dtype=float) @ XYZ_TO_LINEAR_SRGB.T)


def srgb_to_xyz(srgb: np.ndarray) -> np.ndarray:
    """CIE XYZ of a display colour given as sRGB in [0, 1], one colour or a stack."""
    return decode_srgb(np.asarray(srgb, dtype=float)) @ LINEAR_SRGB_TO_XYZ.T


def spectrum_to_srgb(spectrum: Spectrum) -> np.ndarray:
    """The colour a surface appears under D65, as sRGB in [0, 1]."""
    return xyz_to_srgb(spectrum_to_xyz(spectrum))


def srgb_to_hex(srgb: np.ndarray) -> str:
    """Format sRGB in [0, 1] as a ``#rrggbb`` string."""
    channels = np.clip(np.round(np.asarray(srgb, dtype=float) * 255), 0, 255)
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in channels))


def hex_to_srgb(colour: str) -> np.ndarray:
    """Parse ``#rrggbb`` (or ``#rgb``) into sRGB in [0, 1]."""
    text = colour.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ValueError(f"{colour!r} is not a hex colour like '#1f3a93'")
    try:
        value = int(text, 16)
    except ValueError as error:
        raise ValueError(f"{colour!r} is not a hex colour like '#1f3a93'") from error
    return np.array([(value >> 16) & 255, (value >> 8) & 255, value & 255]) / 255.0


def xyz_to_lab(xyz: np.ndarray, white_point: np.ndarray | None = None) -> np.ndarray:
    """CIE 1976 L*a*b*, in which equal distances are roughly equal perceived differences.

    Accepts one colour or any stack of them, the last axis being the three components.
    """
    white_point = D65_WHITE_POINT_XYZ if white_point is None else white_point
    ratio = np.asarray(xyz, dtype=float) / white_point

    # The cube root is replaced by a straight line near black, where it would otherwise
    # have an infinite slope.
    linear_segment_limit = (6 / 29) ** 3
    transformed = np.where(
        ratio > linear_segment_limit,
        np.cbrt(np.maximum(ratio, 0.0)),
        ratio / (3 * (6 / 29) ** 2) + 4 / 29,
    )
    x, y, z = (transformed[..., index] for index in range(3))
    return np.stack([116 * y - 16, 500 * (x - y), 200 * (y - z)], axis=-1)


def lab_to_xyz(lab: np.ndarray, white_point: np.ndarray | None = None) -> np.ndarray:
    """CIE XYZ from CIELAB -- the inverse of :func:`xyz_to_lab`, one colour or a stack."""
    white_point = D65_WHITE_POINT_XYZ if white_point is None else white_point
    lab = np.asarray(lab, dtype=float)
    lightness, green_red, blue_yellow = (lab[..., index] for index in range(3))

    y = (lightness + 16) / 116
    x = y + green_red / 500
    z = y - blue_yellow / 200

    linear_segment_limit = 6 / 29
    ratios = np.stack([x, y, z], axis=-1)
    return (
        np.where(
            ratios > linear_segment_limit,
            ratios**3,
            3 * linear_segment_limit**2 * (ratios - 4 / 29),
        )
        * white_point
    )


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """The display colour of a CIELAB coordinate, as sRGB in [0, 1]."""
    return xyz_to_srgb(lab_to_xyz(lab))


def spectrum_to_lab(spectrum: Spectrum) -> np.ndarray:
    """CIELAB coordinates of a reflecting surface under D65."""
    return xyz_to_lab(spectrum_to_xyz(spectrum))


def srgb_to_lab(srgb: np.ndarray) -> np.ndarray:
    """CIELAB coordinates of a display colour given as sRGB in [0, 1]."""
    return xyz_to_lab(srgb_to_xyz(srgb))


def delta_e_1976(first_lab: np.ndarray, second_lab: np.ndarray) -> float:
    """Straight-line distance between two colours in CIELAB.

    Crude as a perceptual measure, but smooth and cheap, which is what an optimiser
    wants. This is the library's default matching objective.
    """
    return float(np.linalg.norm(np.asarray(first_lab) - np.asarray(second_lab)))


def delta_e_2000(first_lab: np.ndarray, second_lab: np.ndarray) -> float:
    """CIEDE2000 colour difference, the current best model of perceived difference.

    Implemented from the formulation in Sharma, Wu and Dalal, "The CIEDE2000
    Color-Difference Formula: Implementation Notes, Supplementary Test Data, and
    Mathematical Observations" (2005), and pinned against that paper's 34-pair test set
    in the test suite. A value near 1 is about one just-noticeable difference.

    Reported on every Recipe, but not used as the optimiser's objective -- its hue-angle
    arithmetic has discontinuities that gradient-based solvers handle badly.
    """
    lightness_1, green_red_1, blue_yellow_1 = (float(v) for v in first_lab)
    lightness_2, green_red_2, blue_yellow_2 = (float(v) for v in second_lab)

    chroma_1 = np.hypot(green_red_1, blue_yellow_1)
    chroma_2 = np.hypot(green_red_2, blue_yellow_2)
    mean_chroma = (chroma_1 + chroma_2) / 2

    # Stretch the a* axis for low-chroma colours, where CIELAB is least uniform.
    stretch = 0.5 * (1 - np.sqrt(mean_chroma**7 / (mean_chroma**7 + 25.0**7)))
    stretched_a_1 = (1 + stretch) * green_red_1
    stretched_a_2 = (1 + stretch) * green_red_2

    stretched_chroma_1 = np.hypot(stretched_a_1, blue_yellow_1)
    stretched_chroma_2 = np.hypot(stretched_a_2, blue_yellow_2)

    hue_1 = _hue_angle_degrees(stretched_a_1, blue_yellow_1)
    hue_2 = _hue_angle_degrees(stretched_a_2, blue_yellow_2)

    delta_lightness = lightness_2 - lightness_1
    delta_chroma = stretched_chroma_2 - stretched_chroma_1

    chroma_product = stretched_chroma_1 * stretched_chroma_2
    if chroma_product == 0:
        delta_hue = 0.0
    else:
        delta_hue = hue_2 - hue_1
        if delta_hue > 180:
            delta_hue -= 360
        elif delta_hue < -180:
            delta_hue += 360
    delta_hue_distance = 2 * np.sqrt(chroma_product) * np.sin(np.radians(delta_hue) / 2)

    mean_lightness = (lightness_1 + lightness_2) / 2
    mean_stretched_chroma = (stretched_chroma_1 + stretched_chroma_2) / 2

    if chroma_product == 0:
        mean_hue = hue_1 + hue_2
    else:
        hue_difference = abs(hue_1 - hue_2)
        hue_sum = hue_1 + hue_2
        if hue_difference <= 180:
            mean_hue = hue_sum / 2
        elif hue_sum < 360:
            mean_hue = (hue_sum + 360) / 2
        else:
            mean_hue = (hue_sum - 360) / 2

    # Four empirical corrections, each fixing a known distortion of CIELAB.
    hue_weighting = (
        1
        - 0.17 * np.cos(np.radians(mean_hue - 30))
        + 0.24 * np.cos(np.radians(2 * mean_hue))
        + 0.32 * np.cos(np.radians(3 * mean_hue + 6))
        - 0.20 * np.cos(np.radians(4 * mean_hue - 63))
    )
    lightness_scale = 1 + (0.015 * (mean_lightness - 50) ** 2) / np.sqrt(
        20 + (mean_lightness - 50) ** 2
    )
    chroma_scale = 1 + 0.045 * mean_stretched_chroma
    hue_scale = 1 + 0.015 * mean_stretched_chroma * hue_weighting

    # The rotation term, which corrects the badly distorted blue region.
    rotation = (
        -2
        * np.sqrt(
            mean_stretched_chroma**7 / (mean_stretched_chroma**7 + 25.0**7)
        )
        * np.sin(np.radians(60 * np.exp(-(((mean_hue - 275) / 25) ** 2))))
    )

    lightness_term = delta_lightness / lightness_scale
    chroma_term = delta_chroma / chroma_scale
    hue_term = delta_hue_distance / hue_scale
    return float(
        np.sqrt(
            lightness_term**2
            + chroma_term**2
            + hue_term**2
            + rotation * chroma_term * hue_term
        )
    )


def _hue_angle_degrees(green_red: float, blue_yellow: float) -> float:
    """Hue angle in [0, 360), with the CIEDE2000 convention that 0 and 0 gives 0."""
    if green_red == 0 and blue_yellow == 0:
        return 0.0
    angle = np.degrees(np.arctan2(blue_yellow, green_red))
    return float(angle + 360 if angle < 0 else angle)
