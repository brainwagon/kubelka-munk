"""Standard colorimetric data, tabulated on this library's wavelength grid.

These are the CIE 1931 2-degree standard observer colour matching functions and the
relative spectral power distribution of standard illuminant D65, decimated to
380-730 nm in 10 nm steps.

The numbers are transcribed from the Colour and Vision Research Laboratories
tabulations, which republish the CIE data:

    colour matching functions  http://www.cvrl.org/database/data/cmfs/ciexyz31.csv
    illuminant D65             http://www.cvrl.org/database/data/cie/Illuminantd65.csv

The source tables are finer than this grid (5 nm for the observer, 1 nm for D65);
values here are taken at the sample points, not resampled or smoothed. Decimation
to 10 nm is standard practice for reflectance work -- reflectance curves of paint
have no structure fine enough to need more.

Do not edit these by hand.
"""

# Wavelength in nanometres, then x-bar, y-bar and z-bar of the 1931 2-degree observer.
CIE_1931_2_DEGREE_OBSERVER: dict[int, tuple[float, float, float]] = {
    380: (0.001368, 0.000039, 0.006450),
    390: (0.004243, 0.000120, 0.020050),
    400: (0.014310, 0.000396, 0.067850),
    410: (0.043510, 0.001210, 0.207400),
    420: (0.134380, 0.004000, 0.645600),
    430: (0.283900, 0.011600, 1.385600),
    440: (0.348280, 0.023000, 1.747060),
    450: (0.336200, 0.038000, 1.772110),
    460: (0.290800, 0.060000, 1.669200),
    470: (0.195360, 0.090980, 1.287640),
    480: (0.095640, 0.139020, 0.812950),
    490: (0.032010, 0.208020, 0.465180),
    500: (0.004900, 0.323000, 0.272000),
    510: (0.009300, 0.503000, 0.158200),
    520: (0.063270, 0.710000, 0.078250),
    530: (0.165500, 0.862000, 0.042160),
    540: (0.290400, 0.954000, 0.020300),
    550: (0.433450, 0.994950, 0.008750),
    560: (0.594500, 0.995000, 0.003900),
    570: (0.762100, 0.952000, 0.002100),
    580: (0.916300, 0.870000, 0.001650),
    590: (1.026300, 0.757000, 0.001100),
    600: (1.062200, 0.631000, 0.000800),
    610: (1.002600, 0.503000, 0.000340),
    620: (0.854450, 0.381000, 0.000190),
    630: (0.642400, 0.265000, 0.000050),
    640: (0.447900, 0.175000, 0.000020),
    650: (0.283500, 0.107000, 0.000000),
    660: (0.164900, 0.061000, 0.000000),
    670: (0.087400, 0.032000, 0.000000),
    680: (0.046770, 0.017000, 0.000000),
    690: (0.022700, 0.008210, 0.000000),
    700: (0.011359, 0.004102, 0.000000),
    710: (0.005790, 0.002091, 0.000000),
    720: (0.002899, 0.001047, 0.000000),
    730: (0.001440, 0.000520, 0.000000),
}

# Wavelength in nanometres, then relative spectral power of illuminant D65.
ILLUMINANT_D65: dict[int, float] = {
    380: 49.9755,
    390: 54.6482,
    400: 82.7549,
    410: 91.4860,
    420: 93.4318,
    430: 86.6823,
    440: 104.8650,
    450: 117.0080,
    460: 117.8120,
    470: 114.8610,
    480: 115.9230,
    490: 108.8110,
    500: 109.3540,
    510: 107.8020,
    520: 104.7900,
    530: 107.6890,
    540: 104.4050,
    550: 104.0460,
    560: 100.0000,
    570: 96.3342,
    580: 95.7880,
    590: 88.6856,
    600: 90.0062,
    610: 89.5991,
    620: 87.6987,
    630: 83.2886,
    640: 83.6992,
    650: 80.0268,
    660: 80.2146,
    670: 82.2778,
    680: 78.2842,
    690: 69.7213,
    700: 71.6091,
    710: 74.3490,
    720: 61.6040,
    730: 69.8856,
}
