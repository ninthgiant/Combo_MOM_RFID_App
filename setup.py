from setuptools import setup

APP = ['COMBO_Viewer_v02b.py']

DATA_FILES = []

OPTIONS = {
    'argv_emulation': False,
    'packages': ['pandas', 'numpy'],
    'includes': ['cmath', 'math'],   # ← FIX ADDED HERE
    'plist': {
        'CFBundleName': 'COMBO Viewer',
        'CFBundleDisplayName': 'COMBO Viewer',
        'CFBundleIdentifier': 'com.bobmauck.combo.viewer',
        'CFBundleVersion': '1.0',
        'CFBundleShortVersionString': '1.0',
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
