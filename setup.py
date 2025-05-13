from setuptools import setup, find_packages

setup(
    name="echelon",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "feedparser",
        "requests",
        "python-dotenv",
        "stix2",
        "joblib",
        "flask",
        "gunicorn",
        "schedule"
    ],
)
