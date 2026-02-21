from setuptools import setup, find_packages

setup(
    name="nanoathens",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],
    extras_require={
        "medgemma": ["torch>=2.0", "transformers>=4.40", "huggingface_hub"],
        "retrieval": ["rank_bm25", "faiss-cpu", "sentence-transformers"],
        "all": ["nanoathens[medgemma,retrieval]"],
    },
)
