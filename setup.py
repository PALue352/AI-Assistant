# AI_Assistant/setup.py
import os
from setuptools import setup, find_packages
from dotenv import load_dotenv

load_dotenv()  # Load .env file
VERSION = '0.1.0'
LONG_DESCRIPTION = "..."
DEFAULT_INSTALL_DIR = os.getenv('AI_INSTALL_DIR', '{userdocs}\\AI_Assistant')  # Inno Setup placeholder

setup(
    name="AI_Assistant",
    version=VERSION,
    author="Your Name",
    author_email="your.email@example.com",
    description="A local AI assistant with modular capabilities",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AI_Assistant",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12.8',
    setup_requires=['pip', 'python-dotenv'],
    data_files=[
        ('', ['requirements.txt', 'install_dependencies.bat', 'AI_Architecture_Document.md', '.env']),
        (os.path.join(DEFAULT_INSTALL_DIR, 'ai_assistant/core/training_data'), [
            'ai_assistant/core/training_data/ethical_training_data.json',
            'ai_assistant/core/training_data/medical_kb.json',
            'ai_assistant/core/training_data/knowledge_base.json',
            'ai_assistant/core/training_data/patent_templates.json',
            'ai_assistant/core/training_data/cognitive_kb.json',
            'ai_assistant/core/training_data/financial_kb.json'
            'ai_assistant/core/training_data/pr_kb.json'
            'ai_assistant/core/training_data/protection_kb.json'
        ]),
    ],
    scripts=['install_dependencies.bat'],
    entry_points={
        'console_scripts': [
            'ai_assistant=ai_assistant.gui.gui_interface:GUIInterface.run',
        ],
    },
)