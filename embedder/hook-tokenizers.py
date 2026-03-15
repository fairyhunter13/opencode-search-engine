# PyInstaller runtime hook for tokenizers
# This runs before any application code is loaded

import os

# Disable tokenizers parallelism to prevent deadlock on macOS
# This must be set before any module imports the tokenizers library
os.environ["TOKENIZERS_PARALLELISM"] = "false"
