"""
Modified processor wrapper to disable image cutting for training
"""
from src.owl3.processing_mplugowl3 import mPLUGOwl3ImageProcessor, mPLUGOwl3Processor

class mPLUGOwl3ImageProcessorNoCut(mPLUGOwl3ImageProcessor):
    """Image processor with cutting disabled by default"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable cutting
        self.cut_enable = False

class mPLUGOwl3ProcessorNoCut(mPLUGOwl3Processor):
    """Processor that uses no-cut image processor"""
    pass

def create_processor_no_cut(tokenizer, image_size=378):
    """Create processor with cutting disabled"""
    image_processor = mPLUGOwl3ImageProcessorNoCut(image_size=image_size)
    processor = mPLUGOwl3ProcessorNoCut(
        image_processor=image_processor,
        tokenizer=tokenizer,
        inference_mode=False  # Set to False for training to include assistant responses
    )
    return processor
